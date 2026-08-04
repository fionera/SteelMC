//! The per-holder generation drive.
//!
//! One admitted holder, one permit, one tokio task, and inside it a loop of
//! fused runs. The task advances *its own* chunk: it resolves the halo the next
//! run needs, dispatches when that halo is ready, and otherwise gives the permit
//! back and leaves nothing but a waiter registration behind.
//!
//! That last part is the whole design. The task model this replaces scheduled a
//! chunk's entire accumulated dependency halo -- up to 529 chunks at radius 11 --
//! at every status layer, and `join_all`ed the result. Three attempts to make a
//! task simply *wait* instead failed:
//!
//! * A parked task that keeps its admission slot deadlocks, because
//!   `run_generation_tasks_b` returns early at zero available slots and the
//!   chunk being waited on can never be admitted.
//! * Releasing the slot but keeping the task removes the only bound: every chunk
//!   ends up with a live parked task retaining a 529-entry halo, which also pins
//!   those holders against unloading. Measured 13.6 GB RSS against 8.4 GB
//!   normal, 30% CPU across 260 sleeping threads, and no 601x601 repetition
//!   completing in fifteen minutes.
//!
//! So a blocked chunk here holds no task, no permit and no halo -- only its
//! drive word and the registrations it filed. The halo is resolved when work is
//! about to run and dropped when it is not.
//!
//! Deadlock freedom is a property of the tables rather than of this code:
//! `chunk_pyramid::run_plan_tests` proves that every cross-chunk requirement of
//! a run is for a status strictly *below* the one the run produces (A1), and
//! that a halo chunk is always allowed to reach what its neighbours need of it
//! (A3). Wait edges therefore always point down a twelve-element chain, and
//! `Empty` needs nothing, so some holder is always dispatchable.

use std::{
    iter, mem,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use steel_utils::ChunkPos;

use crate::chunk::{
    chunk_holder::{ChunkHolder, GENERATION_DRIVE_COUNTERS},
    chunk_map::{ChunkMap, GENERATION_FANOUT, StalledGeneration},
    chunk_pyramid::{GENERATION_PYRAMID, RUN_PLANS, RunPlan, max_ring_requirement},
    generation_drive::DecOutcome,
    static_cache_2d::StaticCache2D,
    status::ChunkStatus,
};

/// Why a drive gave up on its holder until something re-arms it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StallReason {
    /// A cell of the halo has no holder. The position carries a ticket level but
    /// no `chunks` entry -- most often a revival deferred behind save
    /// preparation, which takes several epochs.
    HaloMiss(ChunkPos),
    /// `apply_step` refused the work: the status stopped being allowed between
    /// the evaluation and the dispatch.
    Refused,
    /// The job ran and failed -- a storage acquire or load error, or a rayon
    /// panic mapped to `None` by the join wrapper.
    JobFailed,
}

impl StallReason {
    /// Index into [`GenerationDriveCounters::stalls_by_reason`].
    ///
    /// [`GenerationDriveCounters::stalls_by_reason`]: crate::chunk::chunk_holder::GenerationDriveCounters::stalls_by_reason
    const fn index(self) -> usize {
        match self {
            Self::HaloMiss(_) => 0,
            Self::Refused => 1,
            Self::JobFailed => 2,
        }
    }
}

/// Consecutive idle epochs before a parked backlog is reported as a stall.
///
/// The idleness test is a sample, not a synchronisation: `run_generation_tasks_b`
/// moves the arrivals into a local before it takes the selection lock, so a
/// sample landing in that window sees an empty inbox, an empty queue and no runs
/// while a whole batch is in flight. One idle epoch therefore proves nothing,
/// and the bound has to be large enough that hitting that window on every epoch
/// in a row is not a thing that happens.
///
/// 64 is about 0.3 s at the 5 ms mean epoch measured over a 601x601
/// pregeneration, against a stall that never resolves -- the run this exists for
/// produced no output for fifteen minutes. The only legitimately quiet window
/// that is anywhere near that long is the stalled-drive backoff, which reaches
/// 640 ms and is excluded outright by `generation_pipeline_is_idle`.
pub(super) const STALL_WATCHDOG_EPOCHS: u32 = 64;

/// Parked holders one report names.
///
/// A stall takes out whole regions at once -- every chunk whose halo contains
/// the stuck one -- so the list is a sample and not an inventory. Thirty-two
/// entries at one line each is what fits in a terminal alongside the count of
/// how many there really are.
const STALL_WATCHDOG_DUMP: usize = 32;

/// First revival delay after a stall.
const STALL_BACKOFF_BASE: Duration = Duration::from_millis(5);
/// Doublings the backoff may take, i.e. a ceiling of `BASE << CAP`.
const STALL_BACKOFF_DOUBLINGS: u32 = 7;

/// Drives `holder` until it can make no further progress on this admission.
///
/// Returning drops the caller's permit, which is what makes a parked or stalled
/// holder free rather than merely idle.
pub(super) async fn drive_holder_generation(holder: Arc<ChunkHolder>, chunk_map: Arc<ChunkMap>) {
    let mut guard = DriveDropGuard::new(&chunk_map, holder.get_pos());
    drive_runs(&holder, &chunk_map).await;
    guard.finish();
}

/// The run loop. Split out so [`DriveDropGuard`] wraps every exit from it
/// without each of the eight of them having to remember to disarm it.
async fn drive_runs(holder: &Arc<ChunkHolder>, chunk_map: &Arc<ChunkMap>) {
    if holder.begin_generation_run().is_none() {
        // The queue entry was stale: the holder was re-armed, abandoned, or is
        // already running. Touching the chunk now would give it two drivers.
        return;
    }

    // Carried across runs, dropped the moment this drive stops running. A
    // *parked* holder must retain nothing -- that is what made the previous
    // attempt reach 13.6 GB and stop completing -- but a *running* one may, and
    // the layer-walking model it replaces held a wider halo for longer.
    let mut cached_halo: Option<CachedHalo> = None;

    loop {
        // Snapshotting the ticket *before* reading anything else is the driver
        // half of the Dekker pairing in `generation_drive`: an armer stores the
        // new allowance and then bumps this word, so a driver that went on to
        // read a stale allowance is guaranteed to lose its exit and come round
        // again.
        let ticket = holder.generation_run_ticket();

        // EVALUATE. No awaits in here: everything below is a plain read, so
        // nothing can move under the decision between the reads that make it.
        let Some(allowed) = holder.highest_allowed_status() else {
            if exit_idle(holder, ticket) {
                return;
            }
            continue;
        };
        if !holder.is_save_lifecycle_active() {
            // The holder is being unloaded and its data is about to be
            // snapshotted for saving. Starting a run now would race the save.
            if exit_idle(holder, ticket) {
                return;
            }
            continue;
        }
        let next = match holder.published_status() {
            None => ChunkStatus::Empty,
            Some(published) if published >= allowed => {
                if exit_idle(holder, ticket) {
                    return;
                }
                continue;
            }
            Some(published) => {
                let Some(next) = published.next() else {
                    // `published < allowed` and `allowed` is a real status, so
                    // `published` is not the last one.
                    unreachable!("a status below the allowance always has a successor");
                };
                next
            }
        };
        // Keyed by the status the run *starts* at, not by the run, so a holder
        // sitting in the middle of a run -- loaded from disk at, say, `Surface`,
        // or left there when a fused run published `Noise` and had the rest of
        // its claims rolled back -- resumes at exactly the next status.
        // Restarting it at the run's first status is what `claim_status_work`
        // panics on.
        let plan = &RUN_PLANS[next.get_index()];

        let resolution = match cached_halo.as_mut() {
            Some(cached) if cached.radius >= plan.halo_radius as i32 => {
                recheck_cached(cached, plan)
            }
            _ => {
                let resolved = resolve_and_check(chunk_map, holder.get_pos(), plan);
                if let HaloResolution::Ready(ref cache) = resolved {
                    cached_halo = Some(CachedHalo::new(holder.get_pos(), plan, Arc::clone(cache)));
                }
                resolved
            }
        };

        match resolution {
            HaloResolution::Ready(halo) => {
                let step = GENERATION_PYRAMID.get_step_to(plan.first);
                let Some(ready) = holder.apply_step(
                    step,
                    chunk_map,
                    &halo,
                    Arc::clone(&chunk_map.generation_pool),
                ) else {
                    if stall(chunk_map, holder, ticket, StallReason::Refused) {
                        return;
                    }
                    continue;
                };
                // `apply_step` has already cloned the halo into the job it
                // spawned, so this reference is spare. `cached_halo` still holds
                // one for the runs after this: the holders stay pinned either
                // way while this drive is running, and re-resolving them per run
                // is what made this dispatcher slower than the model it
                // replaces.
                drop(halo);
                match ready.await {
                    Some(()) => holder.clear_stall_backoff(),
                    // Never `continue`: the failure modes behind `None` are a
                    // storage acquire/load error and a rayon panic, neither of
                    // which the next poll fixes. Looping would be an unbounded
                    // hot retry holding an admission permit.
                    None => return stall_failed_job(chunk_map, holder),
                }
            }
            // The permit is released by returning, before anything waits for
            // this holder -- that is the difference between this design and the
            // three that deadlocked.
            HaloResolution::Unmet(unmet) => {
                // Freed before the park handshake, not after: once registered,
                // this holder can be woken and re-admitted by another thread
                // while this stack is still unwinding.
                cached_halo = None;
                match park(holder, ticket, &unmet) {
                    ParkOutcome::Parked => return,
                    ParkOutcome::ReEvaluate => {}
                }
            }
            HaloResolution::Missing(pos) => {
                if stall(chunk_map, holder, ticket, StallReason::HaloMiss(pos)) {
                    return;
                }
            }
        }
        // Round again on the same task and the same permit. A chunk that never
        // blocks therefore makes one admission for all six of its runs, not six.
    }
}

/// Retires the drive. `false` means the exit was refused and the caller must
/// re-evaluate.
fn exit_idle(holder: &Arc<ChunkHolder>, ticket: u64) -> bool {
    if !holder.to_idle(ticket) {
        return false;
    }

    // The re-arm side stores the new allowance and then reads this drive word;
    // this side writes the drive word and then reads the allowance. Both
    // accesses are `SeqCst` on both sides, which is what gives the Store-Load
    // edge -- with `AcqRel` both sides may legally read the other's pre-store
    // value, each concludes the other will do the work, and the chunk sits below
    // its allowed status forever.
    //
    // The lifecycle is re-tested for a narrower reason: the unload path begins
    // unloading, withdraws the drive and only *then* clears the allowance, so a
    // drive that exits inside that window still reads a status it is allowed to
    // reach. Re-arming on it would queue the holder straight back into an
    // evaluation that can only exit again, spinning until the clear lands. A
    // revival re-arms the holder itself, through the same scheduling epoch that
    // reactivates it, so nothing is lost by declining here.
    if holder.is_save_lifecycle_active() && holder.needs_generation() && holder.arm() {
        holder.queue_for_generation();
    }
    true
}

/// Stalls after a job that ran and failed, against whatever ticket the drive
/// word currently carries.
///
/// The other two stall sites answer a refusal by re-evaluating, and that costs
/// one halo rescan. This one cannot afford it: re-evaluating after a failed job
/// resolves the same plan and re-dispatches the step that just failed -- another
/// storage acquire, another `tracing::error!`, another rayon job -- so any
/// source of epoch bumps during a failing run turns the stall this code exists
/// to take into the hot retry it exists to prevent. Deferring the re-evaluation
/// to the backoff revival costs at most `STALL_BACKOFF_BASE << attempts`, and
/// `to_stalled` bumps the epoch itself, so nothing holding the spent ticket can
/// act on the holder in the meantime.
///
/// The retry terminates for the same reason any compare-exchange loop does:
/// only the driver leaves `Running`, so the phase never stops accepting and the
/// only way round again is another writer landing between the read and the CAS.
fn stall_failed_job(chunk_map: &Arc<ChunkMap>, holder: &Arc<ChunkHolder>) {
    while !stall(
        chunk_map,
        holder,
        holder.generation_run_ticket(),
        StallReason::JobFailed,
    ) {}
}

/// Parks the drive until something it needs is published. `false` means the exit
/// was refused and the caller must re-evaluate.
fn stall(
    chunk_map: &Arc<ChunkMap>,
    holder: &Arc<ChunkHolder>,
    ticket: u64,
    reason: StallReason,
) -> bool {
    if !holder.to_stalled(ticket) {
        return false;
    }
    chunk_map.record_generation_stall(holder, reason);
    true
}

/// A halo resolved for one run and reused by the runs after it.
struct CachedHalo {
    radius: i32,
    cache: Arc<StaticCache2D<Arc<ChunkHolder>>>,
    /// The cells a later run can still be blocked by, outermost first.
    ///
    /// Invariant: every entry has `!is_settled(distance, seen)`, and every cell
    /// of the square that is *not* here has been seen at or past
    /// [`max_ring_requirement`] for its distance. Both halves are maintained at
    /// the two points a `seen` is written -- [`CachedHalo::new`] and the read in
    /// [`recheck_cached`] -- and nowhere else.
    pending: Vec<PendingCell>,
}

/// A halo cell a re-check may still have to read.
struct PendingCell {
    holder: Arc<ChunkHolder>,
    /// Chebyshev distance from the drive's chunk, which is what selects this
    /// cell's requirement out of a plan's ring. Constant for the life of the
    /// cache: the square is centred on that chunk and the drive never moves.
    distance: usize,
    /// The highest status this cell has been *observed* at -- either a read of
    /// `published_status`, or the requirement a resolve proved it past. Never an
    /// inference from a previous run's requirement being met, which is the trap:
    /// "satisfied" is only meaningful against one requirement, and a later run
    /// asks more of the same cell.
    ///
    /// A published status only ever rises -- `raise_published_status` drops a
    /// raise that does not exceed the current word, nothing resets it, and the
    /// holder behind an `Arc` this drive holds cannot be replaced -- so this
    /// stays a lower bound on the cell's real status for as long as the cache
    /// lives. That is the whole licence for skipping a cell: `seen >= required`
    /// implies `published >= required` now and for the rest of the drive.
    seen: Option<ChunkStatus>,
}

/// Whether a cell seen at `seen` can be dropped from the walk for good.
///
/// [`max_ring_requirement`] is the most *any* run's ring asks at this distance,
/// so a cell at or past it satisfies every requirement any remaining run of this
/// drive can put on it. `None` there names a distance no ring mentions -- the
/// halo-only outer window the `Light` run reads opportunistically -- and
/// `None >= None` retires those cells without ever reading one.
fn is_settled(distance: usize, seen: Option<ChunkStatus>) -> bool {
    // `Option`'s ordering puts `None` below every status, which is exactly the
    // meaning both sides carry here: unpublished, and unrequired.
    seen >= max_ring_requirement(distance)
}

impl CachedHalo {
    /// Wraps a halo `resolve_and_check` just returned `Ready` for.
    ///
    /// The cells are seeded from `plan.ring` rather than from the statuses the
    /// resolve read. A `Ready` answer *is* the proof that every cell at a
    /// distance is at or past what the ring asks there, so the requirement is a
    /// sound lower bound, and it is one the pass does not have to carry out.
    /// It is a weaker bound than the reading -- a neighbour that had already run
    /// ahead is recorded as merely meeting the ring -- which costs at most one
    /// re-read of that cell on the next run and settles it then.
    fn new(center: ChunkPos, plan: &RunPlan, cache: Arc<StaticCache2D<Arc<ChunkHolder>>>) -> Self {
        let radius = plan.halo_radius as i32;
        let mut pending = Vec::new();
        // Descending, like `resolve_and_check`: `recheck_cached` walks this list
        // in order and `fanout_selection` takes the tail of what it emits.
        for distance in (0..=radius).rev() {
            let seen = plan.ring.get(distance as usize);
            if is_settled(distance as usize, seen) {
                // The whole ring at once, without visiting a cell. This is where
                // a radius-8 halo sheds its 280 outer cells: it only resolved
                // because they were all at `StructureStarts`, and that is the
                // most any plan asks out there.
                continue;
            }
            for (x, z) in ring_cells(center, distance) {
                let Some(holder) = cache.try_get(x, z) else {
                    unreachable!("the halo was just resolved out to this radius");
                };
                pending.push(PendingCell {
                    holder: Arc::clone(holder),
                    distance: distance as usize,
                    seen,
                });
            }
        }

        Self {
            radius,
            cache,
            pending,
        }
    }
}

/// What one pass over the run's square found.
enum HaloResolution {
    /// Every cell is present and every required cell is at or past what the run
    /// needs of it.
    Ready(Arc<StaticCache2D<Arc<ChunkHolder>>>),
    /// At least one required cell is behind. The halo is *not* carried along --
    /// see [`resolve_and_check`].
    Unmet(Vec<UnmetDependency>),
    /// A cell has no holder at all.
    Missing(ChunkPos),
}

/// One neighbour a run is waiting on, and what it is waiting for.
struct UnmetDependency {
    holder: Arc<ChunkHolder>,
    required: ChunkStatus,
}

/// Resolves the run's halo and checks its ring in a single pass.
///
/// One pass, not a probe followed by a resolve. Two passes cost twice the map
/// lookups on the hottest path there is -- 289 cells for a radius-8 ring, per
/// run, per chunk -- and they open a window the single pass does not have, where
/// the probe finds a cell present and the resolve finds it gone.
///
/// Cells are visited in descending Chebyshev distance because that is the order
/// the answers arrive in: the outer rings are the ones whose holders may not
/// exist yet, and `Missing` aborts immediately.
///
/// Cells beyond the ring are halo-only. They are collected and never gate, which
/// is exactly how the `Light` run keeps its opportunistic radius-2 outer window:
/// `run_light_stage` reads that ring with `try_get`, so a short halo does not
/// fail, it silently lights the chunk differently.
///
/// An `Unmet` answer retains *nothing*. Holding the halo across a park is what
/// made the third attempt cost 13.6 GB: a parked holder would pin up to 529
/// neighbours against unloading for as long as it waited.
fn resolve_and_check(chunk_map: &ChunkMap, center: ChunkPos, plan: &RunPlan) -> HaloResolution {
    let radius = plan.halo_radius as i32;
    let size = radius * 2 + 1;
    let min_x = center.0.x - radius;
    let min_z = center.0.y - radius;

    let mut halo: Vec<Option<Arc<ChunkHolder>>> = vec![None; (size * size) as usize];
    let mut unmet: Vec<UnmetDependency> = Vec::new();

    for distance in (0..=radius).rev() {
        let required = plan.ring.get(distance as usize);
        for (x, z) in ring_cells(center, distance) {
            let pos = ChunkPos::new(x, z);
            // Once any cell is behind, this pass is going to park, and the only
            // holders it still needs are the ones it will actually wait on.
            // Cloning the rest would be a cross-core refcount increment followed
            // immediately by a decrement -- two coherence round-trips on a line
            // other workers are also writing, and by stall weight the most
            // expensive thing this function does.
            //
            // So the decision to clone is made inside the guard, where the
            // status is already in hand, rather than cloning first and sorting
            // it out afterwards. One lookup still takes both the reference and
            // the status off the same holder under the same read guard; reading
            // the status through a second lookup would double the cost of the
            // pass.
            let parking = !unmet.is_empty();
            let cell = chunk_map.chunks.read_sync(&pos, |_, holder| {
                let published = holder.published_status();
                let behind = required.is_some_and(|required| {
                    published.is_none_or(|published| published < required)
                });
                let retained = (!parking || behind).then(|| Arc::clone(holder));
                (retained, behind)
            });
            let Some((retained, behind)) = cell else {
                return HaloResolution::Missing(pos);
            };

            if behind {
                if unmet.is_empty() {
                    // First unmet cell: this pass is going to park, so free the
                    // halo now rather than at the end of the pass.
                    halo = Vec::new();
                }
                unmet.push(UnmetDependency {
                    holder: retained.expect("a cell that is behind is always retained"),
                    required: required.expect("only a cell with a requirement can be behind"),
                });
                continue;
            }

            if let Some(holder) = retained {
                let index = ((z - min_z) * size + (x - min_x)) as usize;
                halo[index] = Some(holder);
            }
        }
    }

    if !unmet.is_empty() {
        return HaloResolution::Unmet(unmet);
    }

    let cells = halo
        .into_iter()
        .map(|cell| cell.expect("every cell of the square is visited exactly once"))
        .collect();
    HaloResolution::Ready(Arc::new(StaticCache2D::from_row_major(
        min_x, min_z, size, cells,
    )))
}

/// Re-checks a halo already resolved for this drive, without touching the map.
///
/// A drive advances a chunk through up to six runs, and the runs' halos nest:
/// radius 0, then 8, then 2. Resolving one per run costs 894 `chunks` lookups
/// and `Arc` clones per chunk, against 529 for the single halo the layer-walking
/// model built -- which is why the first version of this dispatcher was 8.3%
/// slower despite doing strictly less scheduling work. Profiling put ~4% of the
/// machine in this pass and another ~4.5% in the epoch pinning underneath it.
///
/// The holders themselves do not change between runs -- a position keeps its
/// `Arc` across revival, and one cannot be replaced while this drive holds a
/// reference -- so only their statuses need re-reading, and those come straight
/// off the cached `Arc`s.
///
/// Caching the square only removed the lookups, not the walk: re-reading all 289
/// statuses of a radius-8 halo on each of the three runs that reuse it left this
/// pass at 2.98% of a 601x601 pregeneration. It does not have to look at a cell
/// twice. `published_status` is monotone, so a cell seen at or past what the run
/// asks of it is met whatever it does next, and [`CachedHalo::pending`] carries
/// exactly the cells that have not yet been seen that high -- nine of the 289,
/// once the halo has resolved. What it carries is the *status* each cell reached
/// rather than a "satisfied" flag: the requirement rises between runs, so a flag
/// set against one run's ring would clear cells the next run is still waiting
/// for.
fn recheck_cached(cached: &mut CachedHalo, plan: &RunPlan) -> HaloResolution {
    let mut unmet: Vec<UnmetDependency> = Vec::new();

    // `retain_mut` keeps the order `CachedHalo::new` built, which is the
    // descending-distance order `resolve_and_check` walks in: the fan-out takes
    // the tail of `unmet`, and that is only the most constraining dependencies
    // if the emitted order is preserved.
    cached.pending.retain_mut(|cell| {
        let Some(required) = plan.ring.get(cell.distance) else {
            // Halo-only for *this* run. The cell has to stay in the cache for
            // the step to read -- that is how `Light` keeps its opportunistic
            // radius-2 window -- but it gates nothing here, so it is neither
            // read nor retired: a later run may still ask something of it.
            return true;
        };
        if cell.seen.is_some_and(|seen| seen >= required) {
            return true;
        }

        cell.seen = cell.holder.published_status();
        if cell.seen.is_none_or(|seen| seen < required) {
            unmet.push(UnmetDependency {
                holder: Arc::clone(&cell.holder),
                required,
            });
            return true;
        }
        !is_settled(cell.distance, cell.seen)
    });

    if unmet.is_empty() {
        HaloResolution::Ready(Arc::clone(&cached.cache))
    } else {
        HaloResolution::Unmet(unmet)
    }
}

/// The cells at exactly Chebyshev distance `distance` from `center`, each once.
fn ring_cells(center: ChunkPos, distance: i32) -> impl Iterator<Item = (i32, i32)> {
    let x = center.0.x;
    let z = center.0.y;
    // The two full rows, then the two columns with their corners already taken.
    // At distance zero the second row and both columns are empty, so the centre
    // is yielded once rather than four times.
    let rows = (-distance..=distance).flat_map(move |offset| {
        iter::once((x + offset, z - distance))
            .chain((distance != 0).then_some((x + offset, z + distance)))
    });
    let columns = ((-distance + 1)..distance)
        .flat_map(move |offset| [(x - distance, z + offset), (x + distance, z + offset)]);
    rows.chain(columns)
}

enum ParkOutcome {
    /// The drive is parked; the caller must return and release its permit.
    Parked,
    /// The park was refused because the run ticket is spent; re-evaluate.
    ReEvaluate,
}

/// The prefix of `unmet` a park may register waiters for, most constraining
/// first.
///
/// The park ends only when *every* registration it filed has fired, so which
/// `GENERATION_FANOUT` of the unmet set is chosen decides how many rounds the
/// holder takes. `resolve_and_check` walks rings outwards-in, and a run's ring
/// never rises with radius -- proved as a table property by
/// `a_runs_ring_never_rises_with_radius` in `chunk_pyramid` -- so `unmet` comes
/// back ordered from the *lowest* requirement to the highest, and the tail is
/// the constraining end. Taking it front-first would register exactly the
/// cells that clear earliest and never the ones that gate the run: a radius-8
/// `Noise` park would wake on 64 `StructureStarts` neighbours, rescan all 289
/// cells, and park again on the next-easiest 64 -- five rounds and five
/// admissions where one would do. Reversed, the wake is the last thing the run
/// was actually waiting for and the rescan usually dispatches.
fn fanout_selection(unmet: &[UnmetDependency]) -> impl Iterator<Item = &UnmetDependency> {
    unmet.iter().rev().take(*GENERATION_FANOUT)
}

/// Registers this holder as waiting on the dependencies it is behind on.
fn park(holder: &Arc<ChunkHolder>, ticket: u64, unmet: &[UnmetDependency]) -> ParkOutcome {
    let Some(epoch) = holder.park_begin(ticket) else {
        // The ticket is spent: an `arm` or an `abandon` landed while the halo
        // was being read, so the dependency set just collected predates the
        // change. Retrying the park at the new epoch would register waiters for
        // the halo the holder no longer has.
        return ParkOutcome::ReEvaluate;
    };

    for dependency in fanout_selection(unmet) {
        if !holder.park_on(&dependency.holder, dependency.required, epoch) {
            // The park is over -- something re-armed the holder mid-pass and
            // already owns its queue entry. The release below reports `Stale`
            // and queues nothing, which is correct: queueing here would hand out
            // a second entry for the same holder.
            break;
        }
    }

    // Releases the bias `park_begin` took. Until this point the park cannot end,
    // however many of its dependencies publish, which is what stops a
    // registration filed later in the pass from landing on a park that is
    // already over and stranding a count nothing will release.
    if holder.finish_dependency(epoch) == DecOutcome::Requeue {
        // Everything this pass meant to wait for published while it was
        // registering. Not a spin: published status is monotone, so the
        // re-admitted pass finds those cells met and dispatches.
        holder.queue_for_generation();
    }
    ParkOutcome::Parked
}

/// Logs and counts a drive future that was dropped before it retired its holder.
///
/// A dropped drive leaves the holder `Running` with nobody inside it: no exit
/// takes the ticket back, so nothing re-admits it, and any work claim it held
/// rolls back underneath a holder that will never be looked at again. That is
/// how the first attempt at this scheduler corrupted claims, and it is silent
/// otherwise.
struct DriveDropGuard {
    chunk_map: Arc<ChunkMap>,
    pos: ChunkPos,
    finished: bool,
}

impl DriveDropGuard {
    fn new(chunk_map: &Arc<ChunkMap>, pos: ChunkPos) -> Self {
        Self {
            chunk_map: Arc::clone(chunk_map),
            pos,
            finished: false,
        }
    }

    const fn finish(&mut self) {
        self.finished = true;
    }
}

impl Drop for DriveDropGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        GENERATION_DRIVE_COUNTERS
            .drive_futures_dropped
            .fetch_add(1, Ordering::Relaxed);
        if self.chunk_map.cancel_token.is_cancelled() {
            // Shutdown aborts the task tracker, which drops every in-flight
            // drive. Expected, and nothing is left to corrupt.
            tracing::debug!(chunk = ?self.pos, "Generation drive dropped during shutdown");
        } else {
            tracing::error!(
                chunk = ?self.pos,
                "Generation drive future dropped mid-run; the holder is stranded in the running \
                 phase and nothing will re-admit it",
            );
        }
    }
}

impl ChunkMap {
    /// Files a stalled holder for a later retry.
    ///
    /// The backoff is not decoration. A stall is usually caused by something
    /// that takes several scheduling epochs to clear -- a deferred revival
    /// leaves a position with a ticket level and no `chunks` entry -- and *every*
    /// holder whose halo contains that position stalls on it. Without a backoff
    /// each of them re-arms, burns an admission slot on a full halo rescan and
    /// stalls again, every epoch, which is a CPU livelock rather than a delay.
    fn record_generation_stall(&self, holder: &Arc<ChunkHolder>, reason: StallReason) {
        GENERATION_DRIVE_COUNTERS.stalls_by_reason[reason.index()].fetch_add(1, Ordering::Relaxed);
        let attempts = holder.record_stall();
        let doublings = (attempts - 1).min(STALL_BACKOFF_DOUBLINGS);
        let backoff = STALL_BACKOFF_BASE * (1u32 << doublings);
        tracing::trace!(
            chunk = ?holder.get_pos(),
            ?reason,
            attempts,
            ?backoff,
            "Generation drive stalled",
        );
        self.stalled_generation_drives
            .lock()
            .push(StalledGeneration {
                holder: Arc::downgrade(holder),
                reason,
                retry_after: Instant::now() + backoff,
            });
    }

    /// Re-arms the stalled holders whose backoff has expired.
    ///
    /// Bounded by the same budget shape as the other scheduling-epoch sweeps:
    /// this phase is serialized against generation-task creation, so its cost is
    /// latency rather than throughput, and whatever is not reached is carried to
    /// the next epoch. The unreached tail keeps its place at the *front* of the
    /// queue so consecutive epochs walk the whole list instead of re-examining
    /// the same prefix.
    pub(super) fn revive_stalled_generation_drives(&self, start: Instant) {
        /// How long one epoch may spend reviving stalled drives.
        const REVIVAL_BUDGET: Duration = Duration::from_millis(1);
        /// Entries examined between deadline checks.
        const BUDGET_CHECK_INTERVAL: usize = 64;

        let stalled = mem::take(&mut *self.stalled_generation_drives.lock());
        if stalled.is_empty() {
            return;
        }

        let now = Instant::now();
        let mut entries = stalled.into_iter();
        let mut deferred = Vec::new();
        let mut interrupted = None;
        for (examined, entry) in entries.by_ref().enumerate() {
            if examined.is_multiple_of(BUDGET_CHECK_INTERVAL)
                && examined != 0
                && start.elapsed() >= REVIVAL_BUDGET
            {
                interrupted = Some(entry);
                break;
            }

            if entry.retry_after > now {
                deferred.push(entry);
                continue;
            }
            // A holder that is gone was unloaded, and its drive went with it.
            let Some(holder) = entry.holder.upgrade() else {
                continue;
            };
            tracing::trace!(
                chunk = ?holder.get_pos(),
                reason = ?entry.reason,
                "Reviving stalled generation drive",
            );
            // Armed unconditionally rather than behind `needs_generation`: this
            // is the only edge that takes a holder out of the stalled phase, so
            // skipping it would leave a holder that no longer needs generation
            // stalled -- and counted as stalled -- for good. The re-admitted
            // pass costs one evaluation and exits idle.
            if holder.arm() {
                holder.queue_for_generation();
            }
        }

        let mut next: Vec<StalledGeneration> = Vec::new();
        next.extend(interrupted);
        next.extend(entries);
        next.append(&mut deferred);
        let mut queue = self.stalled_generation_drives.lock();
        // Anything filed while this ran goes behind what is already waiting.
        next.append(&mut queue);
        *queue = next;
    }

    /// Turns "the pregeneration hangs" into a list of which chunks are stuck and
    /// on what.
    ///
    /// The failure this exists for is silent by construction: a holder parked
    /// with nothing left that will ever wake it holds no task, no permit and no
    /// halo, so there is nothing to see in a thread dump, a profile or the task
    /// counters. It presented once as a 601x601 repetition that produced no
    /// output for fifteen minutes and had to be diagnosed by rebuilding with
    /// extra logging.
    ///
    /// Called once per scheduling epoch, and the gauge read below is almost
    /// always the whole cost: nothing is parked in a healthy pipeline.
    pub(super) fn check_generation_stall_watchdog(&self) {
        let parked = GENERATION_DRIVE_COUNTERS
            .parked_holders
            .load(Ordering::Relaxed);
        if parked <= 0 || !self.generation_pipeline_is_idle() {
            self.generation_stall_epochs.store(0, Ordering::Relaxed);
            self.generation_stall_reported
                .store(false, Ordering::Relaxed);
            return;
        }

        let epochs = self
            .generation_stall_epochs
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if epochs < STALL_WATCHDOG_EPOCHS || self.generation_stall_reported.load(Ordering::Relaxed)
        {
            return;
        }

        // The gauge is process-wide -- `DependencyWaiter::drop` runs on rayon
        // workers that hold no map -- while every condition above is this map's
        // own. An idle Nether therefore reaches here on the Overworld's parks
        // with nothing wrong, so the report is only made once this map's own
        // holders confirm it, and the count restarts from zero if they do not.
        if self.report_parked_holders(parked) {
            self.generation_stall_reported
                .store(true, Ordering::Relaxed);
        } else {
            // Not ours. Resetting to zero would rescan a whole `chunks` map --
            // 361k entries at 601x601 -- every `STALL_WATCHDOG_EPOCHS`, forever,
            // on the epoch thread, where every neighbouring pass is deadline
            // bounded. Back the next attempt off instead: another map's parks do
            // not become ours by being looked at again sooner.
            self.generation_stall_epochs.store(
                STALL_WATCHDOG_EPOCHS.saturating_sub(STALL_WATCHDOG_EPOCHS / 8),
                Ordering::Relaxed,
            );
        }
    }

    /// Whether nothing is running and nothing is queued to run.
    ///
    /// Each lock is taken and released on its own line rather than across one
    /// `&&` chain, whose temporaries would live to the end of the statement and
    /// hold the admission inbox while taking the selection queue -- the nesting
    /// `GenerationInbox` documents as forbidden.
    fn generation_pipeline_is_idle(&self) -> bool {
        if self.running_generation_tasks.load(Ordering::Acquire) != 0 {
            return false;
        }
        if !self.pending_generation_tasks.lock().is_empty() {
            return false;
        }
        if !self.generation_inbox.is_empty() {
            return false;
        }
        // Neither of these is in the watchdog's brief, and both have to be here
        // anyway: a backlogged holder is one this map has admitted but not yet
        // armed, and a stalled one is re-armed by `revive_stalled_generation_drives`
        // after a backoff that reaches `STALL_BACKOFF_BASE << STALL_BACKOFF_DOUBLINGS`
        // -- 640 ms, which is over a hundred epochs at the 5 ms mean measured
        // over a 601x601 pregeneration. Without them every deep backoff would
        // report a stall that resolves itself.
        if !self.pending_schedule_backlog.lock().is_empty() {
            return false;
        }
        // KNOWN BLIND SPOT. Excluding the stall list stops a deep backoff being
        // reported as a stall, but it also means a drive that *re-stalls
        // forever* keeps this list permanently non-empty and so silences the
        // watchdog -- and a permanently re-stalling drive produces exactly the
        // symptom the watchdog exists for, a run that emits nothing for minutes.
        // Distinguishing the two needs the entries' due times, not just
        // emptiness: legitimately-waiting entries are in backoff, a wedged one is
        // permanently overdue. Left as-is rather than guessed at, because a
        // watchdog that cries wolf during normal backoff is worse than one with a
        // documented gap.
        self.stalled_generation_drives.lock().is_empty()
    }

    /// Dumps the parked holders of *this* map. `false` means it has none, i.e.
    /// the gauge was raised by another map.
    fn report_parked_holders(&self, gauge: i64) -> bool {
        let mut sample: Vec<Arc<ChunkHolder>> = Vec::new();
        let mut parked = 0usize;
        // Collected first and inspected afterwards: reading a holder's blocker
        // resolves a halo, which takes `chunks` read guards, and taking one
        // inside `iter_sync` would reenter the map mid-iteration.
        self.chunks.iter_sync(|_, holder| {
            if holder.parked_generation_state().is_some() {
                parked += 1;
                if sample.len() < STALL_WATCHDOG_DUMP {
                    sample.push(Arc::clone(holder));
                }
            }
            true
        });
        if parked == 0 {
            return false;
        }

        tracing::error!(
            parked,
            gauge,
            listed = sample.len(),
            epochs = STALL_WATCHDOG_EPOCHS,
            "Chunk generation has stopped with holders still parked: no generation run is in \
             flight, the admission inbox and the selection queue are empty, and nothing is left \
             that will wake them",
        );
        for holder in sample {
            // Re-read rather than carried from the scan: the states below must
            // agree with the blocker resolved from the same pass, and a park
            // that ended in between is worth reporting as exactly that.
            let Some(state) = holder.parked_generation_state() else {
                tracing::error!(
                    chunk = ?holder.get_pos(),
                    "Parked generation drive left its park while the stall was being reported",
                );
                continue;
            };
            let published = holder.published_status();
            let next = match published {
                None => Some(ChunkStatus::Empty),
                Some(published) => published.next(),
            };
            let blocker = next.map_or(ParkBlocker::NoRunLeft, |next| {
                self.park_blocker(holder.get_pos(), next)
            });
            tracing::error!(
                chunk = ?holder.get_pos(),
                ?published,
                ?next,
                allowed = ?holder.highest_allowed_status(),
                outstanding = state.outstanding,
                epoch = state.epoch,
                ?blocker,
                "Parked chunk generation drive",
            );
        }
        true
    }

    /// What the run this holder would make next is still waiting for.
    pub(super) fn park_blocker(&self, pos: ChunkPos, next: ChunkStatus) -> ParkBlocker {
        match resolve_and_check(self, pos, &RUN_PLANS[next.get_index()]) {
            // The most constraining one, i.e. the dependency `fanout_selection`
            // would have registered first, because that is the one the park is
            // actually waiting on -- the easier ones publish long before it.
            HaloResolution::Unmet(unmet) => {
                fanout_selection(&unmet)
                    .next()
                    .map_or(ParkBlocker::NothingUnmet, |dependency| {
                        ParkBlocker::Dependency {
                            chunk: dependency.holder.get_pos(),
                            required: dependency.required,
                        }
                    })
            }
            HaloResolution::Missing(chunk) => ParkBlocker::MissingHolder(chunk),
            HaloResolution::Ready(_) => ParkBlocker::NothingUnmet,
        }
    }
}

/// What one parked holder is waiting for, as of the moment the watchdog looked.
///
/// Every payload is read through the `Debug` rendering the log line carries and
/// through nothing else, which dead-code analysis deliberately does not count.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "rendered into the stall report by Debug")
)]
pub(super) enum ParkBlocker {
    /// The dependency the park is really waiting on.
    Dependency {
        chunk: ChunkPos,
        required: ChunkStatus,
    },
    /// A halo cell with no holder. Nothing there can publish a status, so the
    /// park cannot end on one -- the drive should have stalled instead, and the
    /// backoff revival should have re-armed it.
    MissingHolder(ChunkPos),
    /// Every dependency of the next run is met, so this park has outlived its
    /// cause: a registration was lost, or the requeue it produced never reached
    /// the admission inbox.
    NothingUnmet,
    /// Parked with no run left to make. The holder is at its last status and
    /// should not have parked at all.
    NoRunLeft,
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;
    use steel_utils::ChunkPos;

    use std::sync::Arc;

    use super::{
        CachedHalo, GENERATION_FANOUT, HaloResolution, PendingCell, RUN_PLANS, StallReason,
        UnmetDependency, fanout_selection, recheck_cached, ring_cells,
    };
    use crate::chunk::{
        chunk_holder::{ChunkHolder, STALL_REASON_COUNT},
        chunk_ticket_manager::ChunkTicketLevel,
        static_cache_2d::StaticCache2D,
        status::ChunkStatus,
    };

    const CENTRE: ChunkPos = ChunkPos::new(-3, 7);

    /// The cell the re-check tests hold back while everything around it moves on.
    const LAGGARD: ChunkPos = ChunkPos::new(CENTRE.0.x + 1, CENTRE.0.y);
    /// The widest halo the pyramid resolves, which is the one worth testing.
    const HALO_RADIUS: i32 = 8;

    /// The pass that fills a run's halo visits each cell by ring and then indexes
    /// the backing vector directly, so a ring that skipped a cell would leave a
    /// hole that `from_row_major`'s `expect` turns into a panic in the middle of
    /// generation -- and a ring that yielded one twice would silently overwrite.
    #[test]
    fn the_rings_tile_the_square_exactly_once() {
        for radius in 0..=8 {
            let mut seen = Vec::new();
            for distance in (0..=radius).rev() {
                seen.extend(ring_cells(CENTRE, distance));
            }

            let expected = ((radius * 2 + 1) * (radius * 2 + 1)) as usize;
            assert_eq!(
                seen.len(),
                expected,
                "radius {radius} yielded the wrong count"
            );
            assert_eq!(
                seen.iter().copied().collect::<FxHashSet<_>>().len(),
                expected,
                "radius {radius} yielded a cell twice",
            );
            for (x, z) in seen {
                assert!(
                    (x - CENTRE.0.x).abs() <= radius && (z - CENTRE.0.y).abs() <= radius,
                    "radius {radius} yielded ({x}, {z}), which is outside the square",
                );
            }
        }
    }

    #[test]
    fn a_ring_holds_exactly_the_cells_at_its_distance() {
        assert_eq!(ring_cells(CENTRE, 0).collect::<Vec<_>>(), vec![(-3, 7)]);
        for distance in 1..=8 {
            let cells = ring_cells(CENTRE, distance).collect::<Vec<_>>();
            assert_eq!(cells.len(), (8 * distance) as usize);
            for (x, z) in cells {
                let chebyshev = (x - CENTRE.0.x).abs().max((z - CENTRE.0.y).abs());
                assert_eq!(chebyshev, distance, "({x}, {z}) is not on ring {distance}");
            }
        }
    }

    /// The resolver only gates on cells the ring names, and reads
    /// `plan.ring.get(distance)` for every cell out to `plan.halo_radius`. A ring
    /// that named a status *beyond* the halo would therefore never be checked at
    /// all, and the run would dispatch against dependencies it had not waited for.
    #[test]
    fn no_run_names_a_requirement_outside_the_halo_it_resolves() {
        for plan in &RUN_PLANS {
            assert!(
                plan.halo_radius >= plan.ring.get_radius(),
                "the run starting at {:?} rings out to {} but only resolves {}",
                plan.first,
                plan.ring.get_radius(),
                plan.halo_radius,
            );
        }
    }

    /// The park only ends once every waiter it filed has fired, so the fan-out
    /// prefix must be the *last* part of the unmet set to clear, not the first.
    /// `resolve_and_check` produces that set outermost-first, and a run's ring
    /// never rises with radius, so the highest requirements sit at the tail.
    /// Registering the head instead wakes the holder on dependencies that were
    /// never what blocked it, and it re-scans the whole square and parks again.
    #[test]
    fn the_fan_out_registers_the_most_constraining_dependencies() {
        // Exactly the shape a radius-8 `Noise` park produces: the outer rings
        // need only `StructureStarts`, the centre needs `Biomes`.
        let mut unmet: Vec<UnmetDependency> = (0..*GENERATION_FANOUT * 2)
            .map(|_| UnmetDependency {
                holder: unmet_holder(),
                required: ChunkStatus::StructureStarts,
            })
            .collect();
        let blocking = *GENERATION_FANOUT / 2;
        for dependency in unmet.iter_mut().rev().take(blocking) {
            dependency.required = ChunkStatus::Biomes;
        }

        let selected: Vec<ChunkStatus> = fanout_selection(&unmet)
            .map(|dependency| dependency.required)
            .collect();

        assert_eq!(selected.len(), *GENERATION_FANOUT, "the prefix is bounded");
        assert_eq!(
            selected
                .iter()
                .filter(|required| **required == ChunkStatus::Biomes)
                .count(),
            blocking,
            "every dependency that actually gates the run must be registered",
        );
    }

    /// A cell that is behind when one run looks at it stays behind when the next
    /// run looks at it, and the next run asks for *more*.
    ///
    /// This is the case the narrowing must not lose: the cell is skipped by
    /// nothing, because it never reaches a status that settles it, and both runs
    /// have to see it.
    #[test]
    fn a_cell_behind_at_one_requirement_is_reported_again_at_the_next() {
        let cache = square(HALO_RADIUS);
        publish_all(&cache, ChunkStatus::StructureStarts);
        let mut cached = CachedHalo::new(
            CENTRE,
            &RUN_PLANS[ChunkStatus::StructureReferences.get_index()],
            Arc::clone(&cache),
        );
        assert_eq!(
            cached.pending.len(),
            9,
            "a radius-8 halo only resolves once every cell is at StructureStarts, which is what \
             retires everything from distance 2 out",
        );

        // The drive and its neighbourhood run on; one neighbour does not.
        for holder in cells(&cache) {
            if holder.get_pos() != LAGGARD {
                holder.finish_generation_status_for_test(ChunkStatus::Carvers);
            }
        }

        let unmet = unmet_of(recheck_cached(
            &mut cached,
            &RUN_PLANS[ChunkStatus::Noise.get_index()],
        ));
        assert_eq!(blocked_by(&unmet), vec![(LAGGARD, ChunkStatus::Biomes)]);

        let unmet = unmet_of(recheck_cached(
            &mut cached,
            &RUN_PLANS[ChunkStatus::Features.get_index()],
        ));
        assert_eq!(blocked_by(&unmet), vec![(LAGGARD, ChunkStatus::Carvers)]);
    }

    /// The trap a "satisfied" bitmap falls into.
    ///
    /// `Noise` needs `Biomes` of its radius-1 neighbours and `Features` needs
    /// `Carvers` of them, so a cell that met the first run's ring exactly is
    /// still a dependency of the second. Recording the *status* a cell reached
    /// rather than a flag is what keeps it in the walk.
    #[test]
    fn a_cell_met_for_one_run_is_re_read_for_the_next_runs_higher_requirement() {
        let cache = square(HALO_RADIUS);
        publish_all(&cache, ChunkStatus::StructureStarts);
        let mut cached = CachedHalo::new(
            CENTRE,
            &RUN_PLANS[ChunkStatus::StructureReferences.get_index()],
            Arc::clone(&cache),
        );

        for holder in cells(&cache) {
            holder.finish_generation_status_for_test(if holder.get_pos() == LAGGARD {
                ChunkStatus::Biomes
            } else {
                ChunkStatus::Carvers
            });
        }

        assert!(
            matches!(
                recheck_cached(&mut cached, &RUN_PLANS[ChunkStatus::Noise.get_index()]),
                HaloResolution::Ready(_),
            ),
            "Biomes is exactly what the Noise run asks of a radius-1 neighbour",
        );

        let unmet = unmet_of(recheck_cached(
            &mut cached,
            &RUN_PLANS[ChunkStatus::Features.get_index()],
        ));
        assert_eq!(blocked_by(&unmet), vec![(LAGGARD, ChunkStatus::Carvers)]);
    }

    /// A cell is dropped from the walk only once it is past everything any run
    /// can ask of its distance -- and then it really is dropped.
    #[test]
    fn a_cell_seen_at_the_retirement_status_leaves_the_walk() {
        let cache = square(HALO_RADIUS);
        publish_all(&cache, ChunkStatus::StructureStarts);
        let mut cached = CachedHalo::new(
            CENTRE,
            &RUN_PLANS[ChunkStatus::StructureReferences.get_index()],
            Arc::clone(&cache),
        );

        publish_all(&cache, ChunkStatus::InitializeLight);
        assert!(matches!(
            recheck_cached(&mut cached, &RUN_PLANS[ChunkStatus::Light.get_index()]),
            HaloResolution::Ready(_),
        ));
        assert_eq!(
            cached
                .pending
                .iter()
                .map(|cell| cell.holder.get_pos())
                .collect::<Vec<_>>(),
            vec![CENTRE],
            "InitializeLight is the most any ring asks at distance 1, so those eight cells are \
             done; distance 0 is asked for Spawn by the Full run and stays",
        );
    }

    /// The re-check emits its unmet set outermost-first, which is the whole
    /// basis of `fanout_selection` taking the tail: a run's ring never rises
    /// with radius, so that order runs from the lowest requirement to the
    /// highest and the tail is what actually gates the run.
    #[test]
    fn the_re_check_emits_its_unmet_set_outermost_first() {
        let cache = square(HALO_RADIUS);
        let mut cached = CachedHalo {
            radius: HALO_RADIUS,
            cache: Arc::clone(&cache),
            // Nothing observed anywhere: the widest walk the narrowing can
            // produce, so the ordering is tested over the whole square rather
            // than over the nine cells a settled halo leaves.
            pending: unobserved(&cache),
        };

        let unmet = unmet_of(recheck_cached(
            &mut cached,
            &RUN_PLANS[ChunkStatus::Noise.get_index()],
        ));
        assert_eq!(
            unmet.len(),
            17 * 17,
            "nothing is published, so nothing is met"
        );

        let distances: Vec<i32> = unmet
            .iter()
            .map(|dependency| chebyshev(dependency.holder.get_pos()))
            .collect();
        assert!(
            distances.windows(2).all(|pair| pair[0] >= pair[1]),
            "the walk must stay outermost-first: {distances:?}",
        );
        assert!(
            unmet
                .windows(2)
                .all(|pair| pair[0].required <= pair[1].required),
            "outermost-first must put the most constraining requirements at the tail",
        );
        assert_eq!(
            fanout_selection(&unmet)
                .next()
                .map(|dependency| dependency.required),
            Some(ChunkStatus::Biomes),
            "the park must register the innermost ring first",
        );
    }

    fn chebyshev(pos: ChunkPos) -> i32 {
        (pos.0.x - CENTRE.0.x)
            .abs()
            .max((pos.0.y - CENTRE.0.y).abs())
    }

    /// A halo of fresh holders around [`CENTRE`], as `resolve_and_check` builds.
    fn square(radius: i32) -> Arc<StaticCache2D<Arc<ChunkHolder>>> {
        let size = radius * 2 + 1;
        let min_x = CENTRE.0.x - radius;
        let min_z = CENTRE.0.y - radius;
        let cells = (0..size * size)
            .map(|index| holder_at(ChunkPos::new(min_x + index % size, min_z + index / size)))
            .collect();
        Arc::new(StaticCache2D::from_row_major(min_x, min_z, size, cells))
    }

    /// Every cell of a [`square`] of `HALO_RADIUS`, outermost first.
    fn cells(
        cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    ) -> impl Iterator<Item = &Arc<ChunkHolder>> {
        (0..=HALO_RADIUS)
            .rev()
            .flat_map(|distance| ring_cells(CENTRE, distance).map(|(x, z)| cache.get(x, z)))
    }

    fn publish_all(cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>, status: ChunkStatus) {
        for holder in cells(cache) {
            holder.finish_generation_status_for_test(status);
        }
    }

    /// The pending set of a halo nothing has been observed in yet.
    fn unobserved(cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>) -> Vec<PendingCell> {
        (0..=HALO_RADIUS)
            .rev()
            .flat_map(|distance| {
                ring_cells(CENTRE, distance).map(move |(x, z)| PendingCell {
                    holder: Arc::clone(cache.get(x, z)),
                    distance: distance as usize,
                    seen: None,
                })
            })
            .collect()
    }

    fn unmet_of(resolution: HaloResolution) -> Vec<UnmetDependency> {
        match resolution {
            HaloResolution::Unmet(unmet) => unmet,
            HaloResolution::Ready(_) => panic!("the run was expected to be blocked"),
            HaloResolution::Missing(pos) => panic!("a cached halo cannot lose {pos:?}"),
        }
    }

    fn blocked_by(unmet: &[UnmetDependency]) -> Vec<(ChunkPos, ChunkStatus)> {
        unmet
            .iter()
            .map(|dependency| (dependency.holder.get_pos(), dependency.required))
            .collect()
    }

    fn unmet_holder() -> Arc<ChunkHolder> {
        holder_at(CENTRE)
    }

    fn holder_at(pos: ChunkPos) -> Arc<ChunkHolder> {
        Arc::new(ChunkHolder::new(
            pos,
            ChunkTicketLevel::FULL_CHUNK,
            Some(ChunkTicketLevel::FULL_CHUNK),
            0,
            16,
        ))
    }

    #[test]
    fn every_stall_reason_has_its_own_counter() {
        let reasons = [
            StallReason::HaloMiss(CENTRE),
            StallReason::Refused,
            StallReason::JobFailed,
        ];
        let mut indices = reasons.map(StallReason::index).to_vec();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), reasons.len(), "two reasons share a counter");
        assert_eq!(
            indices.len(),
            STALL_REASON_COUNT,
            "the counter array does not match the reasons that index it",
        );
        assert!(indices.iter().all(|index| *index < STALL_REASON_COUNT));
    }

    /// The evaluation turns a published status into the run that comes next, so a
    /// status that is not a run's first would resolve the wrong plan -- and
    /// restarting a run's first status on a holder already past it is exactly what
    /// `claim_status_work` panics on.
    #[test]
    fn every_status_indexes_the_run_that_starts_at_it() {
        for (index, plan) in RUN_PLANS.iter().enumerate() {
            let status = ChunkStatus::from_index(index).expect("status decodes");
            assert_eq!(plan.first, status);
        }
    }
}
