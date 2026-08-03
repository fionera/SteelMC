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
    chunk_pyramid::{GENERATION_PYRAMID, RUN_PLANS, RunPlan},
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

        match resolve_and_check(chunk_map, holder.get_pos(), plan) {
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
                // spawned. Holding a second reference across the await would
                // keep every one of those holders alive against unloading for
                // the whole run, for nothing.
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
            HaloResolution::Unmet(unmet) => match park(holder, ticket, &unmet) {
                ParkOutcome::Parked => return,
                ParkOutcome::ReEvaluate => {}
            },
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
fn resolve_and_check(
    chunk_map: &Arc<ChunkMap>,
    center: ChunkPos,
    plan: &RunPlan,
) -> HaloResolution {
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
            // One lookup that takes both the reference and the status off the
            // same holder under the same read guard. Reading the status through
            // a second lookup would double the cost of the pass.
            let cell = chunk_map.chunks.read_sync(&pos, |_, holder| {
                (Arc::clone(holder), holder.published_status())
            });
            let Some((holder, published)) = cell else {
                return HaloResolution::Missing(pos);
            };

            if let Some(required) = required
                && published.is_none_or(|published| published < required)
            {
                if unmet.is_empty() {
                    // First unmet cell: this pass is going to park, so free the
                    // halo now rather than at the end of the pass.
                    halo = Vec::new();
                }
                unmet.push(UnmetDependency { holder, required });
                continue;
            }

            if unmet.is_empty() {
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
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;
    use steel_utils::ChunkPos;

    use std::sync::Arc;

    use super::{
        GENERATION_FANOUT, RUN_PLANS, StallReason, UnmetDependency, fanout_selection, ring_cells,
    };
    use crate::chunk::{
        chunk_holder::{ChunkHolder, STALL_REASON_COUNT},
        chunk_ticket_manager::ChunkTicketLevel,
        status::ChunkStatus,
    };

    const CENTRE: ChunkPos = ChunkPos::new(-3, 7);

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

    fn unmet_holder() -> Arc<ChunkHolder> {
        Arc::new(ChunkHolder::new(
            CENTRE,
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
