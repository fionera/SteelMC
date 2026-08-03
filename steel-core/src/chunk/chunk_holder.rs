//! `ChunkHolder` manages chunk state and asynchronous generation tasks.
use futures::Future;
use rustc_hash::FxHashSet;
use std::fmt::Debug;
use std::mem;
use std::ptr;
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::{Arc, OnceLock, Weak};
use steel_utils::atomic_wait_queue::{AtomicWaitQueue, WaitOutcome};
use steel_utils::{BlockPos, ChunkPos, PackedSectionBlockPos, SectionPos, locks::SyncMutex};
use tokio::sync::{Notify, oneshot};
#[cfg(feature = "slow_chunk_gen")]
use tokio::time::sleep;

#[cfg(feature = "slow_chunk_gen")]
use std::time::Duration;

/// When `true`, each chunk generation stage sleeps 200 ms after completing.
/// Set by the spawn progress display to make the terminal grid visible.
#[cfg(feature = "slow_chunk_gen")]
pub static SLOW_CHUNK_GEN: AtomicBool = AtomicBool::new(false);

use crate::chunk::chunk_generation_task::{NeighborReady, StaticCache2D};
use crate::chunk::chunk_map::{GenerationInbox, STAGE1};
use crate::chunk::chunk_ticket_manager::{
    ChunkTicketLevel, generation_status, is_entity_ticking, is_full,
};
use crate::chunk::full_chunk_readiness::FullPublicationQueue;
use crate::chunk::generation_drive::{DecOutcome, DrivePhase, GenerationDrive};
use crate::chunk::light::{
    LightLayer, LightSectionRange, LightWorkWindowGate, LightWorkWindowReservation,
};
use crate::chunk_saver::ChunkStorage;
use crate::entity::EntityVisibility;
use crate::worldgen::WorldGenContext;
use crate::{
    ChunkMap,
    chunk::{
        Chunk,
        chunk_generation_task::ChunkGenerationTask,
        chunk_pyramid::{ChunkStep, GENERATION_PYRAMID, can_fuse},
        full_chunk::{FullChunkPromotion, FullChunkRef},
        status::ChunkStatus,
    },
};

const STATUS_NONE: u8 = u8::MAX;
/// Values of [`ChunkHolder::drive_gauge`].
const DRIVE_GAUGE_NONE: u8 = 0;
const DRIVE_GAUGE_PARKED: u8 = 1;
const DRIVE_GAUGE_STALLED: u8 = 2;
const UNPUBLISHED_STATUS: u8 = 0;
const NO_TICKET_LEVEL: u8 = u8::MAX;
const SAVE_LIFECYCLE_ACTIVE: u8 = 0;
const SAVE_LIFECYCLE_UNLOADING: u8 = 1;
const SAVE_LIFECYCLE_PREPARING: u8 = 2;

fn optional_ticket_level_raw(level: Option<ChunkTicketLevel>) -> u8 {
    level.map_or(NO_TICKET_LEVEL, ChunkTicketLevel::raw)
}

const fn optional_ticket_level_from_raw(raw: u8) -> Option<ChunkTicketLevel> {
    if raw == NO_TICKET_LEVEL {
        None
    } else {
        ChunkTicketLevel::new(raw)
    }
}

const fn encoded_published_status(status: ChunkStatus) -> u8 {
    status.get_index() as u8 + 1
}

fn decoded_published_status(status: u8) -> Option<ChunkStatus> {
    if status == UNPUBLISHED_STATUS {
        return None;
    }

    let decoded = ChunkStatus::from_index(usize::from(status - 1));
    assert!(
        decoded.is_some(),
        "invalid published chunk status: {status}"
    );
    decoded
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum TickingReadiness {
    Unready,
    BlockTicking,
    EntityTicking,
}

/// Exact ticking-readiness generation captured by concurrent consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TickingReadinessSnapshot(u64);

impl TickingReadinessSnapshot {
    #[must_use]
    pub(crate) const fn readiness(self) -> TickingReadiness {
        match self.0 & 0b11 {
            0 => TickingReadiness::Unready,
            1 => TickingReadiness::BlockTicking,
            2 => TickingReadiness::EntityTicking,
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub(crate) const fn is_block_ticking(self) -> bool {
        matches!(
            self.readiness(),
            TickingReadiness::BlockTicking | TickingReadiness::EntityTicking
        )
    }

    #[must_use]
    pub(crate) const fn is_entity_ticking(self) -> bool {
        matches!(self.readiness(), TickingReadiness::EntityTicking)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostProcessGenerationError {
    ChunkNotFull,
    WorldUnavailable,
}

#[derive(Debug, Default)]
struct ChangedLightSectionSets {
    sky: FxHashSet<SectionPos>,
    block: FxHashSet<SectionPos>,
}

/// Pending light sections to send to players tracking a chunk.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChangedLightSections {
    /// Changed sky-light sections.
    pub sky: Vec<SectionPos>,
    /// Changed block-light sections.
    pub block: Vec<SectionPos>,
}

impl ChangedLightSections {
    /// Returns true when no light sections changed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sky.is_empty() && self.block.is_empty()
    }
}

/// Instrumentation for the per-holder generation drive.
///
/// Statics rather than a field on `ChunkMap`, because two of these are bumped
/// from places that hold no map: [`DependencyWaiter::drop`] runs on whichever
/// rayon worker dropped the wait queue and has only a `Weak<ChunkHolder>` to
/// work from. The worldgen ore profile keeps its totals in a static for the
/// same reason.
///
/// Everything here stays at zero until the drive is wired into the scheduler.
pub(crate) static GENERATION_DRIVE_COUNTERS: GenerationDriveCounters =
    GenerationDriveCounters::new();

/// The counters behind [`GENERATION_DRIVE_COUNTERS`].
///
/// The gauges are signed: they are raised and lowered from different threads,
/// and an unmatched decrement has to read as `-1` rather than as `u64::MAX`,
/// which is the difference between "we have a bug" and "the counter is
/// meaningless".
pub(crate) struct GenerationDriveCounters {
    /// Holders currently parked on at least one dependency.
    pub(crate) parked_holders: AtomicI64,
    /// Holders currently stalled, i.e. unable to progress until re-armed.
    pub(crate) stalled_holders: AtomicI64,
    /// [`DependencyWaiter`]s created and not yet released.
    pub(crate) live_dependency_registrations: AtomicI64,
    /// Registrations released by the wait queue being dropped rather than by
    /// the status they wait for being published. See [`DependencyWaiter::drop`]
    /// for why this is expected to stay at zero.
    pub(crate) dependency_waiters_dropped_unfired: AtomicU64,
    /// Parks whose last registration resolved, handing the holder back for
    /// admission.
    pub(crate) drive_wakes: AtomicU64,
    /// Drive futures dropped before they retired their holder.
    ///
    /// Expected to stay at zero outside shutdown: a dropped drive leaves the
    /// holder's phase at `Running` with nobody inside it, and any work claim it
    /// held rolls back under a holder nothing will re-admit. That is how the
    /// first attempt at this scheduler corrupted claims.
    pub(crate) drive_futures_dropped: AtomicU64,
    /// Work claims lost to another claimant.
    ///
    /// Only counted under the per-holder drive, where it must stay at zero: a
    /// holder has exactly one driver, so a lost claim means two dispatchers ran
    /// against the same holder and `claim_status_work` is one step away from
    /// panicking the server.
    pub(crate) contended_status_claims: AtomicU64,
    /// Admission permits currently blocked on the light work-window gate.
    ///
    /// See [`ChunkHolder::await_light_work_window_and_apply_step`]: the drive
    /// deliberately holds its permit across that wait, and this is the gauge
    /// that says how much admission capacity it costs.
    pub(crate) permits_waiting_on_light_window: AtomicI64,
    /// Stalls, indexed by the drive's stall reason.
    pub(crate) stalls_by_reason: [AtomicU64; STALL_REASON_COUNT],
}

/// Number of stall reasons the drive distinguishes.
///
/// [`StallReason`]: crate::chunk::chunk_map::generation_drive_dispatch::StallReason
pub(crate) const STALL_REASON_COUNT: usize = 3;

impl GenerationDriveCounters {
    const fn new() -> Self {
        Self {
            parked_holders: AtomicI64::new(0),
            stalled_holders: AtomicI64::new(0),
            live_dependency_registrations: AtomicI64::new(0),
            dependency_waiters_dropped_unfired: AtomicU64::new(0),
            drive_wakes: AtomicU64::new(0),
            drive_futures_dropped: AtomicU64::new(0),
            contended_status_claims: AtomicU64::new(0),
            permits_waiting_on_light_window: AtomicI64::new(0),
            stalls_by_reason: [const { AtomicU64::new(0) }; STALL_REASON_COUNT],
        }
    }
}

/// What a holder's status queue hands back when the status it waits for is
/// published.
pub(crate) enum StatusWaiter {
    /// An `await_status_with` future.
    Oneshot(oneshot::Sender<()>),
    /// Another holder's parked generation drive.
    Dependency(DependencyWaiter),
}

/// One parked holder's registration on a neighbour's status.
///
/// Exactly one of [`Self::fire`] and [`Self::drop`] releases the registration
/// with the parent's drive; see either for how that is arranged.
pub(crate) struct DependencyWaiter {
    /// Weak, and never an `Arc`. A strong reference held from one holder's wait
    /// queue to another holder keeps the target's `Arc::strong_count` above one
    /// for as long as the queue lives, and `ChunkMap::process_unloads` uses
    /// exactly that count to decide a holder is unreferenced -- an earlier
    /// attempt at this scheduler leaked every holder it generated that way.
    parent: Weak<ChunkHolder>,
    /// Never read: the wait queue itself decides when this fires, from the
    /// encoded status the registration was filed under. Kept because a waiter
    /// pulled out of a core dump or a debugger is otherwise anonymous, and the
    /// question asked of a stuck park is always "waiting for what".
    #[expect(dead_code, reason = "diagnostic only; see the field comment")]
    required: ChunkStatus,
    /// The park epoch of `parent` this registration belongs to. A release
    /// carrying any other epoch is refused by the drive, so a registration can
    /// never decrement a park it did not arm.
    epoch: u64,
}

impl DependencyWaiter {
    pub(crate) fn new(parent: &Arc<ChunkHolder>, required: ChunkStatus, epoch: u64) -> Self {
        GENERATION_DRIVE_COUNTERS
            .live_dependency_registrations
            .fetch_add(1, Ordering::Relaxed);
        Self {
            parent: Arc::downgrade(parent),
            required,
            epoch,
        }
    }

    /// Releases this registration because the status it waited for was
    /// published, and reports the parent when this caller now owns its requeue.
    fn fire(mut self) -> Option<Arc<ChunkHolder>> {
        // Empty the `Weak` so the `Drop` that runs at the end of this function
        // finds nothing to release. `mem::forget` would do that too, but it
        // would leak the `Weak` -- and its weak count keeps the parent's
        // allocation alive. Doing neither releases twice, taking the drive's
        // 16-bit outstanding count one below zero; it wraps to ~65k and the
        // parent never leaves its park.
        let parent = mem::replace(&mut self.parent, Weak::new());
        // A parent that is gone was unloaded, and its park went with it.
        let parent = parent.upgrade()?;
        (parent.finish_dependency(self.epoch) == DecOutcome::Requeue).then_some(parent)
    }
}

impl Drop for DependencyWaiter {
    /// Runs on both paths: [`Self::fire`] consumes the waiter, so the gauge is
    /// lowered here exactly once per registration however it ended.
    ///
    /// Reaching the release below, on the other hand, means the wait queue
    /// itself was dropped with this waiter still registered -- the neighbour was
    /// destroyed before it published the status. That should not happen, since a
    /// neighbour cannot fall below the ticket level a dependent needs while that
    /// dependent still holds one, so the counter exists to say whether it ever
    /// does. The release is defence in depth: without it the parent keeps an
    /// outstanding count for a status nobody will ever publish. It goes through
    /// the same requeue as `fire` for the same reason -- the release that takes
    /// the count to zero is the only one that will ever be told to admit the
    /// parent again, whichever path it arrives on.
    fn drop(&mut self) {
        GENERATION_DRIVE_COUNTERS
            .live_dependency_registrations
            .fetch_sub(1, Ordering::Relaxed);
        let Some(parent) = self.parent.upgrade() else {
            return;
        };
        GENERATION_DRIVE_COUNTERS
            .dependency_waiters_dropped_unfired
            .fetch_add(1, Ordering::Relaxed);
        if parent.finish_dependency(self.epoch) == DecOutcome::Requeue {
            parent.requeue_for_generation();
        }
    }
}

/// Holds chunk data and coordinates asynchronous generation work.
///
/// The published status is released only after the corresponding data and Full
/// tick containers are installed. Synchronous readers acquire it before reading
/// `data`.
///
/// It lives in `status` as the wait queue's status word, so publishing a status
/// and releasing the waiters for it are one atomic step: a waiter cannot be
/// registered for a status that has already been published, and a publish
/// cannot miss a waiter registered concurrently with it.
///
/// `status_changed` remains for the events that are *not* status raises --
/// generation becoming disallowed during unload, and an abandoned work claim
/// being rolled back. Those must wake waiters without moving the status, which
/// the queue deliberately cannot express.
pub struct ChunkHolder {
    data: OnceLock<Chunk>,
    /// Published status, and everything waiting for a later one.
    ///
    /// Payloads are dropped wherever the queue is drained or destroyed, which
    /// after the scheduler rewrite includes a rayon generation worker inside
    /// the publish path. Every payload's drop must therefore stay a leaf:
    /// `oneshot::Sender::drop` wakes a receiver, `Weak::drop` decrements a
    /// refcount, and [`DependencyWaiter::drop`] touches one atomic word plus
    /// the target's inbox. None of them re-enters the chunk map, and nothing
    /// added here may either -- a payload drop that took a map lock would do so
    /// while every chunk waiting on the status being published is blocked.
    status: AtomicWaitQueue<StatusWaiter>,
    /// The per-holder generation state machine. Armed only when
    /// [`STAGE1`](crate::chunk::chunk_map::STAGE1) selects the per-holder
    /// dispatcher.
    drive: GenerationDrive,
    /// Which of the drive gauges this holder is currently counted in.
    ///
    /// The gauges have to be exact to be worth anything, and "how many holders
    /// are parked" cannot be recovered by sampling: the transitions out of
    /// `Parked` happen on rayon workers that hold no map. So the holder carries
    /// its own membership and every transition swaps it, which makes the
    /// increment and the decrement a single owner's business.
    drive_gauge: AtomicU8,
    /// Consecutive stalls, for the revival backoff. Reset by a run that
    /// dispatched work.
    stall_attempts: AtomicU32,
    status_changed: Notify,
    generation_task: SyncMutex<Option<Arc<ChunkGenerationTask>>>,
    generation_task_target: AtomicU8,
    pos: ChunkPos,
    /// The current loading ticket level of the chunk.
    load_level: AtomicU8,
    /// The current simulation ticket level of the chunk.
    simulation_level: AtomicU8,
    /// The highest status that has started work.
    started_work: AtomicUsize,
    /// Number of save dependencies that have not completed yet.
    active_save_dependencies: AtomicUsize,
    /// Coordinates unloading revival with the short immutable save-preparation phase.
    save_lifecycle: AtomicU8,
    /// The highest status that generation is allowed to reach.
    highest_allowed_status: AtomicU8,
    /// The minimum Y coordinate of the world.
    min_y: i32,
    /// The total height of the world.
    height: i32,
    /// Whether any sections have pending block changes.
    has_changed_sections: AtomicBool,
    /// Whether this holder is already queued for the next broadcast flush.
    queued_for_broadcast: AtomicBool,
    /// Monotonic revision for client-visible chunk packet content.
    packet_content_revision: AtomicU64,
    /// Packed ticking readiness generation. The low two bits store `TickingReadiness`.
    ticking_readiness: AtomicU64,
    /// Whether Full post-load initialization completed and was published for readiness.
    full_status_initialized: AtomicBool,
    /// Weak sink for Full status publication notifications.
    full_publications: Weak<FullPublicationQueue>,
    /// Weak sink for holders this one hands back for generation admission.
    generation_inbox: Weak<GenerationInbox>,
    /// Per-section sets of changed block positions.
    /// Index is `(block_y - min_y) / 16`.
    changed_blocks_per_section: Box<[SyncMutex<FxHashSet<PackedSectionBlockPos>>]>,
    /// Changed light sections grouped by light layer.
    changed_light_sections: SyncMutex<ChangedLightSectionSets>,
}

struct StatusWorkClaim {
    holder: Arc<ChunkHolder>,
    status: ChunkStatus,
}

impl StatusWorkClaim {
    const fn new(holder: Arc<ChunkHolder>, status: ChunkStatus) -> Self {
        Self { holder, status }
    }
}

impl Drop for StatusWorkClaim {
    fn drop(&mut self) {
        self.holder.release_status_work_claim(self.status);
    }
}

/// Raises [`GenerationDriveCounters::permits_waiting_on_light_window`] for as
/// long as it is held.
struct LightWindowWaitGauge;

impl LightWindowWaitGauge {
    fn new() -> Self {
        GENERATION_DRIVE_COUNTERS
            .permits_waiting_on_light_window
            .fetch_add(1, Ordering::Relaxed);
        Self
    }
}

impl Drop for LightWindowWaitGauge {
    fn drop(&mut self) {
        GENERATION_DRIVE_COUNTERS
            .permits_waiting_on_light_window
            .fetch_sub(1, Ordering::Relaxed);
    }
}

pub(crate) struct ChunkSaveDependency {
    holder: Arc<ChunkHolder>,
}

impl Drop for ChunkSaveDependency {
    fn drop(&mut self) {
        self.holder
            .active_save_dependencies
            .fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct ChunkSavePreparationGuard {
    holder: Arc<ChunkHolder>,
}

impl Drop for ChunkSavePreparationGuard {
    fn drop(&mut self) {
        let result = self.holder.save_lifecycle.compare_exchange(
            SAVE_LIFECYCLE_PREPARING,
            SAVE_LIFECYCLE_UNLOADING,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert!(
            result.is_ok(),
            "chunk save preparation ended outside the preparing lifecycle"
        );
    }
}

impl ChunkHolder {
    /// Gets the chunk position.
    pub const fn get_pos(&self) -> ChunkPos {
        self.pos
    }

    /// Gets the minimum Y coordinate of the world.
    pub const fn min_y(&self) -> i32 {
        self.min_y
    }

    /// Gets the total height of the world.
    pub const fn height(&self) -> i32 {
        self.height
    }

    /// Creates a new chunk holder.
    #[must_use]
    pub fn new(
        pos: ChunkPos,
        load_level: ChunkTicketLevel,
        simulation_level: Option<ChunkTicketLevel>,
        min_y: i32,
        height: i32,
    ) -> Self {
        Self::new_with_map_sinks(
            pos,
            load_level,
            simulation_level,
            min_y,
            height,
            Weak::new(),
            Weak::new(),
        )
    }

    /// Creates a holder wired to the map's sinks.
    ///
    /// [`Self::new`] passes `Weak::new()` for both, which never upgrades, so the
    /// holders built by hand in tests, benches and worldgen simply publish
    /// nowhere instead of each having to construct a map's queues.
    pub(crate) fn new_with_map_sinks(
        pos: ChunkPos,
        load_level: ChunkTicketLevel,
        simulation_level: Option<ChunkTicketLevel>,
        min_y: i32,
        height: i32,
        full_publications: Weak<FullPublicationQueue>,
        generation_inbox: Weak<GenerationInbox>,
    ) -> Self {
        let highest_allowed_status =
            generation_status(Some(load_level)).map_or(STATUS_NONE, |s| s.get_index() as u8);

        let section_count = (height / 16) as usize;
        let changed_blocks_per_section = (0..section_count)
            .map(|_| SyncMutex::new(FxHashSet::default()))
            .collect::<Box<[_]>>();

        Self {
            data: OnceLock::new(),
            status: AtomicWaitQueue::new(u16::from(UNPUBLISHED_STATUS)),
            drive: GenerationDrive::new(),
            drive_gauge: AtomicU8::new(DRIVE_GAUGE_NONE),
            stall_attempts: AtomicU32::new(0),
            status_changed: Notify::new(),
            generation_task: SyncMutex::new(None),
            generation_task_target: AtomicU8::new(STATUS_NONE),
            pos,
            load_level: AtomicU8::new(load_level.raw()),
            simulation_level: AtomicU8::new(optional_ticket_level_raw(simulation_level)),
            started_work: AtomicUsize::new(usize::MAX),
            active_save_dependencies: AtomicUsize::new(0),
            save_lifecycle: AtomicU8::new(SAVE_LIFECYCLE_ACTIVE),
            highest_allowed_status: AtomicU8::new(highest_allowed_status),
            min_y,
            height,
            has_changed_sections: AtomicBool::new(false),
            queued_for_broadcast: AtomicBool::new(false),
            packet_content_revision: AtomicU64::new(0),
            ticking_readiness: AtomicU64::new(0),
            full_status_initialized: AtomicBool::new(false),
            full_publications,
            generation_inbox,
            changed_blocks_per_section,
            changed_light_sections: SyncMutex::new(ChangedLightSectionSets::default()),
        }
    }

    /// Returns the current load ticket level.
    pub fn load_level(&self) -> Option<ChunkTicketLevel> {
        optional_ticket_level_from_raw(self.load_level.load(Ordering::Relaxed))
    }

    /// Stores the current load ticket level and returns the previous level.
    pub(crate) fn swap_load_level(&self, level: ChunkTicketLevel) -> Option<ChunkTicketLevel> {
        optional_ticket_level_from_raw(self.load_level.swap(level.raw(), Ordering::Relaxed))
    }

    /// Clears the current load ticket level.
    pub(crate) fn clear_load_level(&self) {
        self.load_level.store(NO_TICKET_LEVEL, Ordering::Relaxed);
    }

    /// Returns the current simulation ticket level.
    pub fn simulation_level(&self) -> Option<ChunkTicketLevel> {
        optional_ticket_level_from_raw(self.simulation_level.load(Ordering::Relaxed))
    }

    /// Stores the current simulation ticket level.
    pub(crate) fn set_simulation_level(&self, level: Option<ChunkTicketLevel>) {
        self.simulation_level
            .store(optional_ticket_level_raw(level), Ordering::Relaxed);
    }

    pub(crate) fn entity_visibility(&self) -> EntityVisibility {
        if self.try_chunk(ChunkStatus::Full).is_none() {
            return EntityVisibility::Hidden;
        }

        if !self.load_level().is_some_and(is_full) {
            return EntityVisibility::Hidden;
        }

        if is_entity_ticking(self.simulation_level())
            && self.ticking_readiness_snapshot().is_entity_ticking()
        {
            EntityVisibility::Ticking
        } else {
            EntityVisibility::Tracked
        }
    }

    /// Updates the highest allowed generation status based on the ticket level.
    ///
    /// `SeqCst`, not `Release`, and every load of this cell is `SeqCst` for the
    /// same reason. The drive protocol pairs "store the new allowance, then arm
    /// the drive" on the ticket side against "end the run, then read the
    /// allowance" on the driver side. That is a Dekker pattern: it needs
    /// Store-Load ordering, which Release/Acquire does not give -- both sides
    /// may legally read the other's pre-store value, each concludes the other
    /// will do the work, and the chunk sits below its allowed status forever.
    /// Only a single total order over the two accesses rules that out.
    pub fn update_highest_allowed_status(&self, ticket_level: Option<ChunkTicketLevel>) {
        let new_status =
            generation_status(ticket_level).map_or(STATUS_NONE, |s| s.get_index() as u8);
        self.highest_allowed_status
            .store(new_status, Ordering::SeqCst);
    }

    /// The highest status generation is currently allowed to reach, or `None`
    /// while the chunk's ticket level allows no generation at all.
    ///
    /// # Panics
    ///
    /// Panics if the stored allowance does not decode to a status. Only
    /// [`Self::update_highest_allowed_status`] writes this cell, and only from
    /// `generation_status`, so an undecodable value means an aliasing write --
    /// and every caller here would otherwise silently read it as "generation is
    /// forbidden" and abandon the chunk.
    #[must_use]
    pub fn highest_allowed_status(&self) -> Option<ChunkStatus> {
        // See `update_highest_allowed_status` for why this is `SeqCst`.
        let raw = self.highest_allowed_status.load(Ordering::SeqCst);
        if raw == STATUS_NONE {
            return None;
        }
        let status = ChunkStatus::from_index(usize::from(raw));
        assert!(status.is_some(), "invalid highest allowed status: {raw}");
        status
    }

    /// Whether this holder has generation work left to do.
    ///
    /// The allowance is read once, into `allowed`, and the target is not
    /// reconstructed from the ticket level: a level read paired with a separate
    /// allowance read can straddle an update and produce a target that was never
    /// allowed (see `ChunkMap::schedule_admitted_holders`, which re-reads the
    /// level for that reason). This cell is authoritative on its own.
    #[must_use]
    pub fn needs_generation(&self) -> bool {
        let Some(allowed) = self.highest_allowed_status() else {
            return false;
        };
        self.published_status()
            .is_none_or(|published| published < allowed)
    }

    /// Records a block change at the given position.
    /// Returns `true` if this is the first change (chunk should be added to broadcast list).
    pub fn block_changed(&self, pos: BlockPos) -> bool {
        if !self.ticking_readiness_snapshot().is_block_ticking()
            || pos.0.y < self.min_y
            || pos.0.y >= self.min_y + self.height
        {
            return false;
        }

        let section_index = ((pos.0.y - self.min_y) / 16) as usize;
        if section_index >= self.changed_blocks_per_section.len() {
            return false;
        }

        let packed = SectionPos::section_relative_pos(pos);
        self.changed_blocks_per_section[section_index]
            .lock()
            .insert(packed);
        self.mark_packet_content_changed();
        self.has_changed_sections.store(true, Ordering::Release);

        !self.queued_for_broadcast.swap(true, Ordering::AcqRel)
    }

    /// Records a light-section change for a full chunk and marks saved light data dirty.
    ///
    /// Returns `true` if this is the first pending broadcast change for the chunk holder.
    pub fn light_changed(&self, layer: LightLayer, section_pos: SectionPos) -> bool {
        let Some(ready_for_packet) = self.mark_valid_light_section_dirty(section_pos) else {
            return false;
        };
        if !ready_for_packet {
            return false;
        }
        self.mark_packet_content_changed();

        let inserted = {
            let mut guard = self.changed_light_sections.lock();
            match layer {
                LightLayer::Sky => guard.sky.insert(section_pos),
                LightLayer::Block => guard.block.insert(section_pos),
            }
        };

        if !inserted {
            return false;
        }

        !self.queued_for_broadcast.swap(true, Ordering::AcqRel)
    }

    /// Marks saved light data dirty without queuing client-visible changes.
    pub fn mark_light_section_dirty(&self, section_pos: SectionPos) -> bool {
        self.mark_valid_light_section_dirty(section_pos).is_some()
    }

    fn mark_valid_light_section_dirty(&self, section_pos: SectionPos) -> Option<bool> {
        if section_pos.x() != self.pos.0.x || section_pos.z() != self.pos.0.y {
            return None;
        }

        let Ok(range) = LightSectionRange::from_world_height(self.min_y, self.height) else {
            return None;
        };
        range.section_index(section_pos.y())?;

        let status = self.published_status()?;
        let chunk = self.data.get()?;
        chunk.mark_dirty();
        Some(status == ChunkStatus::Full && self.ticking_readiness_snapshot().is_block_ticking())
    }

    /// Returns whether there are pending changes to broadcast.
    pub fn has_changes_to_broadcast(&self) -> bool {
        self.queued_for_broadcast.load(Ordering::Acquire)
    }

    /// Allows later changes to enqueue this holder for a future broadcast.
    pub fn clear_broadcast_queued(&self) {
        self.queued_for_broadcast.store(false, Ordering::Release);
    }

    /// Takes all pending block changes, grouped by section index.
    /// Returns a vec of (`section_index`, set of packed positions).
    pub fn take_changed_blocks(&self) -> Vec<(usize, FxHashSet<PackedSectionBlockPos>)> {
        if !self.has_changed_sections.swap(false, Ordering::AcqRel) {
            return Vec::new();
        }

        let mut result = Vec::new();
        for (section_index, section_changes) in self.changed_blocks_per_section.iter().enumerate() {
            let mut guard = section_changes.lock();
            if !guard.is_empty() {
                result.push((section_index, mem::take(&mut *guard)));
            }
        }
        result
    }

    /// Takes all pending light-section changes.
    pub fn take_changed_light_sections(&self) -> ChangedLightSections {
        let mut guard = self.changed_light_sections.lock();
        ChangedLightSections {
            sky: guard.sky.drain().collect(),
            block: guard.block.drain().collect(),
        }
    }

    /// Marks the holder's client-visible chunk packet content as changed.
    pub fn mark_packet_content_changed(&self) {
        self.packet_content_revision.fetch_add(1, Ordering::AcqRel);
    }

    /// Returns the current client-visible content revision.
    pub fn packet_content_revision(&self) -> u64 {
        self.packet_content_revision.load(Ordering::Acquire)
    }

    #[must_use]
    pub(crate) fn ticking_readiness_snapshot(&self) -> TickingReadinessSnapshot {
        TickingReadinessSnapshot(self.ticking_readiness.load(Ordering::Acquire))
    }

    #[must_use]
    pub(crate) fn is_full_status_initialized(&self) -> bool {
        self.full_status_initialized.load(Ordering::Acquire)
    }

    pub(crate) fn transition_ticking_readiness(
        &self,
        target: TickingReadiness,
    ) -> Option<TickingReadiness> {
        let mut current = self.ticking_readiness.load(Ordering::Acquire);
        loop {
            let snapshot = TickingReadinessSnapshot(current);
            let previous = snapshot.readiness();
            if previous == target {
                return None;
            }

            let generation = current >> 2;
            assert!(
                generation != u64::MAX >> 2,
                "chunk ticking readiness generation exhausted"
            );
            let next_generation = generation + 1;
            let next = (next_generation << 2) | target as u64;
            match self.ticking_readiness.compare_exchange(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(previous),
                Err(observed) => current = observed,
            }
        }
    }

    /// Returns the number of sections in this chunk.
    pub fn section_count(&self) -> usize {
        self.changed_blocks_per_section.len()
    }

    /// Checks if the given status is disallowed.
    pub fn is_status_disallowed(&self, status: ChunkStatus) -> bool {
        // Goes through the accessor so this shares its single `SeqCst` read of
        // the cell; a second, differently ordered read of the same allowance is
        // exactly what the Dekker pairing cannot tolerate.
        self.highest_allowed_status()
            .is_none_or(|allowed| status > allowed)
    }

    /// Schedules a generation task for this chunk if needed.
    ///
    /// Returns `true` if a new task was actually scheduled, `false` if the chunk
    /// already has a suitable task or is already at the target status.
    #[inline]
    pub(crate) fn schedule_chunk_generation_task_b(
        &self,
        status: ChunkStatus,
        chunk_map: &Arc<ChunkMap>,
    ) -> bool {
        if self.is_status_disallowed(status) {
            return false;
        }

        if self.try_chunk(status).is_some() {
            return false;
        }

        let status_index = status.get_index() as u8;
        let current_target = self.generation_task_target.load(Ordering::Acquire);
        if current_target != STATUS_NONE && status_index <= current_target {
            return false;
        }

        let task = self.generation_task.lock();

        if task
            .as_ref()
            .is_some_and(|task| status <= task.target_status)
        {
            return false;
        }

        drop(task);
        self.reschedule_chunk_task_b(status, chunk_map);
        true
    }

    /// Reschedules the chunk task to the given status.
    #[inline]
    pub(crate) fn reschedule_chunk_task_b(&self, status: ChunkStatus, chunk_map: &Arc<ChunkMap>) {
        let new_task = chunk_map.schedule_generation_task_b(status, self.pos);
        let mut old_task_guard = self.generation_task.lock();

        let old_task = old_task_guard.replace(new_task);
        self.generation_task_target
            .store(status.get_index() as u8, Ordering::Release);
        drop(old_task_guard);

        if let Some(old_task) = old_task {
            old_task.cancel();
        }

        chunk_map.notify_generation_refill();
    }

    /// Gets access to the chunk if it has reached the given status.
    #[inline]
    pub fn try_chunk(&self, status: ChunkStatus) -> Option<&Chunk> {
        let published = self.encoded_status();
        (published >= encoded_published_status(status))
            .then(|| self.data.get())
            .flatten()
    }

    /// Gets the Full-only capability after Full status is published.
    #[must_use]
    pub fn try_full_chunk(&self) -> Option<FullChunkRef<'_>> {
        self.try_chunk(ChunkStatus::Full)
            .map(FullChunkRef::from_full_context)
    }


    /// Waits until the chunk has reached the given status without reading chunk data.
    /// Retained with no production caller on purpose: its two tests
    /// (`status_waiter_observes_publication_after_subscribing`,
    /// `pending_status_waiters_wake_after_publication`) are the only coverage of
    /// the publish/wake protocol that the `AtomicWaitQueue` migration replaces,
    /// so they are the regression net for that change.
    pub async fn await_chunk_status(&self, status: ChunkStatus) -> Option<ChunkStatus> {
        self.await_status_with(status, |_| false).await
    }

    async fn await_claimed_chunk_status(&self, status: ChunkStatus) -> Option<ChunkStatus> {
        self.await_status_with(status, |holder| !holder.status_work_covers(status))
            .await
    }

    /// Waits for `status` to be published, or for `bail` to become true.
    ///
    /// Two different events are in play and only one of them is a status raise.
    /// The queue delivers the raise exactly once, to exactly the waiters it
    /// satisfies. The bail conditions -- generation becoming disallowed, or a
    /// work claim being rolled back -- do not move the status, so they arrive on
    /// `status_changed` and are re-checked on each wake.
    ///
    /// Ordering matters: the waiter is registered *before* the bail conditions
    /// are tested, so a bail that lands between the test and the registration
    /// still wakes it.
    async fn await_status_with<F>(&self, status: ChunkStatus, bail: F) -> Option<ChunkStatus>
    where
        F: Fn(&Self) -> bool,
    {
        loop {
            let bailed = self.status_changed.notified();
            let (sender, receiver) = oneshot::channel();

            // The queue releases a waiter once its status is *exceeded*, so a
            // waiter for encoded status `e` registers at `e - 1`.
            let wait_for = u16::from(encoded_published_status(status)) - 1;
            match self.status.wait(wait_for, StatusWaiter::Oneshot(sender)) {
                WaitOutcome::AlreadySatisfied(_) => return self.published_status(),
                WaitOutcome::Cancelled(_) => return None,
                WaitOutcome::Registered => {}
            }

            if self.is_status_disallowed(status) || bail(self) {
                return None;
            }

            tokio::select! {
                _ = receiver => return self.published_status(),
                () = bailed => {}
            }
        }
    }

    /// Gets the published status of the chunk.
    pub fn published_status(&self) -> Option<ChunkStatus> {
        decoded_published_status(self.encoded_status())
    }

    /// The raw encoded status, as stored in the wait queue's status word.
    #[inline]
    fn encoded_status(&self) -> u8 {
        // The queue is never cancelled today (see `wake_all_watchers`), so
        // `None` cannot occur; treat it as unpublished rather than panicking.
        self.status.status().map_or(UNPUBLISHED_STATUS, |encoded| {
            u8::try_from(encoded).unwrap_or(UNPUBLISHED_STATUS)
        })
    }

    /// Returns whether vanilla timed tickets may age for this chunk.
    #[must_use]
    pub fn is_ready_for_saving(&self) -> bool {
        self.active_save_dependencies.load(Ordering::Acquire) == 0
    }

    pub(crate) fn add_save_dependency(self: &Arc<Self>) -> ChunkSaveDependency {
        self.active_save_dependencies.fetch_add(1, Ordering::AcqRel);
        ChunkSaveDependency {
            holder: Arc::clone(self),
        }
    }

    /// Moves an active holder into the unloading lifecycle.
    pub(crate) fn begin_unloading(&self) {
        let result = self.save_lifecycle.compare_exchange(
            SAVE_LIFECYCLE_ACTIVE,
            SAVE_LIFECYCLE_UNLOADING,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert!(
            result.is_ok(),
            "an active chunk holder entered unloading from an invalid lifecycle"
        );
    }

    /// Reserves the unloading holder while its immutable save input is assembled.
    pub(crate) fn try_begin_save_preparation(
        self: &Arc<Self>,
    ) -> Option<ChunkSavePreparationGuard> {
        self.save_lifecycle
            .compare_exchange(
                SAVE_LIFECYCLE_UNLOADING,
                SAVE_LIFECYCLE_PREPARING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| ChunkSavePreparationGuard {
                holder: Arc::clone(self),
            })
    }

    /// Attempts to reactivate an unloading holder without waiting for save preparation.
    pub(crate) fn try_revive_from_unloading(&self) -> bool {
        self.save_lifecycle
            .compare_exchange(
                SAVE_LIFECYCLE_UNLOADING,
                SAVE_LIFECYCLE_ACTIVE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Applies a step to the chunk.
    ///
    /// Cancellation is handled structurally by the owning generation task: its
    /// `run` loop races the whole `join_all` of dependency-wait futures against
    /// its cancel token and drops them on cancellation, so the returned futures
    /// don't each re-check it. A failed dependency surfaces as
    /// `await_chunk_status` returning `None`.
    ///
    /// # Panics
    /// Panics if the target status is not Empty and has no parent, or if the
    /// chunk status is invalid during generation.
    pub fn apply_step(
        self: &Arc<Self>,
        step: &'static ChunkStep,
        chunk_map: &Arc<ChunkMap>,
        cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
        thread_pool: Arc<rayon::ThreadPool>,
    ) -> Option<NeighborReady> {
        let target_status = step.target_status;

        if self.is_status_disallowed(target_status) {
            return None;
        }

        if target_status == ChunkStatus::Light {
            let light_work_window_gate = chunk_map.light_work_window_gate();
            let Some(light_work_window_reservation) =
                light_work_window_gate.try_reserve_centered(self.pos)
            else {
                return Some(Self::await_light_work_window_and_apply_step(
                    Arc::clone(self),
                    step,
                    Arc::clone(chunk_map),
                    Arc::clone(cache),
                    thread_pool,
                    light_work_window_gate,
                ));
            };

            return self.apply_step_with_light_work_window_reservation(
                step,
                chunk_map,
                cache,
                thread_pool,
                Some(light_work_window_reservation),
            );
        }

        self.apply_step_with_light_work_window_reservation(
            step,
            chunk_map,
            cache,
            thread_pool,
            None,
        )
    }

    fn await_light_work_window_and_apply_step(
        holder: Arc<Self>,
        step: &'static ChunkStep,
        chunk_map: Arc<ChunkMap>,
        cache: Arc<StaticCache2D<Arc<ChunkHolder>>>,
        thread_pool: Arc<rayon::ThreadPool>,
        light_work_window_gate: Arc<LightWorkWindowGate>,
    ) -> NeighborReady {
        Box::pin(async move {
            // The per-holder drive awaits this inline, holding its admission
            // permit, and that is accepted cost rather than an oversight. It is
            // deadlock-safe because the gate is only ever released by a job that
            // is *running* -- it already holds a permit and is making progress
            // -- never by something waiting to be admitted. Turning the wait
            // into a park is not available: `reserve_centered_with` grants
            // inline on the releasing thread, the gate has no deregistration
            // path, and a grant handed to a holder that has moved on re-enters
            // `Drop` -> `grant_unblocked` recursively. This gauge is what says
            // how much admission capacity the choice costs.
            let light_work_window_reservation = {
                // Scoped through a guard, not a bare pair of bumps: the task
                // model drops these futures on cancellation, and a decrement
                // skipped that way would drift the gauge permanently.
                let _waiting = LightWindowWaitGauge::new();
                light_work_window_gate.reserve_centered(holder.pos).await
            };
            let ready = holder.apply_step_with_light_work_window_reservation(
                step,
                &chunk_map,
                &cache,
                thread_pool,
                Some(light_work_window_reservation),
            )?;
            ready.await
        })
    }

    fn apply_step_with_light_work_window_reservation(
        self: &Arc<Self>,
        step: &'static ChunkStep,
        chunk_map: &Arc<ChunkMap>,
        cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
        thread_pool: Arc<rayon::ThreadPool>,
        light_work_window_reservation: Option<LightWorkWindowReservation>,
    ) -> Option<NeighborReady> {
        let target_status = step.target_status;
        debug_assert!(
            target_status != ChunkStatus::Light || light_work_window_reservation.is_some()
        );

        if self.is_status_disallowed(target_status) {
            return None;
        }

        let Some(status_claim) = self.claim_status_work(target_status) else {
            // Another task is already generating this chunk to `target_status`;
            // just wait for it. Parent cancellation is handled by the owning
            // task's run loop dropping this future; a failed dependency returns
            // `None` from `await_claimed_chunk_status`.
            //
            // Under the per-holder drive this branch is unreachable: a holder
            // has exactly one driver, and it is inside this call. Counted, not
            // asserted, because the same holder is one `claim_status_work` away
            // from the panic that aborts the server, and the counter says
            // whether the invariant held before that happens.
            if *STAGE1 {
                GENERATION_DRIVE_COUNTERS
                    .contended_status_claims
                    .fetch_add(1, Ordering::Relaxed);
            }
            let self_clone = self.clone();
            return Some(Box::pin(async move {
                self_clone
                    .await_claimed_chunk_status(target_status)
                    .await
                    .map(|_| ())
            }));
        };

        // Extend the claim into a fused run.
        //
        // Consecutive steps that need nothing from other chunks beyond what
        // this one already needed can run back-to-back in a single job, which
        // saves a rayon dispatch, a tokio task and a oneshot wake apiece and
        // keeps the chunk's data in cache across them. Claiming the whole run
        // up front is safe because a claim only advances `started_work`;
        // publishing is separate and still happens per status, so a neighbour
        // waiting on the first status of a run is not held up by the rest.
        //
        // The caller keeps asking for one status at a time. Those it finds
        // already claimed and published fall into the branch above and resolve
        // without dispatching anything.
        let mut claims = vec![status_claim];
        let mut steps = vec![step];
        if target_status != ChunkStatus::Empty {
            let mut previous = step;
            while let Some(next_status) = previous.target_status.next() {
                let next_step = GENERATION_PYRAMID.get_step_to(next_status);
                if !can_fuse(previous, next_step) || self.is_status_disallowed(next_status) {
                    break;
                }
                let Some(claim) = self.claim_status_work(next_status) else {
                    break;
                };
                claims.push(claim);
                steps.push(next_step);
                previous = next_step;
            }
        }

        let cache = cache.clone();
        let context = chunk_map.world_gen_context.clone();
        let self_clone = self.clone();
        let storage = chunk_map.storage.clone();
        let save_dependency = self.add_save_dependency();

        let future = chunk_map.task_tracker.spawn(async move {
            // Keep the claims alive for the producer task so Drop can roll back
            // abandoned work. Rolling several back is safe in any order: each
            // rolls `started_work` to the published index, so whichever runs
            // while it still matches wins and the rest are no-ops.
            let _status_claims = claims;
            let _save_dependency = save_dependency;
            let result = if target_status == ChunkStatus::Empty {
                Self::apply_empty_step(self_clone, step, context, cache, storage, thread_pool).await
            } else {
                Self::apply_generated_steps(
                    self_clone,
                    steps,
                    context,
                    cache,
                    thread_pool,
                    light_work_window_reservation,
                )
                .await
            };

            #[cfg(feature = "slow_chunk_gen")]
            if result.is_some() && SLOW_CHUNK_GEN.load(Ordering::Relaxed) {
                sleep(Duration::from_millis(200)).await;
            }

            result
        });

        Some(Box::pin(async move {
            match future.await {
                Ok(result) => result,
                Err(e) => {
                    log::error!("Chunk generation task panicked: {e}");
                    None
                }
            }
        }))
    }

    async fn apply_empty_step(
        holder: Arc<Self>,
        step: &'static ChunkStep,
        context: Arc<WorldGenContext>,
        cache: Arc<StaticCache2D<Arc<ChunkHolder>>>,
        storage: Arc<ChunkStorage>,
        thread_pool: Arc<rayon::ThreadPool>,
    ) -> Option<()> {
        let target_status = step.target_status;
        let chunk_exists = match storage.acquire_chunk(holder.pos).await {
            Ok(chunk_exists) => chunk_exists,
            Err(error) => {
                tracing::error!(
                    chunk = ?holder.pos,
                    "Failed to acquire chunk storage before load/generation: {error}",
                );
                return None;
            }
        };

        if holder.is_status_disallowed(target_status) {
            tracing::debug!(
                chunk = ?holder.pos,
                ?target_status,
                load_level = ?holder.load_level(),
                simulation_level = ?holder.simulation_level(),
                current_status = ?holder.published_status(),
                "Dropping storage load after chunk holder target became disallowed before load/generation: chunk={:?}, target_status={:?}, load_level={:?}, simulation_level={:?}, current_status={:?}",
                holder.pos,
                target_status,
                holder.load_level(),
                holder.simulation_level(),
                holder.published_status(),
            );
            if let Err(error) = storage.release_chunk(holder.pos).await {
                tracing::error!(
                    chunk = ?holder.pos,
                    "Failed to release canceled chunk storage task: {error}",
                );
            }
            return None;
        }

        if chunk_exists {
            match Self::apply_existing_empty_step(
                &holder,
                target_status,
                &context,
                &storage,
                &thread_pool,
            )
            .await
            {
                Some(true) => return Some(()),
                Some(false) => {}
                None => return None,
            }
        }

        if holder.is_status_disallowed(target_status) {
            tracing::debug!(
                chunk = ?holder.pos,
                ?target_status,
                load_level = ?holder.load_level(),
                simulation_level = ?holder.simulation_level(),
                current_status = ?holder.published_status(),
                "Dropping storage load after chunk holder target became disallowed after load attempt: chunk={:?}, target_status={:?}, load_level={:?}, simulation_level={:?}, current_status={:?}",
                holder.pos,
                target_status,
                holder.load_level(),
                holder.simulation_level(),
                holder.published_status(),
            );
            if let Err(error) = storage.release_chunk(holder.pos).await {
                tracing::error!(
                    chunk = ?holder.pos,
                    "Failed to release canceled chunk storage task: {error}",
                );
            }
            return None;
        }

        let holder_for_notify = holder.clone();
        let world = context.world();
        let pos = holder.pos;
        Self::run_step_task(thread_pool, step, context, cache, holder, move || {
            holder_for_notify.finish_generation_status(target_status);
        })
        .await;
        if target_status == ChunkStatus::Empty {
            world.on_entity_chunk_loaded(pos);
        }
        Some(())
    }

    async fn apply_existing_empty_step(
        holder: &Arc<Self>,
        target_status: ChunkStatus,
        context: &Arc<WorldGenContext>,
        storage: &Arc<ChunkStorage>,
        thread_pool: &rayon::ThreadPool,
    ) -> Option<bool> {
        let loaded = match storage
            .load_chunk(
                holder.pos,
                holder.min_y(),
                holder.height(),
                context.weak_world(),
                thread_pool,
            )
            .await
        {
            Ok(Some(loaded)) => loaded,
            Ok(None) => {
                tracing::warn!(
                    chunk = ?holder.pos,
                    "Chunk storage entry disappeared or was discarded as corrupt; regenerating it",
                );
                return Some(false);
            }
            Err(error) => {
                tracing::error!(
                    chunk = ?holder.pos,
                    "Failed to load existing chunk; aborting generation to avoid overwriting saved data: {error}",
                );
                if let Err(release_error) = storage.release_chunk(holder.pos).await {
                    tracing::error!(
                        chunk = ?holder.pos,
                        "Failed to release chunk storage after load failure: {release_error}",
                    );
                }
                return None;
            }
        };

        let loaded_status = loaded.status;
        if holder.is_status_disallowed(target_status) {
            tracing::debug!(
                chunk = ?holder.pos,
                ?target_status,
                ?loaded_status,
                load_level = ?holder.load_level(),
                simulation_level = ?holder.simulation_level(),
                current_status = ?holder.published_status(),
                "Dropping storage load that completed after chunk holder target became disallowed: chunk={:?}, target_status={:?}, loaded_status={:?}, load_level={:?}, simulation_level={:?}, current_status={:?}",
                holder.pos,
                target_status,
                loaded_status,
                holder.load_level(),
                holder.simulation_level(),
                holder.published_status(),
            );
            if let Err(error) = storage.release_chunk(holder.pos).await {
                tracing::error!(
                    chunk = ?holder.pos,
                    "Failed to release canceled chunk storage load: {error}",
                );
            }
            return None;
        }

        holder.store_and_publish_chunk_status(loaded.chunk, loaded_status);
        let world = context.world();
        world.on_entity_chunk_loaded(holder.pos);
        world.update_entity_chunk_visibility(holder.pos, holder.entity_visibility());
        if !loaded.pending_entities.is_empty() {
            world.register_loaded_chunk_entities(
                holder.pos,
                loaded_status,
                loaded.pending_entities,
            );
        }
        if loaded_status == ChunkStatus::Full {
            holder.publish_full();
        }
        Some(true)
    }

    /// Runs one generation step on the generation pool.
    ///
    /// `on_complete` runs on the same worker as soon as the step returns. Only
    /// the `Empty` step uses this; generated steps go through
    /// [`Self::apply_generated_steps`], which fuses consecutive steps into one
    /// job.
    async fn run_step_task<F>(
        thread_pool: Arc<rayon::ThreadPool>,
        step: &'static ChunkStep,
        context: Arc<WorldGenContext>,
        cache: Arc<StaticCache2D<Arc<ChunkHolder>>>,
        holder: Arc<Self>,
        on_complete: F,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let task = step.task;
        rayon_spawn(&thread_pool, move || {
            task(context, step, &cache, holder);
            on_complete();
        })
        .await;
    }

    /// Runs a fused run of generation steps as one job on the generation pool.
    ///
    /// `steps` is one or more consecutive steps whose claims are already held;
    /// see [`can_fuse`] for when a step may join a run.
    async fn apply_generated_steps(
        holder: Arc<Self>,
        steps: Vec<&'static ChunkStep>,
        context: Arc<WorldGenContext>,
        cache: Arc<StaticCache2D<Arc<ChunkHolder>>>,
        thread_pool: Arc<rayon::ThreadPool>,
        light_work_window_reservation: Option<LightWorkWindowReservation>,
    ) -> Option<()> {
        let first = *steps.first().expect("a fused run has at least one step");
        let Some(parent_status) = first.target_status.parent() else {
            panic!("Target status must have parent if not Empty");
        };
        let has_parent = holder
            .published_status()
            .is_some_and(|status| parent_status <= status);

        assert!(has_parent, "Parent chunk missing");

        rayon_spawn(&thread_pool, move || {
            let mut reservation = light_work_window_reservation;
            for step in steps {
                let task = step.task;
                task(Arc::clone(&context), step, &cache, Arc::clone(&holder));

                // Publish on the generation worker that just did the work
                // rather than after waking a task back up. Every chunk whose
                // next step depends on this status is blocked until the publish
                // lands, so routing it through a oneshot wake put a
                // cross-runtime scheduler round trip in the critical path of
                // every step of every chunk.
                holder.finish_generation_status(step.target_status);

                if step.target_status == ChunkStatus::Light {
                    // Release the light window as soon as the light work is
                    // done instead of holding it across the rest of the run.
                    reservation = None;
                }
            }
            drop(reservation);
        })
        .await;
        Some(())
    }

    fn claim_status_work(self: &Arc<Self>, status: ChunkStatus) -> Option<StatusWorkClaim> {
        let status_index = status.get_index();
        let parent_index = status.parent().map_or(usize::MAX, ChunkStatus::get_index);

        let previous_started = self.started_work.compare_exchange(
            parent_index,
            status_index,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );

        match previous_started {
            Ok(_) => Some(StatusWorkClaim::new(Arc::clone(self), status)),
            Err(current) => {
                if current != usize::MAX && current >= status_index {
                    None
                } else {
                    panic!(
                        "Unexpected started work status: {current:?} (index {current}) while trying to start: {status:?} (index {status_index})"
                    );
                }
            }
        }
    }

    fn release_status_work_claim(self: &Arc<Self>, status: ChunkStatus) {
        let status_index = status.get_index();
        let rollback_index = self
            .published_status()
            .map_or(usize::MAX, ChunkStatus::get_index);

        if rollback_index != usize::MAX && rollback_index >= status_index {
            return;
        }

        if self
            .started_work
            .compare_exchange(
                status_index,
                rollback_index,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            self.wake_all_watchers();

            // Deliberately no re-arm here under the per-holder drive, however
            // much a rollback looks like the place for one.
            //
            // A claim only ever lives inside the tokio task
            // `apply_step_with_light_work_window_reservation` spawns, and the
            // drive is parked on that task's `JoinHandle`. Tokio drops a task's
            // future -- and with it these claims -- before it resolves the
            // handle, so a rollback is always observed while the drive that
            // dispatched the run is still `Running`. `GenerationDrive::arm` on a
            // running drive queues nothing and *bumps the epoch*, which spends
            // the run ticket the drive is about to stall on: `to_stalled` then
            // refuses, the `JobFailed` stall is skipped along with its backoff,
            // and the drive loops straight back into re-dispatching the step
            // that just failed -- a hot retry pinning an admission permit,
            // measured as ticket 1 -> 2 and `to_stalled == false` on a holder
            // whose `Empty` claim was dropped unpublished.
            //
            // Nothing is lost by staying quiet: the drive is still inside the
            // loop, and it either re-evaluates or stalls with backoff. The one
            // case where nobody is left to look at the holder is a drive future
            // dropped mid-run, and an `arm` cannot rescue that either -- the
            // drive is stranded `Running`, so `arm` returns `false` there too.
            // `DriveDropGuard` reports it instead.
        }
    }

    fn mark_status_work_published(&self, status: ChunkStatus) {
        let status_index = status.get_index();
        let mut current = self.started_work.load(Ordering::Acquire);

        loop {
            if current != usize::MAX && current >= status_index {
                return;
            }

            match self.started_work.compare_exchange(
                current,
                status_index,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return,
                Err(next) => current = next,
            }
        }
    }

    fn status_work_covers(&self, status: ChunkStatus) -> bool {
        let current = self.started_work.load(Ordering::Acquire);
        current != usize::MAX && current >= status.get_index()
    }

    /// Upgrades the chunk to a full chunk.
    ///
    /// If the chunk is already Full (e.g., loaded from disk), this is a no-op.
    ///
    /// # Panics
    /// Panics if no chunk has been installed or Full runtime initialization repeats.
    pub(crate) fn upgrade_to_full(&self) {
        if self.published_status() == Some(ChunkStatus::Full) {
            return;
        }
        let Some(chunk) = self.data.get() else {
            panic!("cannot promote an uninitialized chunk holder");
        };
        let FullChunkPromotion {
            chunk: full,
            pending_entities,
        } = chunk.promote_to_full();
        let promoted_entities = Some((full.get_level(), chunk.pos, pending_entities));
        if let Some((world, pos, pending_entities)) = promoted_entities
            && let Some(world) = world
        {
            world.register_loaded_chunk_entities(pos, ChunkStatus::Full, pending_entities);
        }
    }

    /// Runs Full-load post-processing and returns the number of packed positions attempted.
    pub(crate) fn post_process_generation(&self) -> Result<usize, PostProcessGenerationError> {
        let postprocessing = {
            let Some(full) = self.try_full_chunk() else {
                return Err(PostProcessGenerationError::ChunkNotFull);
            };
            let world = full
                .get_level()
                .ok_or(PostProcessGenerationError::WorldUnavailable)?;
            full.take_postprocessing()
                .map(|postprocessing| (world, full.common().pos, full.min_y(), postprocessing))
        };

        let post_process_position_count =
            if let Some((world, pos, min_y, postprocessing)) = postprocessing {
                let position_count = postprocessing.iter().map(Vec::len).sum();
                FullChunkRef::post_process_generation(&world, pos, min_y, postprocessing);
                position_count
            } else {
                0
            };
        let Some(full) = self.try_full_chunk() else {
            return Err(PostProcessGenerationError::ChunkNotFull);
        };
        full.promote_pending_block_entities();
        Ok(post_process_position_count)
    }

    /// Finishes a generated status on the async scheduler after the Rayon task returns.
    fn finish_generation_status(self: &Arc<Self>, status: ChunkStatus) {
        if let Some(stored_chunk) = self.data.get()
            && self
                .published_status()
                .is_none_or(|published| published < status)
        {
            stored_chunk.mark_dirty();
        }

        if status == ChunkStatus::Full {
            self.register_full_chunk_ticks();
        }

        self.mark_status_work_published(status);
        self.publish_generated_status(status);

        if status == ChunkStatus::Full {
            self.publish_full();
        }
    }

    #[cfg(test)]
    pub(crate) fn finish_generation_status_for_test(self: &Arc<Self>, status: ChunkStatus) {
        self.finish_generation_status(status);
    }

    /// Inserts a chunk into the holder with a specific status.
    /// This notifies watchers - use `insert_chunk_no_notify` + separate notification
    /// if calling from a rayon thread to avoid contention.
    ///
    /// # Panics
    ///
    /// Panics if `status` claims Full without initialized Full runtime state, or if
    /// initialized Full runtime state is paired with a lower status.
    pub fn insert_chunk(self: &Arc<Self>, chunk: Chunk, status: ChunkStatus) {
        self.store_and_publish_chunk_status(chunk, status);
        if status == ChunkStatus::Full {
            self.publish_full();
        }
    }

    fn store_and_publish_chunk_status(&self, chunk: Chunk, status: ChunkStatus) {
        assert_eq!(
            self.encoded_status(),
            UNPUBLISHED_STATUS,
            "initial chunk installation cannot replace published data"
        );
        assert_eq!(
            status == ChunkStatus::Full,
            chunk.full_runtime().is_some(),
            "initial chunk status must match its Full runtime state"
        );
        assert!(
            self.data.set(chunk).is_ok(),
            "initial chunk installation cannot replace existing data"
        );
        if status == ChunkStatus::Full {
            self.register_full_chunk_ticks();
        }
        self.mark_status_work_published(status);
        // A disk load jumps straight from unpublished to the loaded status, so
        // this raise can skip several statuses at once. The queue releases every
        // waiter it passes, not just the next one.
        self.raise_published_status(encoded_published_status(status));
    }

    fn publish_generated_status(&self, status: ChunkStatus) {
        self.raise_published_status(encoded_published_status(status));
    }

    /// Publishes an encoded status and releases the waiters it satisfies.
    ///
    /// Idempotent: a raise to a status already reached is dropped, matching the
    /// `fetch_max` this replaced. The queue asserts monotonicity in debug.
    fn raise_published_status(&self, encoded: u8) {
        if encoded <= self.encoded_status() {
            return;
        }
        // Stack-local, and published in one go below. This runs on the rayon
        // generation worker that just did the work, once per status of a fused
        // run, with every chunk waiting on this status blocked behind it:
        // pushing each holder into the map's inbox as it comes off the queue
        // would take and drop that lock once per waiter, and a wide ring can
        // release hundreds at once. Nothing here may `tokio::spawn` either --
        // a rayon worker has no ambient runtime.
        let mut woken: Vec<Arc<Self>> = Vec::new();
        self.status
            .advance_and_notify(u16::from(encoded), |waiter| match waiter {
                // A dropped receiver just means the waiter went away.
                StatusWaiter::Oneshot(sender) => {
                    let _ = sender.send(());
                }
                StatusWaiter::Dependency(dependency) => {
                    if let Some(parent) = dependency.fire() {
                        woken.push(parent);
                    }
                }
            });

        if woken.is_empty() {
            return;
        }
        GENERATION_DRIVE_COUNTERS
            .drive_wakes
            .fetch_add(woken.len() as u64, Ordering::Relaxed);
        Self::publish_woken_dependents(&woken);
    }

    /// Hands a batch of woken dependents back to their maps.
    ///
    /// Each holder publishes to *its own* sink, not to the publisher's: they
    /// coincide for every holder of one map, but a holder built without a map --
    /// tests, benches, worldgen fixtures -- publishes nowhere, and using the
    /// publisher's sink would queue it into a map that does not own it.
    ///
    /// Consecutive holders sharing a sink go in under one lock hold, which in
    /// production is the whole batch: the wake path runs on the rayon worker
    /// that just published, with every chunk waiting on that status blocked
    /// behind it.
    fn publish_woken_dependents(woken: &[Arc<Self>]) {
        let mut index = 0;
        while index < woken.len() {
            let Some(inbox) = woken[index].generation_inbox.upgrade() else {
                index += 1;
                continue;
            };
            let mut end = index + 1;
            while end < woken.len()
                && ptr::eq(woken[end].generation_inbox.as_ptr(), Arc::as_ptr(&inbox))
            {
                end += 1;
            }
            inbox.push_all(&woken[index..end]);
            index = end;
        }
    }

    /// Releases one dependency registration taken against `epoch` of this
    /// holder's park.
    pub(crate) fn finish_dependency(&self, epoch: u64) -> DecOutcome {
        let outcome = self.drive.finish_dependency(epoch);
        if outcome == DecOutcome::Requeue {
            // The only release per park that is told the park is over, so the
            // only one that can take the holder back out of the parked gauge.
            self.set_drive_gauge(DRIVE_GAUGE_NONE);
        }
        outcome
    }

    /// Hands this holder back to the map for admission, after the park it was
    /// waiting in ended.
    ///
    /// Exactly one release per park observes [`DecOutcome::Requeue`], so this
    /// cannot queue the same park twice.
    fn requeue_for_generation(self: &Arc<Self>) {
        GENERATION_DRIVE_COUNTERS
            .drive_wakes
            .fetch_add(1, Ordering::Relaxed);
        self.queue_for_generation();
    }

    /// Publishes this holder to the map's admission inbox.
    ///
    /// A missing inbox means the map is gone and nothing is going to admit
    /// anything again.
    pub(crate) fn queue_for_generation(self: &Arc<Self>) {
        if let Some(inbox) = self.generation_inbox.upgrade() {
            inbox.push(self);
        }
    }

    /// Moves this holder between the drive gauges.
    ///
    /// One swap rather than a load and a store: `abandon` runs from the unload
    /// path while the drive's own transitions run on the drive task, and two
    /// read-modify-writes could otherwise interleave into a gauge that never
    /// comes back down.
    fn set_drive_gauge(&self, next: u8) {
        let previous = self.drive_gauge.swap(next, Ordering::AcqRel);
        if previous == next {
            return;
        }
        let counters = &GENERATION_DRIVE_COUNTERS;
        match previous {
            DRIVE_GAUGE_PARKED => {
                counters.parked_holders.fetch_sub(1, Ordering::Relaxed);
            }
            DRIVE_GAUGE_STALLED => {
                counters.stalled_holders.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
        match next {
            DRIVE_GAUGE_PARKED => {
                counters.parked_holders.fetch_add(1, Ordering::Relaxed);
            }
            DRIVE_GAUGE_STALLED => {
                counters.stalled_holders.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Marks this holder as needing a generation run.
    ///
    /// `true` means the caller now owns a queue entry and must publish it with
    /// [`Self::queue_for_generation`]; see [`GenerationDrive::arm`] for why an
    /// already-running holder answers `false` without queueing anything.
    pub(crate) fn arm(&self) -> bool {
        let queued = self.drive.arm();
        if queued {
            // Only a transition out of `Idle`, `Parked` or `Stalled` returns
            // `true`, so this is where a park or a stall ends by re-arming.
            self.set_drive_gauge(DRIVE_GAUGE_NONE);
        }
        queued
    }

    /// Claims this holder's queue entry, returning the run ticket.
    pub(crate) fn begin_generation_run(&self) -> Option<u64> {
        let ticket = self.drive.begin_run();
        if ticket.is_some() {
            self.set_drive_gauge(DRIVE_GAUGE_NONE);
        }
        ticket
    }

    /// Opens a dependency-registration pass, returning the park epoch.
    pub(crate) fn park_begin(&self, ticket: u64) -> Option<u64> {
        let epoch = self.drive.park_begin(ticket);
        if epoch.is_some() {
            self.set_drive_gauge(DRIVE_GAUGE_PARKED);
        }
        epoch
    }

    /// Registers this parked holder as waiting for `dependency` to publish
    /// `required`.
    ///
    /// The slot is armed *before* the waiter is registered, never after:
    /// `AtomicWaitQueue::wait` links the node before it returns, so the
    /// dependency can publish and fire the waiter while this call is still
    /// running. Arming afterwards would let that release run against a count
    /// that has not been raised yet, and the raise would then strand a phantom
    /// registration nothing will ever release.
    ///
    /// The registration is filed one below the encoded status because the queue
    /// releases a waiter when the status *exceeds* what it waited for.
    ///
    /// `false` means the park is already over and the caller must stop
    /// registering.
    pub(crate) fn park_on(
        self: &Arc<Self>,
        dependency: &Arc<Self>,
        required: ChunkStatus,
        epoch: u64,
    ) -> bool {
        if !self.drive.arm_dependency(epoch) {
            return false;
        }
        let wait_for = u16::from(encoded_published_status(required)) - 1;
        let waiter = StatusWaiter::Dependency(DependencyWaiter::new(self, required, epoch));
        match dependency.status.wait(wait_for, waiter) {
            WaitOutcome::Registered => {}
            // Satisfied while this registration was being filed, or the
            // dependency's queue is gone. The slot has to go back, and it has to
            // go back exactly once: releasing it here and then letting the
            // returned waiter drop releases it *twice*, which takes the park
            // below the count it armed and hands the requeue to a pass that is
            // still registering. `fire` is the one release that consumes the
            // waiter, so the `Drop` that follows finds nothing left to give
            // back.
            WaitOutcome::AlreadySatisfied(returned) | WaitOutcome::Cancelled(returned) => {
                let StatusWaiter::Dependency(waiter) = returned else {
                    unreachable!("the queue hands back exactly the payload it was given");
                };
                if let Some(parent) = waiter.fire() {
                    // Only reachable if something ended the park underneath this
                    // pass; the bias otherwise keeps the count above zero until
                    // the pass releases it.
                    parent.queue_for_generation();
                }
            }
        }
        true
    }

    /// Ends a run with nothing left to do.
    pub(crate) fn to_idle(&self, ticket: u64) -> bool {
        self.drive.to_idle(ticket)
    }

    /// Ends a run that cannot progress until something re-arms it.
    pub(crate) fn to_stalled(&self, ticket: u64) -> bool {
        let stalled = self.drive.to_stalled(ticket);
        if stalled {
            self.set_drive_gauge(DRIVE_GAUGE_STALLED);
        }
        stalled
    }

    /// The drive's current epoch, i.e. the ticket a run must act on.
    pub(crate) fn generation_run_ticket(&self) -> u64 {
        self.drive.epoch()
    }

    /// Whether this holder still holds the queue entry it was pushed with.
    ///
    /// The selection queue's entries are only ever dropped by the pass that
    /// reads this, so an entry whose holder has been withdrawn since (a level
    /// drop, an unload) has to be recognisable here.
    pub(crate) fn is_queued_for_generation(&self) -> bool {
        self.drive.phase() == DrivePhase::Queued
    }

    /// Withdraws this holder from generation, e.g. because its ticket is gone.
    pub(crate) fn abandon_generation_drive(&self) {
        if self.drive.abandon() {
            self.set_drive_gauge(DRIVE_GAUGE_NONE);
        }
    }

    /// Records a stall and returns how many consecutive stalls this holder has
    /// now had, saturating so the backoff cannot wrap back to zero.
    pub(crate) fn record_stall(&self) -> u32 {
        let previous = self
            .stall_attempts
            .try_update(Ordering::AcqRel, Ordering::Acquire, |attempts| {
                Some(attempts.saturating_add(1))
            })
            .unwrap_or(0);
        previous.saturating_add(1)
    }

    /// Clears the stall backoff after a run that dispatched work.
    pub(crate) fn clear_stall_backoff(&self) {
        self.stall_attempts.store(0, Ordering::Release);
    }

    /// Whether this holder is still in the active half of its save lifecycle.
    ///
    /// A holder that has begun unloading must not start new generation work:
    /// its data is about to be snapshotted for saving.
    pub(crate) fn is_save_lifecycle_active(&self) -> bool {
        self.save_lifecycle.load(Ordering::Acquire) == SAVE_LIFECYCLE_ACTIVE
    }

    /// Registers tick queues before Full status becomes observable to watchers.
    fn register_full_chunk_ticks(&self) {
        let Some(chunk) = self.data.get() else {
            panic!("Full status must have installed chunk data");
        };
        let Some(_) = chunk.full_runtime() else {
            panic!("Full status must expose a Full chunk view");
        };
        let full = FullChunkRef::from_full_context(chunk);
        let Some(world) = full.get_level() else {
            // Focused holder tests construct chunks without a live world. Real
            // loaded/generated chunks always carry the WorldGenContext world.
            return;
        };
        if let Err(error) = world.register_full_chunk_ticks(full) {
            panic!("Full chunk scheduled-tick registration invariant failed: {error:?}");
        }
    }

    fn publish_full(self: &Arc<Self>) {
        let Some(full) = self.try_full_chunk() else {
            return;
        };
        let world = full.get_level();
        if let Some(world) = world {
            world.update_entity_chunk_visibility(self.pos, self.entity_visibility());
        }
        self.full_status_initialized.store(true, Ordering::Release);
        if let Some(publications) = self.full_publications.upgrade() {
            publications.publish(self);
        }
    }

    /// Inserts a chunk into the holder without notifying watchers.
    /// The caller is responsible for notifying via the completion channel.
    pub(crate) fn insert_chunk_no_notify(&self, chunk: Chunk) {
        assert!(
            self.data.set(chunk).is_ok(),
            "initial chunk installation cannot replace existing data"
        );
    }

    /// Wakes all `await_chunk` watchers without changing the chunk result.
    /// This allows waiting futures to re-check `is_status_disallowed` and bail
    /// out during chunk unload.
    pub fn wake_all_watchers(&self) {
        self.status_changed.notify_waiters();
    }

    /// Cancels the current generation task.
    pub fn cancel_generation_task(&self) {
        let mut task_guard = self.generation_task.lock();
        self.generation_task_target
            .store(STATUS_NONE, Ordering::Release);
        if let Some(task) = task_guard.take() {
            task.cancel();
        }
    }

    /// Clears the current generation task if it is still the supplied task.
    pub(crate) fn clear_generation_task_if_current(&self, task: &Arc<ChunkGenerationTask>) {
        let mut task_guard = self.generation_task.lock();
        if task_guard
            .as_ref()
            .is_some_and(|current_task| Arc::ptr_eq(current_task, task))
        {
            task_guard.take();
            self.generation_task_target
                .store(STATUS_NONE, Ordering::Release);
        }
    }
}

fn rayon_spawn<F, R>(thread_pool: &rayon::ThreadPool, func: F) -> impl Future<Output = R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static + Debug,
{
    let (sender, receiver) = oneshot::channel();
    thread_pool.spawn(move || {
        // Ignore a dropped receiver rather than panicking on it. The awaiting
        // task can go away -- the task tracker aborting at shutdown is enough --
        // and with `panic = "abort"` in release a panic here takes the server
        // down rather than losing one result. The work has already run by this
        // point; only its delivery is lost.
        let _ = sender.send(func());
    });
    async move { receiver.await.expect("Failed to receive rayon task result") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{task::Poll, time::Duration as TestDuration};
    use tokio::time::sleep as test_sleep;

    use crate::behavior::init_behaviors;
    use crate::chunk::Chunk;
    use crate::chunk::section::{ChunkSection, Sections};
    use crate::test_support::fresh_test_world;
    use crate::world::tick_scheduler::TickPriority;
    use steel_registry::{test_support::init_test_registry, vanilla_blocks, vanilla_fluids};

    fn init_chunk_test_registry() {
        init_test_registry();
        init_behaviors();
    }

    fn test_holder() -> Arc<ChunkHolder> {
        Arc::new(ChunkHolder::new(
            ChunkPos::new(0, 0),
            ChunkTicketLevel::FULL_CHUNK,
            Some(ChunkTicketLevel::FULL_CHUNK),
            0,
            16,
        ))
    }

    fn test_proto_chunk(_status: ChunkStatus) -> Chunk {
        Chunk::new(
            Sections::from_owned(vec![ChunkSection::new_empty()].into_boxed_slice()),
            ChunkPos::new(0, 0),
            0,
            16,
            Weak::new(),
        )
    }

    #[test]
    fn insert_chunk_publishes_the_authoritative_status() {
        init_chunk_test_registry();
        let holder = test_holder();
        let proto = Chunk::new(
            Sections::from_owned(vec![ChunkSection::new_empty()].into_boxed_slice()),
            ChunkPos::new(0, 0),
            0,
            16,
            Weak::new(),
        );

        holder.insert_chunk(proto, ChunkStatus::Light);

        let Some(chunk) = holder.try_chunk(ChunkStatus::Light) else {
            panic!("inserted chunk should be available at published status");
        };
        assert_eq!(holder.published_status(), Some(ChunkStatus::Light));
        assert!(chunk.full_runtime().is_none());
    }

    #[test]
    #[should_panic(expected = "initial chunk status must match its Full runtime state")]
    fn insert_chunk_rejects_full_status_for_proto_data() {
        init_chunk_test_registry();
        test_holder().insert_chunk(test_proto_chunk(ChunkStatus::Spawn), ChunkStatus::Full);
    }

    #[test]
    fn full_readiness_publication_waits_for_post_load_initialization() {
        init_chunk_test_registry();
        let publications = Arc::new(FullPublicationQueue::default());
        let holder = Arc::new(ChunkHolder::new_with_map_sinks(
            ChunkPos::new(0, 0),
            ChunkTicketLevel::FULL_CHUNK,
            None,
            0,
            16,
            Arc::downgrade(&publications),
            Weak::new(),
        ));
        let full = test_proto_chunk(ChunkStatus::Light);
        let _ = full.promote_to_full();

        holder.store_and_publish_chunk_status(full, ChunkStatus::Full);

        assert_eq!(holder.published_status(), Some(ChunkStatus::Full));
        assert!(!holder.is_full_status_initialized());
        assert!(publications.drain().is_empty());

        holder.publish_full();

        assert!(holder.is_full_status_initialized());
        assert_eq!(publications.drain().len(), 1);
    }

    #[test]
    fn generated_full_status_is_accessible_when_readiness_is_published() {
        init_chunk_test_registry();
        let holder = test_holder();
        holder.insert_chunk(test_proto_chunk(ChunkStatus::Light), ChunkStatus::Light);
        holder.upgrade_to_full();

        assert_eq!(holder.entity_visibility(), EntityVisibility::Hidden);
        assert!(!holder.is_full_status_initialized());

        holder.finish_generation_status(ChunkStatus::Full);

        assert_eq!(holder.entity_visibility(), EntityVisibility::Tracked);
        assert!(holder.is_full_status_initialized());
    }

    #[test]
    fn late_lower_generation_completion_does_not_regress_published_status() {
        init_chunk_test_registry();
        let holder = test_holder();
        holder.insert_chunk(test_proto_chunk(ChunkStatus::Light), ChunkStatus::Light);

        holder.finish_generation_status(ChunkStatus::Spawn);
        holder.finish_generation_status(ChunkStatus::Features);

        assert_eq!(holder.published_status(), Some(ChunkStatus::Spawn));
        assert!(holder.try_chunk(ChunkStatus::Spawn).is_some());
    }

    #[tokio::test]
    async fn status_waiter_observes_publication_after_subscribing() {
        init_chunk_test_registry();
        let holder = test_holder();
        let waiter = holder.await_chunk_status(ChunkStatus::Empty);

        holder.insert_chunk(test_proto_chunk(ChunkStatus::Empty), ChunkStatus::Empty);

        assert_eq!(waiter.await, Some(ChunkStatus::Empty));
    }

    #[tokio::test]
    async fn pending_status_waiters_wake_after_publication() {
        init_chunk_test_registry();
        let holder = test_holder();
        let first_waiter = holder.await_chunk_status(ChunkStatus::Empty);
        let second_waiter = holder.await_chunk_status(ChunkStatus::Empty);
        tokio::pin!(first_waiter, second_waiter);
        assert!(matches!(futures::poll!(&mut first_waiter), Poll::Pending));
        assert!(matches!(futures::poll!(&mut second_waiter), Poll::Pending));

        let publishing_holder = Arc::clone(&holder);
        let publish_task = tokio::spawn(async move {
            publishing_holder
                .insert_chunk(test_proto_chunk(ChunkStatus::Empty), ChunkStatus::Empty);
        });

        let (first_status, second_status) = tokio::select! {
            biased;
            () = test_sleep(TestDuration::from_secs(1)) => {
                panic!("pending status waiters were not woken by publication");
            }
            statuses = async { tokio::join!(&mut first_waiter, &mut second_waiter) } => statuses,
        };
        assert_eq!(first_status, Some(ChunkStatus::Empty));
        assert_eq!(second_status, Some(ChunkStatus::Empty));
        assert!(publish_task.await.is_ok());
    }

    #[test]
    fn full_registration_transfers_prepublication_block_and_fluid_ticks() {
        init_chunk_test_registry();
        let world = fresh_test_world("prepublication_tick_transfer");
        let chunk_pos = ChunkPos::new(0, 0);
        let min_y = world.get_min_y();
        let height = world.get_height();
        let sections = (0..height / 16)
            .map(|_| ChunkSection::new_empty())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let proto = Chunk::new(
            Sections::from_owned(sections),
            chunk_pos,
            min_y,
            height,
            Arc::downgrade(&world),
        );
        let block_pos = BlockPos::new(1, min_y + 1, 1);
        let fluid_pos = BlockPos::new(2, min_y + 1, 2);
        proto.schedule_block_tick(block_pos, &vanilla_blocks::STONE, TickPriority::High);
        proto.schedule_fluid_tick(fluid_pos, &vanilla_fluids::WATER, TickPriority::Low);

        let holder = Arc::new(ChunkHolder::new(
            chunk_pos,
            ChunkTicketLevel::FULL_CHUNK,
            Some(ChunkTicketLevel::FULL_CHUNK),
            min_y,
            height,
        ));
        let _ = world
            .chunk_map
            .chunks
            .insert_sync(chunk_pos, Arc::clone(&holder));
        holder.insert_chunk(proto, ChunkStatus::Light);
        holder.upgrade_to_full();

        assert!(!world.has_registered_full_chunk_ticks(chunk_pos));
        holder.finish_generation_status(ChunkStatus::Full);

        assert!(world.has_registered_full_chunk_ticks(chunk_pos));
        assert!(world.has_scheduled_block_tick(block_pos, &vanilla_blocks::STONE));
        assert!(world.has_scheduled_fluid_tick(fluid_pos, &vanilla_fluids::WATER));
    }

    #[test]
    fn client_deltas_require_confirmed_block_readiness() {
        init_chunk_test_registry();
        let holder = test_holder();
        let full = test_proto_chunk(ChunkStatus::Light);
        let _ = full.promote_to_full();
        holder.insert_chunk(full, ChunkStatus::Full);
        let pos = BlockPos::new(1, 1, 1);
        let section_pos = SectionPos::new(0, 0, 0);
        let revision = holder.packet_content_revision();
        let chunk = holder
            .try_chunk(ChunkStatus::Full)
            .expect("the test holder should contain a Full chunk");
        chunk.clear_dirty();

        assert!(!holder.block_changed(pos));
        assert!(!holder.light_changed(LightLayer::Block, section_pos));
        assert_eq!(holder.packet_content_revision(), revision);
        assert!(
            holder
                .try_chunk(ChunkStatus::Full)
                .is_some_and(Chunk::is_dirty),
            "pre-readiness light changes must still be persisted"
        );

        holder.transition_ticking_readiness(TickingReadiness::BlockTicking);

        assert!(holder.light_changed(LightLayer::Block, section_pos));
        assert_eq!(holder.packet_content_revision(), revision + 1);
        holder.clear_broadcast_queued();
        assert!(holder.block_changed(pos));
        assert_eq!(holder.packet_content_revision(), revision + 2);
    }

    #[test]
    fn unpublished_status_claim_rolls_back_to_unloaded() {
        let holder = test_holder();
        let claim = holder
            .claim_status_work(ChunkStatus::Empty)
            .expect("empty status should be claimable");

        assert!(holder.claim_status_work(ChunkStatus::Empty).is_none());

        drop(claim);

        assert!(!holder.status_work_covers(ChunkStatus::Empty));
        let retry = holder
            .claim_status_work(ChunkStatus::Empty)
            .expect("abandoned empty status should be claimable again");
        drop(retry);
    }

    #[test]
    fn unpublished_child_claim_rolls_back_to_published_parent() {
        init_chunk_test_registry();
        let holder = test_holder();
        holder.insert_chunk(test_proto_chunk(ChunkStatus::Empty), ChunkStatus::Empty);

        let claim = holder
            .claim_status_work(ChunkStatus::StructureStarts)
            .expect("child status should be claimable after parent is published");

        drop(claim);

        assert!(holder.status_work_covers(ChunkStatus::Empty));
        assert!(!holder.status_work_covers(ChunkStatus::StructureStarts));
        let retry = holder
            .claim_status_work(ChunkStatus::StructureStarts)
            .expect("abandoned child status should be claimable again");
        drop(retry);
    }

    #[test]
    fn empty_claim_can_publish_a_higher_loaded_status() {
        init_chunk_test_registry();
        let holder = test_holder();
        let empty_claim = holder
            .claim_status_work(ChunkStatus::Empty)
            .expect("empty status should be claimable");

        holder.insert_chunk(
            test_proto_chunk(ChunkStatus::StructureStarts),
            ChunkStatus::StructureStarts,
        );
        drop(empty_claim);

        assert!(holder.status_work_covers(ChunkStatus::StructureStarts));
        assert!(!holder.status_work_covers(ChunkStatus::StructureReferences));
        let next_claim = holder
            .claim_status_work(ChunkStatus::StructureReferences)
            .expect("next status should be claimable from loaded status");
        drop(next_claim);
    }

    /// A failed run rolls its claim back from inside the tokio task that held
    /// it, and tokio drops that task's locals before it resolves the
    /// `JoinHandle` -- so the rollback is always observed while the drive that
    /// dispatched the run is still `Running` and still holding its ticket. If
    /// the rollback re-arms, `GenerationDrive::arm` queues nothing and bumps the
    /// epoch, the drive's `to_stalled` is refused, and the `JobFailed` stall and
    /// its backoff are skipped: the drive loops back and re-dispatches the step
    /// that just failed, forever, holding an admission permit.
    ///
    /// Only the `STEEL_STAGE1=1` run of this suite exercises the dispatcher this
    /// protects, but the rollback path itself is shared, so the assertion holds
    /// in both.
    #[test]
    fn a_rolled_back_claim_leaves_the_running_drive_able_to_stall() {
        let holder = test_holder();
        assert!(holder.arm());
        let ticket = holder
            .begin_generation_run()
            .expect("an armed drive can run");
        let claim = holder
            .claim_status_work(ChunkStatus::Empty)
            .expect("empty status should be claimable");

        // The failing job's claim going away without publishing anything.
        drop(claim);

        assert_eq!(
            holder.generation_run_ticket(),
            ticket,
            "a claim rollback must not spend the run ticket of the drive that dispatched it",
        );
        assert!(
            holder.to_stalled(ticket),
            "the drive must still be able to take its JobFailed stall, or it hot-retries the \
             failed step while holding an admission permit",
        );
    }

    #[tokio::test]
    async fn claimed_status_waiter_finishes_when_claim_is_abandoned() {
        let holder = test_holder();
        let claim = holder
            .claim_status_work(ChunkStatus::Empty)
            .expect("empty status should be claimable");
        let waiter = holder.await_claimed_chunk_status(ChunkStatus::Empty);

        drop(claim);

        assert!(waiter.await.is_none());
    }

    #[test]
    fn save_dependency_controls_ready_for_saving() {
        let holder = test_holder();
        assert!(holder.is_ready_for_saving());

        let first = holder.add_save_dependency();
        let second = holder.add_save_dependency();
        assert!(!holder.is_ready_for_saving());

        drop(first);
        assert!(!holder.is_ready_for_saving());

        drop(second);
        assert!(holder.is_ready_for_saving());
    }

    #[test]
    fn save_preparation_defers_revival_only_until_the_snapshot_is_built() {
        let holder = test_holder();
        holder.begin_unloading();
        let preparation = holder
            .try_begin_save_preparation()
            .expect("an unloading holder should begin save preparation");

        assert!(!holder.try_revive_from_unloading());

        drop(preparation);

        assert!(holder.try_revive_from_unloading());
        assert!(holder.try_begin_save_preparation().is_none());
    }

    /// Serialises the tests that read [`GENERATION_DRIVE_COUNTERS`]. The
    /// counters are process-wide, so two of these running at once would each
    /// see the other's registrations in their deltas.
    static COUNTER_LOCK: SyncMutex<()> = SyncMutex::new(());

    fn dropped_unfired() -> u64 {
        GENERATION_DRIVE_COUNTERS
            .dependency_waiters_dropped_unfired
            .load(Ordering::Relaxed)
    }

    fn live_registrations() -> i64 {
        GENERATION_DRIVE_COUNTERS
            .live_dependency_registrations
            .load(Ordering::Relaxed)
    }

    /// Parks `holder` holding the park bias plus `registrations` slots, and
    /// returns the park epoch every registration must carry.
    fn park(holder: &Arc<ChunkHolder>, registrations: u32) -> u64 {
        assert!(holder.drive.arm());
        let ticket = holder.drive.begin_run().expect("an armed drive can run");
        let epoch = holder
            .drive
            .park_begin(ticket)
            .expect("a running drive can park");
        for _ in 0..registrations {
            assert!(holder.drive.arm_dependency(epoch));
        }
        epoch
    }

    #[test]
    fn allowed_status_follows_the_ticket_level() {
        init_chunk_test_registry();
        let holder = test_holder();

        assert_eq!(holder.highest_allowed_status(), Some(ChunkStatus::Full));
        assert!(holder.needs_generation());
        assert!(!holder.is_status_disallowed(ChunkStatus::Full));

        let full = test_proto_chunk(ChunkStatus::Light);
        let _ = full.promote_to_full();
        holder.insert_chunk(full, ChunkStatus::Full);

        assert!(
            !holder.needs_generation(),
            "a chunk published at its allowance has nothing left to generate"
        );

        holder.update_highest_allowed_status(None);

        assert_eq!(holder.highest_allowed_status(), None);
        assert!(holder.is_status_disallowed(ChunkStatus::Empty));
        assert!(
            !holder.needs_generation(),
            "a chunk that may not generate at all never needs generation"
        );
    }

    #[test]
    fn an_unpublished_chunk_below_its_allowance_needs_generation() {
        init_chunk_test_registry();
        let holder = test_holder();
        holder.insert_chunk(test_proto_chunk(ChunkStatus::Light), ChunkStatus::Light);

        assert!(holder.needs_generation());

        holder.update_highest_allowed_status(Some(ChunkTicketLevel::FULL_CHUNK));
        assert!(holder.needs_generation());
    }

    /// The fire/drop pair. `fire` consumes the waiter, so its `Drop` still runs;
    /// if that `Drop` released the registration again the park would lose a slot
    /// it never armed, and at zero the count wraps to ~65k and strands the
    /// holder.
    #[test]
    fn firing_a_dependency_waiter_releases_exactly_one_registration() {
        let _lock = COUNTER_LOCK.lock();
        let holder = test_holder();
        let epoch = park(&holder, 2);
        let live_before = live_registrations();
        let dropped_before = dropped_unfired();

        let waiter = DependencyWaiter::new(&holder, ChunkStatus::Light, epoch);
        assert_eq!(live_registrations(), live_before + 1);

        assert!(
            waiter.fire().is_none(),
            "a park with registrations left does not requeue yet"
        );

        assert_eq!(holder.drive.outstanding(), 2);
        assert_eq!(
            live_registrations(),
            live_before,
            "the gauge must fall exactly once per registration"
        );
        assert_eq!(
            dropped_unfired(),
            dropped_before,
            "a fired waiter is not a waiter dropped unfired"
        );
    }

    #[test]
    fn dropping_a_dependency_waiter_unfired_still_releases_its_registration() {
        let _lock = COUNTER_LOCK.lock();
        let holder = test_holder();
        let epoch = park(&holder, 2);
        let live_before = live_registrations();
        let dropped_before = dropped_unfired();

        drop(DependencyWaiter::new(&holder, ChunkStatus::Light, epoch));

        assert_eq!(holder.drive.outstanding(), 2);
        assert_eq!(live_registrations(), live_before);
        assert_eq!(
            dropped_unfired(),
            dropped_before + 1,
            "the drop path must be distinguishable from the fire path"
        );
    }

    #[test]
    fn a_dependency_waiter_outliving_its_parent_releases_nothing() {
        let _lock = COUNTER_LOCK.lock();
        let holder = test_holder();
        let epoch = park(&holder, 1);
        let live_before = live_registrations();
        let dropped_before = dropped_unfired();
        let waiter = DependencyWaiter::new(&holder, ChunkStatus::Light, epoch);

        drop(holder);
        drop(waiter);

        assert_eq!(live_registrations(), live_before);
        assert_eq!(
            dropped_unfired(),
            dropped_before,
            "an unloaded parent has no park left to release"
        );
    }

    /// The publish path with a dependency payload on the queue: the neighbour
    /// publishes on its generation worker, the last registration of the park
    /// resolves, and the parent lands in the map's inbox exactly once.
    #[test]
    fn publishing_a_status_requeues_the_dependent_it_was_the_last_dependency_of() {
        // Takes the counter lock even though it asserts on no gauge: it builds a
        // `DependencyWaiter`, which raises the process-wide live-registration
        // count, and that count straddles the sampling windows of the delta
        // tests above. Without this the three of them fail intermittently.
        let _lock = COUNTER_LOCK.lock();
        init_chunk_test_registry();
        let inbox = Arc::new(GenerationInbox::default());
        let parent = Arc::new(ChunkHolder::new_with_map_sinks(
            ChunkPos::new(1, 0),
            ChunkTicketLevel::FULL_CHUNK,
            None,
            0,
            16,
            Weak::new(),
            Arc::downgrade(&inbox),
        ));
        let epoch = park(&parent, 1);

        let neighbour = test_holder();
        let wait_for = u16::from(encoded_published_status(ChunkStatus::Light)) - 1;
        assert!(matches!(
            neighbour.status.wait(
                wait_for,
                StatusWaiter::Dependency(DependencyWaiter::new(&parent, ChunkStatus::Light, epoch)),
            ),
            WaitOutcome::Registered
        ));

        // The registration pass is complete; releasing the bias leaves this one
        // registration as all the park is waiting for, so the publication below
        // owns the requeue.
        assert_eq!(parent.drive.finish_dependency(epoch), DecOutcome::Pending);
        assert!(inbox.take_all().is_empty());
        neighbour.insert_chunk(test_proto_chunk(ChunkStatus::Light), ChunkStatus::Light);

        let requeued = inbox.take_all();
        assert_eq!(requeued.len(), 1);
        assert!(Arc::ptr_eq(&requeued[0], &parent));
        assert_eq!(parent.drive.outstanding(), 0);
    }

    /// A queued holder that loses its ticket before the drain must still be
    /// freeable: `ChunkMap::process_unloads` releases an unloading holder only
    /// at `strong_count == 1`, and nothing purges this queue on unload, so an
    /// entry that counted would pin the holder in `unloading_chunks` forever.
    #[test]
    fn a_queued_holder_is_not_kept_alive_by_the_inbox() {
        // Holds the counter lock despite asserting on no gauge: it builds a
        // `DependencyWaiter`, which moves the process-wide live-registration
        // count, and that count straddles the sampling windows of the delta
        // tests above. Without this those three fail intermittently.
        let _lock = COUNTER_LOCK.lock();
        init_chunk_test_registry();
        let inbox = Arc::new(GenerationInbox::default());
        let parent = Arc::new(ChunkHolder::new_with_map_sinks(
            ChunkPos::new(2, 0),
            ChunkTicketLevel::FULL_CHUNK,
            None,
            0,
            16,
            Weak::new(),
            Arc::downgrade(&inbox),
        ));
        let epoch = park(&parent, 1);

        let neighbour = test_holder();
        let wait_for = u16::from(encoded_published_status(ChunkStatus::Light)) - 1;
        assert!(matches!(
            neighbour.status.wait(
                wait_for,
                StatusWaiter::Dependency(DependencyWaiter::new(&parent, ChunkStatus::Light, epoch)),
            ),
            WaitOutcome::Registered
        ));
        assert_eq!(parent.drive.finish_dependency(epoch), DecOutcome::Pending);
        neighbour.insert_chunk(test_proto_chunk(ChunkStatus::Light), ChunkStatus::Light);

        // The publication queued the parent. This handle stands in for the
        // map's own entry, and it has to be the last one.
        assert_eq!(Arc::strong_count(&parent), 1);

        let unloaded = Arc::downgrade(&parent);
        drop(parent);
        assert!(unloaded.upgrade().is_none());
        assert!(inbox.take_all().is_empty());
    }

    #[test]
    fn revival_winning_the_lifecycle_race_cancels_save_preparation() {
        let holder = test_holder();
        holder.begin_unloading();

        assert!(holder.try_revive_from_unloading());
        assert!(holder.try_begin_save_preparation().is_none());
    }
}
