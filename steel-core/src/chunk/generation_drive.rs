//! Per-holder generation state machine.
//!
//! One `AtomicU64` per [`ChunkHolder`](super::chunk_holder::ChunkHolder) decides
//! who is allowed to drive that chunk's generation: whether it is idle, waiting
//! for admission, running, parked on dependencies, or stalled. Every transition
//! is a single compare-exchange on that word, so no pair of transitions can
//! interleave and produce two drivers, a lost re-arm, or a dependency count that
//! outlives the park it belongs to.
//!
//! The module is deliberately standalone: it has no reference to the holder, the
//! chunk map, or the scheduler, so the table below can be exhaustively tested
//! without building a world.
//!
//! # Memory ordering
//!
//! Every operation uses `SeqCst`, not `Acquire`/`Release`. The surrounding
//! protocol is a Dekker pattern: the ticket side stores the holder's newly
//! allowed status and then reads this word, while the drive side writes this
//! word and then reads the allowed status. Release/Acquire orders Store-Store,
//! Load-Load and Load-Store but *not* Store-Load, so with `AcqRel` both sides
//! may legally read the other's pre-store value and both conclude that the other
//! will do the work. Only a single total order over the two accesses -- i.e.
//! `SeqCst` on both sides -- rules that out.
//!
//! # Run tickets
//!
//! [`GenerationDrive::begin_run`] hands out the epoch its run is allowed to act
//! on, and every operation that *leaves* `Running` -- [`GenerationDrive::park_begin`],
//! [`GenerationDrive::to_idle`], [`GenerationDrive::to_stalled`] -- takes that
//! ticket back and refuses when the word has moved on.
//!
//! The precondition is not decoration. [`GenerationDrive::arm`] on a running
//! drive deliberately returns `false` and queues nothing, on the promise that
//! the epoch bump makes the run re-evaluate. If a run could leave `Running` from
//! *any* epoch, that promise is unenforceable: the driver decides it is done,
//! `arm` bumps, the driver stores `Idle`, and the newly allowed status is never
//! generated because the one caller that would have queued the holder has
//! already been told `false`. Measured on the phase-only version: 194 silently
//! lost re-arms in 200k two-thread races. Reading `epoch()` and comparing before
//! the call does not fix it -- the bump fits between the load and the
//! compare-exchange.
//!
//! A driver therefore *loops*: snapshot the ticket, then read the holder's
//! allowed status, do the work, then try to leave. That order matters, and it is
//! the same Dekker pairing as above -- an armer stores the status before it
//! bumps, so a driver that read a stale status is guaranteed to lose its exit
//! and come round again. `None`/`false` from an exit is never an error, it means
//! the situation changed underneath and the evaluation must be redone against
//! the ticket the word now carries.

// Nothing holds a `GenerationDrive` yet; the scheduler that owns one per holder
// lands separately. The tests below construct and drive it, so the expectation
// is scoped to non-test builds.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "wired into the chunk scheduler in a follow-up change"
    )
)]

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

/// Failed compare-exchanges, counted so the racing tests can prove they raced.
///
/// A two-thread test that synchronises through a `Barrier` looks like a race and
/// is not one: the barrier release goes through a futex, the threads come out
/// tens of microseconds apart, and the loser always reads the already-final word
/// on its *initial* load. The retry loop -- the only thing that makes a
/// transition atomic under contention -- is then never entered, and the test
/// covers nothing a single-threaded test does not. The tests assert this counter
/// moved, so they fail if they ever degrade into that.
#[cfg(test)]
static CAS_RETRIES: AtomicUsize = AtomicUsize::new(0);

/// Bit 0..3 of the word.
const PHASE_BITS: u32 = 3;
const PHASE_MASK: u64 = (1 << PHASE_BITS) - 1;

/// Bit 3..19 of the word.
const OUTSTANDING_SHIFT: u32 = PHASE_BITS;
const OUTSTANDING_BITS: u32 = 16;
const OUTSTANDING_MASK: u64 = (1 << OUTSTANDING_BITS) - 1;
const MAX_OUTSTANDING: u32 = OUTSTANDING_MASK as u32;

/// Bit 19..64 of the word.
const EPOCH_SHIFT: u32 = OUTSTANDING_SHIFT + OUTSTANDING_BITS;
const EPOCH_BITS: u32 = u64::BITS - EPOCH_SHIFT;
const EPOCH_MASK: u64 = (1 << EPOCH_BITS) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrivePhase {
    Idle = 0,
    Queued = 1,
    Running = 2,
    Parked = 3,
    Stalled = 4,
}

impl DrivePhase {
    const fn bits(self) -> u64 {
        self as u64
    }

    const fn from_bits(bits: u64) -> Self {
        match bits {
            0 => Self::Idle,
            1 => Self::Queued,
            2 => Self::Running,
            3 => Self::Parked,
            4 => Self::Stalled,
            // Only this module ever writes the phase field, and it only ever
            // writes packed variants, so anything else means the word was
            // corrupted by an aliasing write. Continuing would hand out a
            // fabricated phase and with it a second driver for the chunk.
            _ => panic!("generation drive phase field holds an unpacked value"),
        }
    }
}

/// Result of releasing one dependency registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecOutcome {
    /// Registrations remain; the holder stays parked.
    Pending,
    /// The last registration went away. The observer -- and only the observer --
    /// must push the holder for admission.
    Requeue,
    /// The park this decrement belonged to is over. The count it would have
    /// decremented belongs to a different park, so nothing was touched.
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DriveState {
    pub(crate) phase: DrivePhase,
    pub(crate) outstanding: u32,
    pub(crate) epoch: u64,
}

/// The generation state machine of a single chunk holder.
///
/// See the module documentation for the ordering requirement and
/// [`GenerationDrive::arm`] onwards for the transition table.
#[derive(Debug)]
pub(crate) struct GenerationDrive {
    word: AtomicU64,
}

const fn pack(phase: DrivePhase, outstanding: u32, epoch: u64) -> u64 {
    debug_assert!(
        outstanding as u64 <= OUTSTANDING_MASK,
        "outstanding count does not fit its field"
    );
    debug_assert!(epoch <= EPOCH_MASK, "epoch does not fit its field");
    phase.bits()
        | ((outstanding as u64 & OUTSTANDING_MASK) << OUTSTANDING_SHIFT)
        | ((epoch & EPOCH_MASK) << EPOCH_SHIFT)
}

const fn phase_of(word: u64) -> DrivePhase {
    DrivePhase::from_bits(word & PHASE_MASK)
}

const fn outstanding_of(word: u64) -> u32 {
    ((word >> OUTSTANDING_SHIFT) & OUTSTANDING_MASK) as u32
}

const fn epoch_of(word: u64) -> u64 {
    (word >> EPOCH_SHIFT) & EPOCH_MASK
}

/// Advances the epoch, wrapping inside its own 45 bits.
///
/// A plain `+ 1` on the field would eventually carry out of bit 63 and be lost,
/// and a `+ 1` on the whole word would carry into `outstanding` and then into
/// `phase` -- silently turning a `Running` drive into something else. Wrapping
/// is harmless because the epoch is only ever compared for equality against a
/// value captured microseconds earlier.
const fn next_epoch(epoch: u64) -> u64 {
    (epoch + 1) & EPOCH_MASK
}

/// The decision behind [`GenerationDrive::arm_dependency`].
///
/// Split out of the method because the method's `debug_assert!` fires *before*
/// the saturating branch below can return, i.e. in every `cargo test` build, so
/// that branch has no other way of being tested. It is the branch whose answer
/// is least obvious, and getting it wrong is silent: returning `false` would
/// make the caller skip the registration and the holder would stay parked
/// forever.
fn arm_dependency_word(word: u64, epoch: u64) -> (Option<u64>, bool) {
    if phase_of(word) != DrivePhase::Parked || epoch_of(word) != epoch {
        return (None, false);
    }

    let outstanding = outstanding_of(word);
    if outstanding >= MAX_OUTSTANDING {
        // Saturate rather than wrap. Wrapping to zero would let the next
        // decrement requeue a holder mid-pass and corrupt it permanently;
        // saturating only under-counts, which requeues the run slightly early
        // and it re-evaluates.
        return (None, true);
    }
    (Some(pack(DrivePhase::Parked, outstanding + 1, epoch)), true)
}

/// The decision behind [`GenerationDrive::finish_dependency`]. Split out for the
/// same reason as [`arm_dependency_word`]: the method's `debug_assert!` preempts
/// the `outstanding == 0` arm, so only a direct call can cover it.
fn finish_dependency_word(word: u64, epoch: u64) -> (Option<u64>, DecOutcome) {
    if phase_of(word) != DrivePhase::Parked || epoch_of(word) != epoch {
        // Either the park ended (re-armed, abandoned, already requeued) or this
        // decrement belongs to an earlier park. Touching the count now would
        // decrement a park that never armed it.
        return (None, DecOutcome::Stale);
    }

    match outstanding_of(word) {
        0 => (None, DecOutcome::Stale),
        // The epoch stays put: the registrations of this park are all resolved,
        // and `begin_run` will hand out this same epoch, so a decrement that
        // raced past the requeue reports `Stale` on the phase check rather than
        // on the epoch check.
        1 => (
            Some(pack(DrivePhase::Queued, 0, epoch)),
            DecOutcome::Requeue,
        ),
        n => (
            Some(pack(DrivePhase::Parked, n - 1, epoch)),
            DecOutcome::Pending,
        ),
    }
}

impl GenerationDrive {
    pub(crate) const fn new() -> Self {
        Self {
            word: AtomicU64::new(pack(DrivePhase::Idle, 0, 0)),
        }
    }

    /// Applies one row of the transition table.
    ///
    /// `decide` maps the observed word to `(successor, result)`, where `None`
    /// means the row is a no-op. It may be called several times, so it must stay
    /// a pure function of the word.
    ///
    /// Every operation funnels through here on purpose: hand-written retry loops
    /// are exactly where the previous attempts dropped or duplicated a
    /// transition. The no-op rows return from a value read with `SeqCst` -- both
    /// the initial load and the failed-compare-exchange load -- so they take
    /// part in the same total order as the rows that write (see the module
    /// documentation on the Dekker pattern).
    fn transition<T>(&self, decide: impl Fn(u64) -> (Option<u64>, T)) -> T {
        let mut current = self.word.load(Ordering::SeqCst);
        loop {
            let (next, result) = decide(current);
            let Some(next) = next else {
                return result;
            };
            match self
                .word
                .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return result,
                Err(observed) => {
                    #[cfg(test)]
                    CAS_RETRIES.fetch_add(1, Ordering::Relaxed);
                    current = observed;
                }
            }
        }
    }

    /// Marks the holder as needing generation work.
    ///
    /// Returns `true` only when the caller now owns a queue entry for it.
    pub(crate) fn arm(&self) -> bool {
        self.transition(|word| {
            let epoch = epoch_of(word);
            match phase_of(word) {
                // Invariant 1: arming a *running* drive must not be a no-op. The
                // epoch bump is the re-arm -- it invalidates any park handshake
                // already in flight, so the running drive re-evaluates against
                // the new ticket level instead of parking on what it read before
                // the change. Skipping the bump loses the change for good: the
                // drive parks, nothing wakes it, and the chunk never reaches the
                // newly allowed status. `false` is still the right answer, since
                // queueing an already-running holder is what gives it a second
                // driver.
                DrivePhase::Running => (
                    Some(pack(
                        DrivePhase::Running,
                        outstanding_of(word),
                        next_epoch(epoch),
                    )),
                    false,
                ),
                // Already queued: the pending entry does the job, and a second
                // entry would admit the holder twice.
                DrivePhase::Queued => (None, false),
                DrivePhase::Idle | DrivePhase::Parked | DrivePhase::Stalled => {
                    // Re-arming out of `Parked` drops the count on the floor
                    // deliberately: the epoch bump makes every registration of
                    // the abandoned park report `Stale`, so no late decrement can
                    // requeue the holder a second time.
                    (Some(pack(DrivePhase::Queued, 0, next_epoch(epoch))), true)
                }
            }
        })
    }

    /// Claims a queue entry. `Some(epoch)` is the ticket for the whole run: it
    /// must be handed to [`Self::park_begin`]'s follow-up calls to prove they
    /// belong to this run and not a later one.
    ///
    /// `None` means the entry was stale (the holder was re-armed, abandoned or
    /// is already running) and the caller must drop it without touching the
    /// chunk.
    pub(crate) fn begin_run(&self) -> Option<u64> {
        self.transition(|word| {
            if phase_of(word) == DrivePhase::Queued {
                let epoch = epoch_of(word);
                (Some(pack(DrivePhase::Running, 0, epoch)), Some(epoch))
            } else {
                (None, None)
            }
        })
    }

    /// Opens a registration pass for the dependencies this run is waiting on.
    /// `epoch` is the run ticket from [`Self::begin_run`].
    ///
    /// The returned epoch is a *new* ticket, and the one that
    /// [`Self::arm_dependency`] and [`Self::finish_dependency`] must be called
    /// with. `None` means the run ticket is spent -- something re-armed or
    /// abandoned the drive underneath -- and the caller must re-evaluate from the
    /// top instead of retrying the park, because the dependency set it was about
    /// to register for was read before the change.
    ///
    /// Checking the ticket is what makes that `None` reachable at all. Nothing
    /// but the driver itself leaves `Running`, so a phase-only precondition
    /// always succeeds: the park then lands on top of an `arm` it never saw
    /// (registering waiters for the pre-change halo), or on top of an `abandon`
    /// that returned `false` precisely because it expected the running drive to
    /// retire itself -- and a parked drive never gets another evaluation, so the
    /// holder stays parked on a chunk whose ticket is gone.
    pub(crate) fn park_begin(&self, epoch: u64) -> Option<u64> {
        self.transition(|word| {
            if phase_of(word) == DrivePhase::Running && epoch_of(word) == epoch {
                let epoch = next_epoch(epoch_of(word));
                // Invariant 3: `outstanding` starts at 1 as a bias, not as a real
                // registration, and is released only once the registration pass
                // is complete. Without it a dependency that publishes between the
                // first and last registration drives the count to zero, requeues
                // a holder that is still registering waiters, and the remaining
                // registrations then land on a park that no longer exists --
                // stranding the holder with a phantom count that nothing will
                // ever decrement.
                (Some(pack(DrivePhase::Parked, 1, epoch)), Some(epoch))
            } else {
                (None, None)
            }
        })
    }

    /// Reserves a slot for one dependency. Must be called *before* the waiter is
    /// registered with that dependency, never after: registering first leaves a
    /// window in which the dependency fires against a count that has not been
    /// raised yet.
    ///
    /// `false` means the park is over and the caller must not register.
    pub(crate) fn arm_dependency(&self, epoch: u64) -> bool {
        self.transition(|word| {
            let (next, armed) = arm_dependency_word(word, epoch);
            // Saturation means ~65k registrations against one park, which is a
            // caller bug; the saturating fallback only keeps the holder alive
            // while it is diagnosed.
            debug_assert!(
                !(armed && next.is_none()),
                "outstanding dependency count saturated; a wrap would corrupt the holder"
            );
            (next, armed)
        })
    }

    /// Releases one registration taken by [`Self::arm_dependency`], or the park
    /// bias taken by [`Self::park_begin`].
    ///
    /// This is also the *disarm* operation: a caller that armed a dependency and
    /// then failed to register the waiter releases the slot with exactly this
    /// call. The two uses are indistinguishable to the word, and keeping one
    /// implementation is what keeps the count balanced.
    ///
    /// [`DecOutcome::Requeue`] is observed by exactly one caller per park.
    pub(crate) fn finish_dependency(&self, epoch: u64) -> DecOutcome {
        self.transition(|word| {
            let (next, outcome) = finish_dependency_word(word, epoch);
            // A parked drive at zero is unreachable by this module's own
            // transitions -- the requeue happens on the way down to zero -- so
            // seeing one means the word is being written by something else.
            debug_assert!(
                !(outcome == DecOutcome::Stale
                    && phase_of(word) == DrivePhase::Parked
                    && epoch_of(word) == epoch),
                "parked drive reached zero outstanding without leaving the parked phase"
            );
            (next, outcome)
        })
    }

    /// Alias of [`Self::finish_dependency`] for call sites that release a slot
    /// they armed but never registered; naming that "finish" reads like the
    /// dependency completed.
    pub(crate) fn disarm_dependency(&self, epoch: u64) -> DecOutcome {
        self.finish_dependency(epoch)
    }

    /// Ends a run with nothing left to do. `epoch` is the run ticket from
    /// [`Self::begin_run`] or [`Self::park_begin`].
    ///
    /// Invariant 1, enforced here rather than only asserted in [`Self::arm`]:
    /// `false` means an `arm` or `abandon` landed mid-run and the exit is
    /// refused, so the driver re-evaluates instead of publishing `Idle` over a
    /// change nobody else will act on -- `arm` returned `false` to its own caller
    /// and queued nothing, so this is the last place the re-arm can survive.
    pub(crate) fn to_idle(&self, epoch: u64) -> bool {
        self.transition(|word| {
            if phase_of(word) == DrivePhase::Running && epoch_of(word) == epoch {
                // No bump: the run ended on exactly the ticket it was handed, so
                // nothing an in-flight waiter captured has become wrong. Bumping
                // here would only invalidate the epoch a fresh `arm` is about to
                // hand out.
                (Some(pack(DrivePhase::Idle, 0, epoch)), true)
            } else {
                (None, false)
            }
        })
    }

    /// Ends a run that cannot make progress and must not be retried until
    /// something re-arms it. `epoch` is the run ticket.
    ///
    /// The ticket check matters more here than in [`Self::to_idle`]: leaving a
    /// re-armed run in a phase documented as "do not retry" loses the change
    /// *and* leaves nothing that will ever look at the holder again.
    pub(crate) fn to_stalled(&self, epoch: u64) -> bool {
        self.transition(|word| {
            if phase_of(word) == DrivePhase::Running && epoch_of(word) == epoch {
                // Bumped, unlike `to_idle`: anything still holding this epoch is
                // waiting for a run that has given up, and must report `Stale`
                // rather than requeue a holder nobody is driving.
                (
                    Some(pack(DrivePhase::Stalled, 0, next_epoch(epoch_of(word)))),
                    true,
                )
            } else {
                (None, false)
            }
        })
    }

    /// Withdraws the holder from generation, e.g. because its ticket is gone.
    ///
    /// `true` means the drive is now `Idle` and the caller owns the teardown.
    pub(crate) fn abandon(&self) -> bool {
        self.transition(|word| {
            let epoch = epoch_of(word);
            match phase_of(word) {
                // Invariant 2: never store `Idle` over `Running`. A running drive
                // has a thread inside it; publishing `Idle` lets the next `arm`
                // queue the holder and a second thread starts driving the same
                // chunk -- the failure that killed two of the three earlier
                // designs. The epoch bump is the whole mechanism: the running
                // drive sees it at its next evaluation and retires itself, and
                // `false` tells this caller it does not own the teardown.
                DrivePhase::Running => (
                    Some(pack(
                        DrivePhase::Running,
                        outstanding_of(word),
                        next_epoch(epoch),
                    )),
                    false,
                ),
                DrivePhase::Queued | DrivePhase::Parked | DrivePhase::Stalled => {
                    (Some(pack(DrivePhase::Idle, 0, next_epoch(epoch))), true)
                }
                // Already withdrawn. A bump here would invalidate nothing, since
                // an idle drive has no waiters.
                DrivePhase::Idle => (None, false),
            }
        })
    }

    #[must_use]
    pub(crate) fn snapshot(&self) -> DriveState {
        // One load of one word: the three fields can never be read from
        // different transitions.
        let word = self.word.load(Ordering::SeqCst);
        DriveState {
            phase: phase_of(word),
            outstanding: outstanding_of(word),
            epoch: epoch_of(word),
        }
    }

    #[must_use]
    pub(crate) fn phase(&self) -> DrivePhase {
        self.snapshot().phase
    }

    #[must_use]
    pub(crate) fn epoch(&self) -> u64 {
        self.snapshot().epoch
    }

    #[must_use]
    pub(crate) fn outstanding(&self) -> u32 {
        self.snapshot().outstanding
    }
}

impl Default for GenerationDrive {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        hint,
        panic::{self, AssertUnwindSafe},
        sync::atomic::{AtomicU64, AtomicUsize, Ordering},
        thread,
    };

    use steel_utils::locks::SyncMutex;

    use super::{
        CAS_RETRIES, DecOutcome, DrivePhase, DriveState, EPOCH_MASK, EPOCH_SHIFT, GenerationDrive,
        MAX_OUTSTANDING, OUTSTANDING_MASK, OUTSTANDING_SHIFT, PHASE_MASK, arm_dependency_word,
        finish_dependency_word, pack,
    };

    const ALL_PHASES: [DrivePhase; 5] = [
        DrivePhase::Idle,
        DrivePhase::Queued,
        DrivePhase::Running,
        DrivePhase::Parked,
        DrivePhase::Stalled,
    ];

    /// An epoch far enough from zero that a missing bump is visible.
    const EPOCH: u64 = 0x1234_5678;

    fn drive(phase: DrivePhase, outstanding: u32, epoch: u64) -> GenerationDrive {
        GenerationDrive {
            word: AtomicU64::new(pack(phase, outstanding, epoch)),
        }
    }

    #[track_caller]
    fn assert_state(drive: &GenerationDrive, phase: DrivePhase, outstanding: u32, epoch: u64) {
        let expected = DriveState {
            phase,
            outstanding,
            epoch,
        };
        assert_eq!(drive.snapshot(), expected);
        // The individual accessors must agree with the snapshot, otherwise a
        // caller reading them separately sees a state that never existed.
        assert_eq!(drive.phase(), phase);
        assert_eq!(drive.outstanding(), outstanding);
        assert_eq!(drive.epoch(), epoch);
    }

    #[test]
    fn new_drive_is_idle_at_epoch_zero() {
        assert_state(&GenerationDrive::new(), DrivePhase::Idle, 0, 0);
        assert_state(&GenerationDrive::default(), DrivePhase::Idle, 0, 0);
    }

    #[test]
    fn fields_do_not_overlap() {
        let outstanding_field = OUTSTANDING_MASK << OUTSTANDING_SHIFT;
        let epoch_field = EPOCH_MASK << EPOCH_SHIFT;
        assert_eq!(
            PHASE_MASK | outstanding_field | epoch_field,
            u64::MAX,
            "the three fields must tile the whole word"
        );
        assert_eq!(
            (PHASE_MASK.count_ones() + outstanding_field.count_ones() + epoch_field.count_ones()),
            u64::BITS,
            "the three fields must not overlap"
        );

        // A maximal neighbour must not leak into its neighbours.
        assert_state(
            &drive(DrivePhase::Parked, MAX_OUTSTANDING, 0),
            DrivePhase::Parked,
            MAX_OUTSTANDING,
            0,
        );
        assert_state(
            &drive(DrivePhase::Parked, 0, EPOCH_MASK),
            DrivePhase::Parked,
            0,
            EPOCH_MASK,
        );
        assert_eq!(PHASE_MASK, 0b111);
        assert_eq!(OUTSTANDING_MASK, 0xFFFF);
        assert_eq!(EPOCH_MASK, (1 << 45) - 1);
    }

    #[test]
    fn arm_from_every_phase() {
        let idle = drive(DrivePhase::Idle, 0, EPOCH);
        assert!(idle.arm());
        assert_state(&idle, DrivePhase::Queued, 0, EPOCH + 1);

        let queued = drive(DrivePhase::Queued, 0, EPOCH);
        assert!(!queued.arm());
        assert_state(&queued, DrivePhase::Queued, 0, EPOCH);

        let running = drive(DrivePhase::Running, 4, EPOCH);
        assert!(!running.arm());
        assert_state(&running, DrivePhase::Running, 4, EPOCH + 1);

        let parked = drive(DrivePhase::Parked, 3, EPOCH);
        assert!(parked.arm());
        assert_state(&parked, DrivePhase::Queued, 0, EPOCH + 1);

        let stalled = drive(DrivePhase::Stalled, 0, EPOCH);
        assert!(stalled.arm());
        assert_state(&stalled, DrivePhase::Queued, 0, EPOCH + 1);
    }

    #[test]
    fn begin_run_from_every_phase() {
        let queued = drive(DrivePhase::Queued, 0, EPOCH);
        assert_eq!(queued.begin_run(), Some(EPOCH));
        assert_state(&queued, DrivePhase::Running, 0, EPOCH);

        // A count left over from an abandoned park must not be inherited.
        let queued_with_count = drive(DrivePhase::Queued, 5, EPOCH);
        assert_eq!(queued_with_count.begin_run(), Some(EPOCH));
        assert_state(&queued_with_count, DrivePhase::Running, 0, EPOCH);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Running,
            DrivePhase::Parked,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert_eq!(drive.begin_run(), None, "{phase:?}");
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn park_begin_from_every_phase() {
        let running = drive(DrivePhase::Running, 0, EPOCH);
        assert_eq!(running.park_begin(EPOCH), Some(EPOCH + 1));
        assert_state(&running, DrivePhase::Parked, 1, EPOCH + 1);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Queued,
            DrivePhase::Parked,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert_eq!(drive.park_begin(EPOCH), None, "{phase:?}");
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn park_begin_rejects_a_spent_run_ticket() {
        for epoch in [EPOCH - 1, EPOCH + 1, 0, EPOCH_MASK] {
            let running = drive(DrivePhase::Running, 0, EPOCH);
            assert_eq!(running.park_begin(epoch), None);
            assert_state(&running, DrivePhase::Running, 0, EPOCH);
        }
    }

    #[test]
    fn arm_dependency_from_every_phase() {
        let parked = drive(DrivePhase::Parked, 1, EPOCH);
        assert!(parked.arm_dependency(EPOCH));
        assert_state(&parked, DrivePhase::Parked, 2, EPOCH);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Queued,
            DrivePhase::Running,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert!(!drive.arm_dependency(EPOCH), "{phase:?}");
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn arm_dependency_rejects_other_epochs() {
        for epoch in [EPOCH - 1, EPOCH + 1] {
            let parked = drive(DrivePhase::Parked, 2, EPOCH);
            assert!(!parked.arm_dependency(epoch));
            assert_state(&parked, DrivePhase::Parked, 2, EPOCH);
        }
    }

    #[test]
    fn finish_dependency_from_every_phase() {
        let many = drive(DrivePhase::Parked, 3, EPOCH);
        assert_eq!(many.finish_dependency(EPOCH), DecOutcome::Pending);
        assert_state(&many, DrivePhase::Parked, 2, EPOCH);

        let last = drive(DrivePhase::Parked, 1, EPOCH);
        assert_eq!(last.finish_dependency(EPOCH), DecOutcome::Requeue);
        assert_state(&last, DrivePhase::Queued, 0, EPOCH);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Queued,
            DrivePhase::Running,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert_eq!(
                drive.finish_dependency(EPOCH),
                DecOutcome::Stale,
                "{phase:?}"
            );
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn stale_epoch_decrements_never_touch_a_live_count() {
        for epoch in [EPOCH - 1, EPOCH + 1, 0, EPOCH_MASK] {
            let parked = drive(DrivePhase::Parked, 3, EPOCH);
            assert_eq!(parked.finish_dependency(epoch), DecOutcome::Stale);
            assert_state(&parked, DrivePhase::Parked, 3, EPOCH);

            // `disarm_dependency` is the same operation and must reject the same
            // epochs, or a failed registration could decrement a later park.
            assert_eq!(parked.disarm_dependency(epoch), DecOutcome::Stale);
            assert_state(&parked, DrivePhase::Parked, 3, EPOCH);
        }

        // A stale decrement must not be able to steal the requeue either.
        let last = drive(DrivePhase::Parked, 1, EPOCH);
        assert_eq!(last.finish_dependency(EPOCH - 1), DecOutcome::Stale);
        assert_state(&last, DrivePhase::Parked, 1, EPOCH);
    }

    #[test]
    fn disarm_and_finish_are_the_same_operation() {
        let parked = drive(DrivePhase::Parked, 2, EPOCH);
        assert_eq!(parked.disarm_dependency(EPOCH), DecOutcome::Pending);
        assert_state(&parked, DrivePhase::Parked, 1, EPOCH);
        assert_eq!(parked.disarm_dependency(EPOCH), DecOutcome::Requeue);
        assert_state(&parked, DrivePhase::Queued, 0, EPOCH);
    }

    #[test]
    fn to_idle_from_every_phase() {
        let running = drive(DrivePhase::Running, 0, EPOCH);
        assert!(running.to_idle(EPOCH));
        assert_state(&running, DrivePhase::Idle, 0, EPOCH);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Queued,
            DrivePhase::Parked,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert!(!drive.to_idle(EPOCH), "{phase:?}");
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn to_stalled_from_every_phase() {
        let running = drive(DrivePhase::Running, 3, EPOCH);
        assert!(running.to_stalled(EPOCH));
        assert_state(&running, DrivePhase::Stalled, 0, EPOCH + 1);

        for phase in [
            DrivePhase::Idle,
            DrivePhase::Queued,
            DrivePhase::Parked,
            DrivePhase::Stalled,
        ] {
            let drive = drive(phase, 2, EPOCH);
            assert!(!drive.to_stalled(EPOCH), "{phase:?}");
            assert_state(&drive, phase, 2, EPOCH);
        }
    }

    #[test]
    fn ending_a_run_rejects_a_spent_run_ticket() {
        for epoch in [EPOCH - 1, EPOCH + 1, 0, EPOCH_MASK] {
            let running = drive(DrivePhase::Running, 0, EPOCH);
            assert!(!running.to_idle(epoch));
            assert_state(&running, DrivePhase::Running, 0, EPOCH);

            assert!(!running.to_stalled(epoch));
            assert_state(&running, DrivePhase::Running, 0, EPOCH);
        }
    }

    /// Invariant 1, end to end: a ticket change that lands mid-run must not be
    /// lost. `arm` returns `false` and queues nothing, so every way out of
    /// `Running` has to refuse the spent ticket -- otherwise the run exits, no
    /// queue entry exists anywhere, and the newly allowed status is never
    /// generated.
    #[test]
    fn arm_during_a_run_survives_the_runs_exit() {
        let drive = drive(DrivePhase::Queued, 0, EPOCH);
        let ticket = drive.begin_run().expect("a queued drive can run");

        assert!(!drive.arm(), "an already-running holder must not be queued");
        assert_state(&drive, DrivePhase::Running, 0, EPOCH + 1);

        // None of the three exits may act on the ticket the change invalidated.
        assert!(!drive.to_idle(ticket));
        assert_state(&drive, DrivePhase::Running, 0, EPOCH + 1);
        assert!(!drive.to_stalled(ticket));
        assert_state(&drive, DrivePhase::Running, 0, EPOCH + 1);
        assert_eq!(drive.park_begin(ticket), None);
        assert_state(&drive, DrivePhase::Running, 0, EPOCH + 1);

        // The driver re-evaluates -- reading the status the armer stored before
        // it bumped -- and only then may it exit, on the new ticket.
        let ticket = drive.epoch();
        assert_eq!(ticket, EPOCH + 1);
        assert!(drive.to_idle(ticket));
        assert_state(&drive, DrivePhase::Idle, 0, EPOCH + 1);
    }

    /// Invariant 2's other half: `abandon` refuses to stomp `Running` and
    /// returns `false`, so nobody but the running drive can retire the holder.
    /// That only works if the exits refuse the spent ticket -- a drive that
    /// parked on the pre-abandon dependency set never gets another evaluation.
    #[test]
    fn abandon_during_a_run_retires_only_through_the_driver() {
        let drive = drive(DrivePhase::Queued, 0, EPOCH);
        let ticket = drive.begin_run().expect("a queued drive can run");

        assert!(!drive.abandon(), "the abandoner does not own the teardown");
        assert_state(&drive, DrivePhase::Running, 0, EPOCH + 1);

        assert_eq!(
            drive.park_begin(ticket),
            None,
            "parking here would strand the holder: nothing would ever evaluate it again"
        );
        assert!(!drive.to_idle(ticket));

        let ticket = drive.epoch();
        assert!(drive.to_idle(ticket), "the driver retires itself");
        assert_state(&drive, DrivePhase::Idle, 0, EPOCH + 1);
    }

    #[test]
    fn epoch_moves_exactly_where_the_table_says() {
        type Op = (&'static str, fn(&GenerationDrive), [u64; 5]);

        // Expected bump per phase, in `ALL_PHASES` order. The operations that
        // take a ticket are called with the current epoch, so they are on their
        // accepting path wherever the phase allows it.
        let ops: [Op; 9] = [
            (
                "abandon",
                |d| {
                    let _ = d.abandon();
                },
                [0, 1, 1, 1, 1],
            ),
            (
                "arm",
                |d| {
                    let _ = d.arm();
                },
                [1, 0, 1, 1, 1],
            ),
            (
                "begin_run",
                |d| {
                    let _ = d.begin_run();
                },
                [0, 0, 0, 0, 0],
            ),
            (
                "park_begin",
                |d| {
                    let _ = d.park_begin(d.epoch());
                },
                [0, 0, 1, 0, 0],
            ),
            (
                "arm_dependency",
                |d| {
                    let _ = d.arm_dependency(d.epoch());
                },
                [0, 0, 0, 0, 0],
            ),
            (
                "finish_dependency",
                |d| {
                    let _ = d.finish_dependency(d.epoch());
                },
                [0, 0, 0, 0, 0],
            ),
            (
                "disarm_dependency",
                |d| {
                    let _ = d.disarm_dependency(d.epoch());
                },
                [0, 0, 0, 0, 0],
            ),
            (
                "to_idle",
                |d| {
                    let _ = d.to_idle(d.epoch());
                },
                [0, 0, 0, 0, 0],
            ),
            (
                "to_stalled",
                |d| {
                    let _ = d.to_stalled(d.epoch());
                },
                [0, 0, 1, 0, 0],
            ),
        ];

        for (name, op, expected) in ops {
            for (phase, bump) in ALL_PHASES.into_iter().zip(expected) {
                // Two outstanding: enough that a decrement stays `Pending`, so
                // every operation is exercised on its non-terminal path.
                let drive = drive(phase, 2, EPOCH);
                op(&drive);
                assert_eq!(
                    drive.epoch(),
                    EPOCH + bump,
                    "{name} from {phase:?} moved the epoch by the wrong amount"
                );
            }
        }
    }

    #[test]
    fn epoch_wraps_inside_its_own_field() {
        for (name, op) in [
            (
                "arm",
                (|d: &GenerationDrive| d.arm()) as fn(&GenerationDrive) -> bool,
            ),
            ("to_stalled", |d| d.to_stalled(EPOCH_MASK)),
            ("abandon", |d| d.abandon()),
        ] {
            // `Running` is the state every one of these three accepts, and the
            // one where a carry out of the epoch field would be fatal.
            let drive = drive(DrivePhase::Running, 7, EPOCH_MASK);
            let _ = op(&drive);
            let state = drive.snapshot();
            assert_eq!(state.epoch, 0, "{name} must wrap the epoch to zero");
            assert_ne!(
                state.phase,
                DrivePhase::Idle,
                "{name} must not have carried into the phase field"
            );
        }

        // The wrap must also survive a park handshake: the epoch handed out is
        // the wrapped one, and the registrations must match it.
        let drive = drive(DrivePhase::Running, 0, EPOCH_MASK);
        let epoch = drive
            .park_begin(EPOCH_MASK)
            .expect("running drives park on their own ticket");
        assert_eq!(epoch, 0);
        assert_state(&drive, DrivePhase::Parked, 1, 0);
        assert!(drive.arm_dependency(0));
        assert_eq!(drive.finish_dependency(EPOCH_MASK), DecOutcome::Stale);
        assert_eq!(drive.finish_dependency(0), DecOutcome::Pending);
        assert_eq!(drive.finish_dependency(0), DecOutcome::Requeue);
        assert_state(&drive, DrivePhase::Queued, 0, 0);
    }

    #[test]
    fn full_park_cycle_requeues_exactly_once() {
        const DEPENDENCIES: u32 = 6;

        let drive = drive(DrivePhase::Queued, 0, EPOCH);
        let ticket = drive.begin_run().expect("a queued drive can run");
        assert_eq!(ticket, EPOCH);

        let epoch = drive.park_begin(ticket).expect("a running drive can park");
        assert_eq!(epoch, EPOCH + 1);
        // The bias, not a registration.
        assert_state(&drive, DrivePhase::Parked, 1, epoch);

        for i in 1..=DEPENDENCIES {
            assert!(drive.arm_dependency(epoch));
            assert_state(&drive, DrivePhase::Parked, 1 + i, epoch);
        }

        // Every dependency resolving mid-pass leaves the holder parked, because
        // the bias is still held.
        for i in (1..=DEPENDENCIES).rev() {
            assert_eq!(drive.finish_dependency(epoch), DecOutcome::Pending);
            assert_state(&drive, DrivePhase::Parked, i, epoch);
        }

        // Releasing the bias is what publishes the requeue.
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Requeue);
        assert_state(&drive, DrivePhase::Queued, 0, epoch);

        // A late decrement from the same park must not requeue it twice.
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Stale);
        assert_state(&drive, DrivePhase::Queued, 0, epoch);

        // The requeued entry is claimable, and its ticket is the park's epoch.
        assert_eq!(drive.begin_run(), Some(epoch));
    }

    #[test]
    fn park_bias_survives_a_dependency_firing_mid_pass() {
        let drive = drive(DrivePhase::Running, 0, EPOCH);
        let epoch = drive.park_begin(EPOCH).expect("a running drive can park");

        // First dependency registered, and it fires immediately.
        assert!(drive.arm_dependency(epoch));
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Pending);
        assert_state(&drive, DrivePhase::Parked, 1, epoch);

        // The pass is still open, so later registrations still land.
        assert!(drive.arm_dependency(epoch));
        assert_state(&drive, DrivePhase::Parked, 2, epoch);

        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Pending);
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Requeue);
    }

    #[test]
    fn rearming_a_park_strands_no_registration() {
        let drive = drive(DrivePhase::Running, 0, EPOCH);
        let epoch = drive.park_begin(EPOCH).expect("a running drive can park");
        assert!(drive.arm_dependency(epoch));

        // A ticket change re-arms the parked holder.
        assert!(drive.arm());
        assert_state(&drive, DrivePhase::Queued, 0, epoch + 1);

        // Both the registration and the bias now report `Stale`, so neither can
        // requeue the holder a second time.
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Stale);
        assert_eq!(drive.finish_dependency(epoch), DecOutcome::Stale);
        assert_state(&drive, DrivePhase::Queued, 0, epoch + 1);
    }

    #[test]
    fn outstanding_saturates_instead_of_wrapping() {
        let drive = drive(DrivePhase::Parked, MAX_OUTSTANDING, EPOCH);

        let previous_hook = panic::take_hook();
        // The debug assertion is the expected outcome here; its backtrace would
        // only make a passing test look like a failing one.
        panic::set_hook(Box::new(|_| {}));
        let armed = panic::catch_unwind(AssertUnwindSafe(|| drive.arm_dependency(EPOCH)));
        panic::set_hook(previous_hook);

        if cfg!(debug_assertions) {
            assert!(
                armed.is_err(),
                "a saturated count must trip the debug assertion"
            );
        } else {
            assert!(
                armed.expect("release builds saturate instead of panicking"),
                "the caller must still register, or the holder parks forever"
            );
        }

        // Either way the count must not have wrapped into the epoch field.
        assert_state(&drive, DrivePhase::Parked, MAX_OUTSTANDING, EPOCH);
    }

    /// The saturating branch itself, which `arm_dependency`'s debug assertion
    /// preempts in every `cargo test` build. Answering `false` here would leave
    /// the caller skipping a registration and the holder parked forever, and
    /// nothing else in the suite would notice.
    #[test]
    fn a_saturated_count_still_tells_the_caller_to_register() {
        let word = pack(DrivePhase::Parked, MAX_OUTSTANDING, EPOCH);
        assert_eq!(arm_dependency_word(word, EPOCH), (None, true));

        // One below the cap still counts up normally.
        let word = pack(DrivePhase::Parked, MAX_OUTSTANDING - 1, EPOCH);
        assert_eq!(
            arm_dependency_word(word, EPOCH),
            (Some(pack(DrivePhase::Parked, MAX_OUTSTANDING, EPOCH)), true)
        );
    }

    /// The `outstanding == 0` arm, likewise unreachable through the method in a
    /// debug build. A parked drive at zero cannot arise from this module's own
    /// transitions, but treating it as a requeue would hand out a second queue
    /// entry for a holder that already has one.
    #[test]
    fn a_parked_drive_at_zero_is_stale_not_a_requeue() {
        let word = pack(DrivePhase::Parked, 0, EPOCH);
        assert_eq!(
            finish_dependency_word(word, EPOCH),
            (None, DecOutcome::Stale)
        );
    }

    /// Iterations per racing test.
    const RACE_ITERATIONS: usize = 20_000;

    /// Serialises the racing tests, so [`CAS_RETRIES`] counts exactly one of
    /// them. Single-threaded tests never retry, so they cannot pollute it.
    static RACE_LOCK: SyncMutex<()> = SyncMutex::new(());

    /// Runs a racing test body and fails unless the threads actually collided.
    ///
    /// Without this the racing tests silently degrade into two serial orderings
    /// that a deterministic test already covers -- measured on the `Barrier`
    /// version of these same tests: zero failed compare-exchanges across 20000
    /// supposedly racing operations.
    fn assert_raced(body: impl FnOnce()) {
        let _lock = RACE_LOCK.lock();
        CAS_RETRIES.store(0, Ordering::Relaxed);
        body();
        assert!(
            CAS_RETRIES.load(Ordering::Relaxed) > 0,
            "no compare-exchange ever failed: the threads did not race, so this test \
             proves nothing beyond what the deterministic tests already cover"
        );
    }

    /// A spinning barrier.
    ///
    /// `std::sync::Barrier` releases through a futex, which spreads the threads
    /// by tens of microseconds -- enough that the loser of every race reads the
    /// already-final word on its initial load and the retry loop is never
    /// entered. Spinning keeps them within nanoseconds of each other.
    struct SpinGate {
        arrived: AtomicUsize,
        generation: AtomicUsize,
        threads: usize,
    }

    impl SpinGate {
        fn new(threads: usize) -> Self {
            Self {
                arrived: AtomicUsize::new(0),
                generation: AtomicUsize::new(0),
                threads,
            }
        }

        fn wait(&self) {
            let generation = self.generation.load(Ordering::Acquire);
            if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.threads {
                // Safe to reset before releasing: every other thread is still
                // spinning on `generation` and cannot have re-entered.
                self.arrived.store(0, Ordering::Release);
                self.generation.fetch_add(1, Ordering::Release);
            } else {
                // Counted, not timed: a peer that panicked inside the gate never
                // arrives, and a bare spin turns its assertion failure into a
                // test run that hangs with no output. A clock read in the loop
                // would cost more than the collision window this gate exists to
                // create, so the bound is a spin count several orders of
                // magnitude above any legitimate wait.
                let mut spins: u64 = 0;
                while self.generation.load(Ordering::Acquire) == generation {
                    hint::spin_loop();
                    spins += 1;
                    assert!(
                        spins < 1 << 30,
                        "a thread never reached the gate; it most likely panicked"
                    );
                }
            }
        }
    }

    #[test]
    fn racing_arm_hands_out_exactly_one_queue_entry() {
        assert_raced(|| {
            for start in [DrivePhase::Idle, DrivePhase::Parked, DrivePhase::Stalled] {
                let drives: Vec<GenerationDrive> = (0..RACE_ITERATIONS)
                    .map(|_| drive(start, 2, EPOCH))
                    .collect();
                let gate = SpinGate::new(2);

                let (left, right) = thread::scope(|scope| {
                    let arm_all = || {
                        drives
                            .iter()
                            .map(|drive| {
                                gate.wait();
                                drive.arm()
                            })
                            .collect::<Vec<_>>()
                    };
                    let left = scope.spawn(arm_all);
                    let right = scope.spawn(arm_all);
                    (
                        left.join().expect("arming thread must not panic"),
                        right.join().expect("arming thread must not panic"),
                    )
                });

                for (index, drive) in drives.iter().enumerate() {
                    assert_eq!(
                        usize::from(left[index]) + usize::from(right[index]),
                        1,
                        "{start:?}: exactly one caller may own the queue entry"
                    );
                    // One bump for the winner; the loser is the `Queued` no-op
                    // row, so a second bump would mean a second transition.
                    assert_state(drive, DrivePhase::Queued, 0, EPOCH + 1);
                }
            }
        });
    }

    #[test]
    fn park_begin_racing_arm_never_parks_on_stale_information() {
        assert_raced(|| {
            let drives: Vec<GenerationDrive> = (0..RACE_ITERATIONS)
                .map(|_| drive(DrivePhase::Running, 0, EPOCH))
                .collect();
            let gate = SpinGate::new(2);

            let (parks, arms) = thread::scope(|scope| {
                let parker = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.park_begin(EPOCH)
                        })
                        .collect::<Vec<_>>()
                });
                let armer = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.arm()
                        })
                        .collect::<Vec<_>>()
                });
                (
                    parker.join().expect("parking thread must not panic"),
                    armer.join().expect("arming thread must not panic"),
                )
            });

            // Both orderings are pinned absolutely, not relative to what
            // `park_begin` returned: a returned epoch that merely agrees with
            // the word is exactly what a park built on a lost `arm` looks like.
            for (index, drive) in drives.iter().enumerate() {
                match (parks[index], arms[index]) {
                    // Park won: it installed EPOCH + 1, then `arm` found a
                    // `Parked` drive and took it away.
                    (Some(epoch), true) => {
                        assert_eq!(epoch, EPOCH + 1);
                        assert_state(drive, DrivePhase::Queued, 0, EPOCH + 2);
                    }
                    // `arm` won: the run ticket is spent, so the park must be
                    // refused rather than land on the pre-change halo.
                    (None, false) => assert_state(drive, DrivePhase::Running, 0, EPOCH + 1),
                    (park, armed) => panic!(
                        "park_begin returned {park:?} while arm returned {armed}: \
                         the park either lost the re-arm or stole its queue entry"
                    ),
                }
            }
        });
    }

    #[test]
    fn arm_racing_the_end_of_a_run_never_loses_the_rearm() {
        assert_raced(|| {
            let drives: Vec<GenerationDrive> = (0..RACE_ITERATIONS)
                .map(|_| drive(DrivePhase::Running, 0, EPOCH))
                .collect();
            let gate = SpinGate::new(2);

            let (arms, exits) = thread::scope(|scope| {
                let armer = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.arm()
                        })
                        .collect::<Vec<_>>()
                });
                let runner = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.to_idle(EPOCH)
                        })
                        .collect::<Vec<_>>()
                });
                (
                    armer.join().expect("arming thread must not panic"),
                    runner.join().expect("driving thread must not panic"),
                )
            });

            for (index, drive) in drives.iter().enumerate() {
                match (arms[index], exits[index]) {
                    // The run ended first, so `arm` found an idle drive and owns
                    // the queue entry.
                    (true, true) => assert_state(drive, DrivePhase::Queued, 0, EPOCH + 1),
                    // `arm` landed first: it queued nothing, so the exit must be
                    // refused and the driver must come round again. An `Idle`
                    // drive here is the lost re-arm -- the chunk would never
                    // reach the newly allowed status.
                    (false, false) => assert_state(drive, DrivePhase::Running, 0, EPOCH + 1),
                    (armed, exited) => panic!(
                        "arm returned {armed} while to_idle returned {exited}: \
                         the re-arm was dropped with no queue entry anywhere"
                    ),
                }
            }
        });
    }

    #[test]
    fn arm_dependency_racing_the_end_of_a_park_never_strands_a_count() {
        assert_raced(|| {
            // Parked holding only the bias, so a single decrement ends the park.
            let drives: Vec<GenerationDrive> = (0..RACE_ITERATIONS)
                .map(|_| drive(DrivePhase::Parked, 1, EPOCH))
                .collect();
            let gate = SpinGate::new(2);

            let (registrations, restarts) = thread::scope(|scope| {
                let registrar = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.arm_dependency(EPOCH)
                        })
                        .collect::<Vec<_>>()
                });
                let publisher = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            let outcome = drive.finish_dependency(EPOCH);
                            if outcome == DecOutcome::Requeue {
                                // The park is over; the holder is admitted again
                                // and parks afresh, one epoch on. A registration
                                // that lands now belongs to nothing.
                                let ticket = drive.begin_run().expect("requeued drives run");
                                assert_eq!(ticket, EPOCH);
                                assert_eq!(drive.park_begin(ticket), Some(EPOCH + 1));
                            }
                            outcome
                        })
                        .collect::<Vec<_>>()
                });
                (
                    registrar.join().expect("registering thread must not panic"),
                    publisher.join().expect("publishing thread must not panic"),
                )
            });

            for (index, drive) in drives.iter().enumerate() {
                match (registrations[index], restarts[index]) {
                    // The registration landed inside the park, so the decrement
                    // found two and left the bias.
                    (true, DecOutcome::Pending) => {
                        assert_state(drive, DrivePhase::Parked, 1, EPOCH);
                    }
                    // The park ended first, so the registration was refused. A
                    // count of two here is the phantom registration: nobody
                    // holds a waiter for it and the holder never leaves `Parked`.
                    (false, DecOutcome::Requeue) => {
                        assert_state(drive, DrivePhase::Parked, 1, EPOCH + 1);
                    }
                    (armed, outcome) => panic!(
                        "arm_dependency returned {armed} while finish_dependency returned \
                         {outcome:?}: a registration crossed the end of its park"
                    ),
                }
            }
        });
    }

    #[test]
    fn finish_dependency_storm_requeues_exactly_once() {
        const THREADS: u32 = 8;
        const DECREMENTS_PER_THREAD: usize = 2;
        const ITERATIONS: usize = 2000;

        assert_raced(|| {
            // Oversubscribed on purpose: 16 decrements against 8 registrations.
            // With exactly one decrement per thread the late-decrement path is
            // unreachable by construction, and `Stale` is what stops a
            // decrement that raced past the requeue from underflowing the count
            // or handing out a second queue entry.
            let drives: Vec<GenerationDrive> = (0..ITERATIONS)
                .map(|_| drive(DrivePhase::Parked, THREADS, EPOCH))
                .collect();
            let gate = SpinGate::new(THREADS as usize);

            let outcomes = thread::scope(|scope| {
                let handles: Vec<_> = (0..THREADS)
                    .map(|_| {
                        let gate = &gate;
                        let drives = &drives;
                        scope.spawn(move || {
                            drives
                                .iter()
                                .flat_map(|drive| {
                                    gate.wait();
                                    [(); DECREMENTS_PER_THREAD].map(|()| {
                                        let outcome = drive.finish_dependency(EPOCH);
                                        // A wrapped count shows up here as ~65k,
                                        // long before the final state is read.
                                        let observed = drive.outstanding();
                                        assert!(
                                            observed < THREADS,
                                            "outstanding {observed} exceeds the registrations taken"
                                        );
                                        outcome
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("decrementing thread must not panic"))
                    .collect::<Vec<_>>()
            });

            for (index, drive) in drives.iter().enumerate() {
                let count = |wanted: DecOutcome| {
                    outcomes
                        .iter()
                        .flat_map(|thread| &thread[index * DECREMENTS_PER_THREAD..][..DECREMENTS_PER_THREAD])
                        .filter(|outcome| **outcome == wanted)
                        .count()
                };
                assert_eq!(count(DecOutcome::Requeue), 1, "exactly one caller may requeue a park");
                assert_eq!(
                    count(DecOutcome::Pending),
                    THREADS as usize - 1,
                    "no decrement may be lost"
                );
                assert_eq!(
                    count(DecOutcome::Stale),
                    THREADS as usize * DECREMENTS_PER_THREAD - THREADS as usize,
                    "every decrement past the requeue must be refused"
                );
                assert_state(drive, DrivePhase::Queued, 0, EPOCH);
            }
        });
    }

    #[test]
    fn abandon_racing_begin_run_never_yields_two_running_observers() {
        assert_raced(|| {
            let drives: Vec<GenerationDrive> = (0..RACE_ITERATIONS)
                .map(|_| drive(DrivePhase::Queued, 0, EPOCH))
                .collect();
            let gate = SpinGate::new(2);

            let (abandons, runs) = thread::scope(|scope| {
                let abandoner = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.abandon()
                        })
                        .collect::<Vec<_>>()
                });
                let runner = scope.spawn(|| {
                    drives
                        .iter()
                        .map(|drive| {
                            gate.wait();
                            drive.begin_run()
                        })
                        .collect::<Vec<_>>()
                });
                (
                    abandoner.join().expect("abandoning thread must not panic"),
                    runner.join().expect("running thread must not panic"),
                )
            });

            for (index, drive) in drives.iter().enumerate() {
                match (abandons[index], runs[index]) {
                    // `begin_run` lost: the entry was withdrawn before it was
                    // claimed, so nobody drives the chunk.
                    (true, None) => assert_state(drive, DrivePhase::Idle, 0, EPOCH + 1),
                    // `begin_run` won: `abandon` must have refused to stomp
                    // `Running` and only moved the epoch on, leaving exactly one
                    // driver, whose ticket is now spent so it must re-evaluate.
                    (false, Some(epoch)) => {
                        assert_eq!(epoch, EPOCH);
                        assert_state(drive, DrivePhase::Running, 0, EPOCH + 1);
                        assert!(!drive.to_idle(epoch));
                    }
                    (abandoned, run) => panic!(
                        "abandon returned {abandoned} while begin_run returned {run:?}: \
                         that is either two drivers or none"
                    ),
                }
            }
        });
    }
}
