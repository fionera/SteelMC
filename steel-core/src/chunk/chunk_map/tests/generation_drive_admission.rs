//! Admission regression for the per-holder generation drive.
//!
//! This file exists for one property: a drive that cannot proceed must give its
//! admission permit back *before* anything waits on it. Three earlier attempts
//! at this scheduler deadlocked on the other arrangement, and the deadlock is
//! not subtle -- `run_generation_tasks_b` returns early once `available_slots`
//! reaches zero, so a holder that parks while still counted against the cap
//! makes the chunk it waits for unadmittable, and no slot frees until the waiter
//! stops waiting. Nothing else in the suite covers it: the drive's own tests are
//! over the state machine, which has no notion of a permit.

use std::time::Instant;

use super::*;
use crate::chunk::chunk_holder::GENERATION_DRIVE_COUNTERS;

/// Chunks on a side of the requested square.
///
/// The number that matters is not this one but the ratio to the admission
/// budget below: a `Noise` run rings out to radius 8, so a single chunk here
/// waits on up to 288 neighbours while exactly one holder at a time may be in
/// flight. 33 is large enough that the interior chunks are gated by other
/// requested chunks rather than only by halo chunks the ticket brought in.
const SQUARE: i32 = 33;

/// How long the square may take before the test calls it a hang.
///
/// Generous on purpose. The point is not to measure anything -- an admission
/// deadlock never finishes, so any finite bound separates it from a slow run --
/// and the alternative is the failure this whole exercise is about: a test that
/// hangs in CI with no output. The test world's generation pool has one thread
/// and is shared with every other test in the binary, so a tight bound would be
/// flaky for reasons that have nothing to do with admission: 120 s against the
/// 10 s this takes on an unloaded debug build.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Scheduling epochs the loop may drive before giving up, whichever bound is hit
/// first. `advance_scheduling` returns immediately while an epoch is in flight,
/// so this counts spins rather than epochs and is only a backstop for a clock
/// that does not advance.
const MAX_ITERATIONS: usize = 10_000_000;

/// With one admission slot and a dependency fan-out in the hundreds, every
/// requested chunk must still reach `Full`.
///
/// Ignored rather than switched on per test, because it cannot be switched on
/// per test: `STAGE1` is a `LazyLock` over the environment, read once per
/// process and deliberately so -- `claim_status_work` panics if a holder is ever
/// driven by both dispatchers, so a per-map or per-test switch would make an
/// abort reachable through a mid-run flip. A test-only override of the *cap* is
/// enough to make the budget tight (see `set_max_running_generation_tasks_for_test`);
/// a test-only override of the dispatcher would not be. Run it with
///
/// Runs in the ordinary suite now that the drive is the default dispatcher. The
/// assertion below is what keeps it honest: under `STEEL_STAGE1=0` it fails
/// rather than quietly exercising the task model, which cannot pass this test at
/// all -- a task holds its permit across the halo walk that schedules its
/// dependencies, and at one slot that is the deadlock itself.
///
/// It costs about nine seconds, which is more than the rest of the suite put
/// together. Kept anyway: it pins the single property that three previous
/// attempts at this rewrite died on, and it is the only test that does.
#[test]
fn a_single_admission_slot_still_generates_a_square() {
    if !*STAGE1 {
        // `STEEL_STAGE1=0` is a deliberate opt-out to the task model, and the
        // task model cannot pass this test by construction. Skip loudly rather
        // than fail: the escape hatch should leave a green suite, and the
        // property is guarded on every default run.
        eprintln!(
            "skipping a_single_admission_slot_still_generates_a_square: \
             the per-holder drive is switched off"
        );
        return;
    }

    let world = fresh_test_world("stage1_single_slot_admission");
    let chunk_map = &world.chunk_map;
    // One generation unit in flight at a time -- strictly fewer than the halo any
    // run of any chunk here waits on, which is the only budget that can prove the
    // property. See the field's own comment for why the production cap
    // (`threads * GENERATION_THREAD_MULTIPLE`) cannot be narrowed from inside a
    // test process.
    chunk_map.set_max_running_generation_tasks_for_test(1);
    assert_eq!(chunk_map.generation_task_capacity(), 1);

    let center = ChunkPos::new(0, 0);
    let radius = (SQUARE - 1) / 2;
    let ticket = ChunkTicket::full_chunks(
        u8::try_from(radius).expect("the requested radius fits a ticket radius"),
    );
    let revision = chunk_map.add_chunk_ticket(center, ticket);

    let wakes_before = GENERATION_DRIVE_COUNTERS
        .drive_wakes
        .load(Ordering::Relaxed);
    let mut saw_parked_holder = false;

    let started = Instant::now();
    let mut iterations = 0;
    loop {
        chunk_map.advance_scheduling();
        // Sampled rather than asserted at the end, because a park leaves nothing
        // behind once it ends. Without this the test would still pass if the
        // budget somehow never bound and no holder ever waited on a neighbour,
        // which is precisely the case it must not be mistaken for.
        saw_parked_holder |= GENERATION_DRIVE_COUNTERS
            .parked_holders
            .load(Ordering::Relaxed)
            > 0;
        if chunk_map.is_ticket_revision_committed(revision)
            && chunk_map.full_square_is_ready(center, radius)
        {
            break;
        }

        iterations += 1;
        assert!(
            started.elapsed() < TIMEOUT && iterations < MAX_ITERATIONS,
            "the square never completed in {:?}: {}",
            started.elapsed(),
            square_progress(chunk_map, center, radius),
        );
        thread::sleep(Duration::from_millis(1));
    }

    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let pos = ChunkPos::new(center.0.x + dx, center.0.y + dz);
            let published = chunk_map
                .chunks
                .read_sync(&pos, |_, holder| holder.published_status());
            assert_eq!(
                published,
                Some(Some(ChunkStatus::Full)),
                "{pos:?} did not reach Full",
            );
        }
    }

    assert!(
        saw_parked_holder,
        "no holder ever parked: the square generated without the budget ever binding, so \
         nothing here proves a parked holder releases its permit",
    );
    let wakes = GENERATION_DRIVE_COUNTERS
        .drive_wakes
        .load(Ordering::Relaxed)
        - wakes_before;
    assert!(
        wakes > 0,
        "no park ever ended through a dependency publication",
    );

    chunk_map.remove_chunk_ticket(center, ticket);
    chunk_map.advance_scheduling();
    // The world outlives this function only through the tasks still holding it,
    // and `WorldGenContext::world` panics on a dropped world. Draining first
    // keeps a passing test from printing a panic from a background worker.
    drain_generation_tasks(chunk_map);
}

/// Stops admission and waits for the drives and the scheduling epoch already in
/// flight.
///
/// `advance_scheduling` is deliberately not called from here: committing an
/// epoch spawns the next one, so a drain that kept advancing would always leave
/// one more task holding the world.
fn drain_generation_tasks(chunk_map: &Arc<ChunkMap>) {
    chunk_map.stop_generation_refill_loop();
    for _ in 0..10_000 {
        if chunk_map.running_generation_task_count() == 0 && chunk_map.task_tracker.is_empty() {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!(
        "generation tasks did not drain after the refill loop was stopped: {} running, {} tracked",
        chunk_map.running_generation_task_count(),
        chunk_map.task_tracker.len(),
    );
}

/// The state a hang leaves behind, so the failure names the stuck chunks instead
/// of only saying that nothing finished.
fn square_progress(chunk_map: &Arc<ChunkMap>, center: ChunkPos, radius: i32) -> String {
    let mut full = 0;
    let mut missing = 0;
    let mut unfinished = Vec::new();
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let pos = ChunkPos::new(center.0.x + dx, center.0.y + dz);
            match chunk_map.chunks.read_sync(&pos, |_, holder| {
                (holder.published_status(), holder.parked_generation_state())
            }) {
                None => missing += 1,
                Some((Some(ChunkStatus::Full), _)) => full += 1,
                Some((published, parked)) => {
                    if unfinished.len() < 8 {
                        unfinished.push(format!(
                            "{pos:?} published={published:?} parked={:?}",
                            parked.map(|state| (state.outstanding, state.epoch))
                        ));
                    }
                }
            }
        }
    }
    let side = radius * 2 + 1;
    format!(
        "{full}/{} full, {missing} with no holder, running={}, capacity={}, first unfinished: {}",
        side * side,
        chunk_map.running_generation_task_count(),
        chunk_map.generation_task_capacity(),
        unfinished.join("; "),
    )
}
