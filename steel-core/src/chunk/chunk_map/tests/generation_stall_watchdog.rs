//! The global stall watchdog.
//!
//! The failure it reports is invisible by construction: a holder parked with
//! nothing left that will wake it holds no task, no permit and no halo, so a
//! stalled pregeneration looks exactly like a slow one -- it produced no output
//! for fifteen minutes once and had to be diagnosed by rebuilding with extra
//! logging. What is tested here is therefore the trip condition, which is the
//! part that can be wrong in both directions: a watchdog that fires on a
//! transient is noise nobody reads, and one that never fires is the fifteen
//! minutes again.

use super::generation_drive_dispatch::{ParkBlocker, STALL_WATCHDOG_EPOCHS};
use super::*;

/// Inserts a holder into `chunk_map` and parks its drive on a dependency that
/// will never resolve, which is the state the watchdog exists to find.
fn insert_parked_holder(chunk_map: &Arc<ChunkMap>, holder: &Arc<ChunkHolder>) {
    let _ = chunk_map
        .chunks
        .insert_sync(holder.get_pos(), Arc::clone(holder));
    assert!(holder.arm(), "a fresh drive can be armed");
    let ticket = holder
        .begin_generation_run()
        .expect("an armed drive can run");
    holder.park_begin(ticket).expect("a running drive can park");
    assert!(holder.parked_generation_state().is_some());
}

/// Runs `epochs` watchdog passes and answers whether the stall was reported.
fn run_watchdog_epochs(chunk_map: &Arc<ChunkMap>, epochs: u32) -> bool {
    for _ in 0..epochs {
        chunk_map.check_generation_stall_watchdog();
    }
    chunk_map.generation_stall_reported.load(Ordering::Relaxed)
}

#[test]
fn a_park_nothing_can_wake_is_reported_after_the_full_epoch_count() {
    let chunk_map = test_chunk_map();
    let holder = unloaded_light_holder(ChunkPos::new(4, -9));
    insert_parked_holder(&chunk_map, &holder);

    assert!(
        !run_watchdog_epochs(&chunk_map, STALL_WATCHDOG_EPOCHS - 1),
        "one epoch short of the bound must stay quiet: the idleness test is a \
         sample and a short run of idle epochs is a normal refill window",
    );
    assert!(run_watchdog_epochs(&chunk_map, 1));

    // The report is latched for the episode, not repeated per epoch: a stall
    // does not resolve on its own, so an unlatched watchdog would print 32
    // chunks every few milliseconds until the operator killed the server. The
    // latch is released by the pipeline having work again, and only by that.
    chunk_map
        .pending_generation_tasks
        .lock()
        .push(unloaded_full_holder(ChunkPos::new(5, -9)));
    assert!(!run_watchdog_epochs(&chunk_map, 1));
    chunk_map.pending_generation_tasks.lock().clear();
    assert!(
        run_watchdog_epochs(&chunk_map, STALL_WATCHDOG_EPOCHS),
        "a stall that outlives the work that interrupted it must be reported again",
    );

    holder.abandon_generation_drive();
}

#[test]
fn queued_work_keeps_the_watchdog_quiet() {
    let chunk_map = test_chunk_map();
    let holder = unloaded_light_holder(ChunkPos::new(-2, 6));
    insert_parked_holder(&chunk_map, &holder);

    // A holder waiting for admission is the ordinary state of a parked chunk
    // whose dependency has just published: parked gauge non-zero, nothing
    // running, and a wake already on its way.
    let waiting = unloaded_full_holder(ChunkPos::new(-3, 6));
    chunk_map.pending_generation_tasks.lock().push(waiting);

    assert!(!run_watchdog_epochs(&chunk_map, STALL_WATCHDOG_EPOCHS * 2));

    // The same counter must start again from zero once the queue drains, rather
    // than resuming where the work interrupted it.
    chunk_map.pending_generation_tasks.lock().clear();
    assert!(!run_watchdog_epochs(&chunk_map, STALL_WATCHDOG_EPOCHS - 1));
    assert!(run_watchdog_epochs(&chunk_map, 1));

    holder.abandon_generation_drive();
}

#[test]
fn a_stalled_drive_awaiting_its_backoff_is_not_a_stalled_pipeline() {
    let chunk_map = test_chunk_map();
    let holder = unloaded_light_holder(ChunkPos::new(11, 11));
    insert_parked_holder(&chunk_map, &holder);

    // `revive_stalled_generation_drives` re-arms this holder once its backoff
    // expires, and that backoff reaches 640 ms -- over a hundred epochs at the
    // 5 ms mean. Without this exclusion every deep backoff would be reported as
    // a stall that then resolves itself.
    let stalled = unloaded_full_holder(ChunkPos::new(12, 11));
    chunk_map
        .stalled_generation_drives
        .lock()
        .push(StalledGeneration {
            holder: Arc::downgrade(&stalled),
            reason: StallReason::Refused,
            retry_after: Instant::now() + Duration::from_secs(1),
        });

    assert!(!run_watchdog_epochs(&chunk_map, STALL_WATCHDOG_EPOCHS * 2));

    holder.abandon_generation_drive();
}

#[test]
fn another_maps_parked_holders_are_not_this_maps_stall() {
    let parked_elsewhere = test_chunk_map();
    let holder = unloaded_light_holder(ChunkPos::new(0, 0));
    insert_parked_holder(&parked_elsewhere, &holder);

    // The parked gauge is process-wide -- `DependencyWaiter::drop` runs on rayon
    // workers that hold no map -- so an idle Nether would otherwise report the
    // Overworld's parks as its own stall, naming no chunks at all.
    let idle_map = test_chunk_map();
    assert!(!run_watchdog_epochs(&idle_map, STALL_WATCHDOG_EPOCHS * 2));

    holder.abandon_generation_drive();
}

/// The report has to say what each holder is waiting *for*; the whole point is
/// to turn a hang into a list of chunks and the statuses that gate them.
#[test]
fn the_report_names_what_the_next_run_is_waiting_for() {
    let chunk_map = test_chunk_map();
    let center = ChunkPos::new(20, -20);
    let holder = unloaded_light_holder(center);
    let _ = chunk_map.chunks.insert_sync(center, Arc::clone(&holder));

    // Published `Light`, so the next run starts at `Spawn`, whose ring needs
    // `Light` of itself and `Biomes` of its eight neighbours.
    assert_eq!(holder.published_status(), Some(ChunkStatus::Light));
    let next = ChunkStatus::Spawn;

    match chunk_map.park_blocker(center, next) {
        ParkBlocker::MissingHolder(pos) => assert_eq!(
            pos.0
                .x
                .abs_diff(center.0.x)
                .max(pos.0.y.abs_diff(center.0.y)),
            1,
            "a halo cell with no holder at all must be named as such, because \
             nothing there can ever publish the status the park waits for",
        ),
        blocker => panic!("expected a missing halo cell, got {blocker:?}"),
    }

    for dz in -1..=1 {
        for dx in -1..=1 {
            let pos = ChunkPos::new(center.0.x + dx, center.0.y + dz);
            if pos == center {
                continue;
            }
            // Present but unpublished: exactly what a neighbour that has not
            // caught up yet looks like.
            let neighbor = Arc::new(ChunkHolder::new(
                pos,
                ChunkTicketLevel::FULL_CHUNK,
                Some(ChunkTicketLevel::FULL_CHUNK),
                0,
                16,
            ));
            let _ = chunk_map.chunks.insert_sync(pos, neighbor);
        }
    }

    match chunk_map.park_blocker(center, next) {
        ParkBlocker::Dependency { chunk, required } => {
            assert_ne!(chunk, center, "the centre's own status is already met");
            assert_eq!(required, ChunkStatus::Biomes);
        }
        blocker => panic!("expected an unmet neighbour, got {blocker:?}"),
    }
}
