//! Scheduling gate for light work cache windows.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use steel_utils::{ChunkPos, locks::SyncMutex};
use tokio::sync::oneshot;

use super::LIGHT_CACHE_RADIUS;

const LIGHT_WORK_CENTER_EXCLUSION_RADIUS: i32 = LIGHT_CACHE_RADIUS * 2;
/// Side length of the bucket grid used to index centers.
///
/// One more than the exclusion radius, so two centers in cells that differ by
/// more than one on either axis can never conflict. That makes both the conflict
/// test and the search for unblocked waiters a fixed nine-cell probe.
const LIGHT_WORK_GRID_CELL: i32 = LIGHT_WORK_CENTER_EXCLUSION_RADIUS + 1;

/// Shared gate used to exclude overlapping light-engine cache windows.
///
/// Light worksets can write light data across a full 5x5 cache window. Two
/// worksets whose windows overlap must therefore not run at the same time:
/// chunk-light status work must publish `ChunkStatus::Light` before overlapping
/// work builds its light cache, and queued light updates must not interleave
/// with either operation.
///
/// Waiters are handed their reservation by whoever releases a conflicting one,
/// rather than being broadcast-woken to re-contend for it. The previous design
/// kept a flat list of active centers and called `Notify::notify_waiters` on
/// every release, waking *every* blocked reserver so each could re-take this
/// mutex and rescan. Under pregeneration that convoy was the largest single cost
/// in the server: profiling put **27% of the whole machine** inside this gate,
/// more than half of it in `try_reserve_centered` alone. With hand-off a release
/// wakes only the waiters it actually unblocks, and they wake already holding
/// the window.
#[derive(Debug)]
pub(crate) struct LightWorkWindowGate {
    state: SyncMutex<GateState>,
}

#[derive(Debug, Default)]
struct GateState {
    /// Centers of the windows currently reserved, bucketed by grid cell.
    active: FxHashMap<(i32, i32), Vec<ChunkPos>>,
    /// Reservers blocked behind a conflicting window, bucketed by grid cell.
    waiters: FxHashMap<(i32, i32), Vec<Waiter>>,
}

/// What a blocked reserver left behind to be run once its window is free.
///
/// It is handed the reservation directly, so a caller does not have to be a
/// future parked on a channel. That matters under pregeneration: with a worker
/// per core and an exclusion radius of `LIGHT_WORK_CENTER_EXCLUSION_RADIUS`, a
/// dense generation front can have every worker wanting an overlapping window at
/// once, and blocking a pool thread there stalls the very tasks that would
/// release it.
///
/// Continuations run without the gate lock held, but they do run on the releaser's
/// thread. Hand the work off rather than doing it inline: a continuation that runs
/// a whole workset and then drops its reservation grants the next waiter from
/// inside its own drop, so a chain of them recurses on the releasing thread's
/// stack.
type GrantContinuation = Box<dyn FnOnce(LightWorkWindowReservation) + Send>;

struct Waiter {
    center: ChunkPos,
    /// Run once the window has been reserved on this waiter's behalf.
    grant: GrantContinuation,
}

impl std::fmt::Debug for Waiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Waiter")
            .field("center", &self.center)
            .finish_non_exhaustive()
    }
}

/// Reservation for one light-engine cache window.
#[derive(Debug)]
pub(crate) struct LightWorkWindowReservation {
    gate: Arc<LightWorkWindowGate>,
    center: ChunkPos,
}

impl LightWorkWindowGate {
    /// Creates an empty light work gate.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: SyncMutex::new(GateState::default()),
        }
    }

    /// Runs `grant` with the radius-2 light cache window centered on `center`,
    /// either immediately or once whoever holds a conflicting window releases it.
    ///
    /// Never blocks the calling thread. See [`GrantContinuation`] for what a
    /// continuation may do.
    pub(crate) fn reserve_centered_with<F>(self: &Arc<Self>, center: ChunkPos, grant: F)
    where
        F: FnOnce(LightWorkWindowReservation) + Send + 'static,
    {
        {
            let mut state = self.state.lock();
            if !state.try_insert_active(center) {
                // Registering has to happen under the same lock hold as the
                // failed attempt. A release landing between the two would find
                // no waiter to hand the window to, and this continuation would
                // never run.
                state
                    .waiters
                    .entry(Self::grid_cell(center))
                    .or_default()
                    .push(Waiter {
                        center,
                        grant: Box::new(grant),
                    });
                return;
            }
        }

        grant(LightWorkWindowReservation {
            gate: Arc::clone(self),
            center,
        });
    }

    /// Reserves the radius-2 light cache window centered on `center`.
    ///
    /// Parks a future on the reservation. Prefer [`Self::reserve_centered_with`]
    /// anywhere the caller occupies a pool thread while it waits.
    ///
    /// # Panics
    /// Panics if the granting continuation is dropped without being run, which
    /// cannot happen while this future holds an `Arc` to the gate.
    pub(crate) async fn reserve_centered(
        self: &Arc<Self>,
        center: ChunkPos,
    ) -> LightWorkWindowReservation {
        let (grant, granted) = oneshot::channel();
        self.reserve_centered_with(center, move |reservation| {
            // A send failure means this future was dropped before the window
            // came free; the reservation then drops here and is handed on.
            let _ = grant.send(reservation);
        });

        granted
            .await
            .expect("light work window grant continuation dropped without running")
    }

    /// Attempts to reserve the radius-2 light cache window centered on `center`.
    pub(crate) fn try_reserve_centered(
        self: &Arc<Self>,
        center: ChunkPos,
    ) -> Option<LightWorkWindowReservation> {
        self.state
            .lock()
            .try_insert_active(center)
            .then(|| LightWorkWindowReservation {
                gate: Arc::clone(self),
                center,
            })
    }

    const fn grid_cell(center: ChunkPos) -> (i32, i32) {
        (
            center.0.x.div_euclid(LIGHT_WORK_GRID_CELL),
            center.0.y.div_euclid(LIGHT_WORK_GRID_CELL),
        )
    }

    const fn windows_overlap(left: ChunkPos, right: ChunkPos) -> bool {
        let dx = left.0.x.abs_diff(right.0.x);
        let dz = left.0.y.abs_diff(right.0.y);
        dx <= LIGHT_WORK_CENTER_EXCLUSION_RADIUS as u32
            && dz <= LIGHT_WORK_CENTER_EXCLUSION_RADIUS as u32
    }
}

impl GateState {
    /// Reserves `center` if nothing overlapping is active.
    fn try_insert_active(&mut self, center: ChunkPos) -> bool {
        if self.is_blocked(center) {
            return false;
        }
        self.insert_active(center);
        true
    }

    fn insert_active(&mut self, center: ChunkPos) {
        self.active
            .entry(LightWorkWindowGate::grid_cell(center))
            .or_default()
            .push(center);
    }

    /// Whether an active window overlaps `center`.
    fn is_blocked(&self, center: ChunkPos) -> bool {
        let (cx, cz) = LightWorkWindowGate::grid_cell(center);
        (-1..=1).any(|dx| {
            (-1..=1).any(|dz| {
                self.active.get(&(cx + dx, cz + dz)).is_some_and(|bucket| {
                    bucket
                        .iter()
                        .any(|&active| LightWorkWindowGate::windows_overlap(center, active))
                })
            })
        })
    }

    fn remove_active(&mut self, center: ChunkPos) -> bool {
        let cell = LightWorkWindowGate::grid_cell(center);
        let Some(bucket) = self.active.get_mut(&cell) else {
            return false;
        };
        let Some(index) = bucket.iter().position(|&active| active == center) else {
            return false;
        };
        bucket.swap_remove(index);
        if bucket.is_empty() {
            self.active.remove(&cell);
        }
        true
    }

    /// Claims the freed window for every waiter that `released` was blocking,
    /// and returns their continuations for the caller to run.
    ///
    /// The windows are already reserved on the waiters' behalf when this
    /// returns, so a racing reserver cannot take one out from under them. The
    /// continuations are handed back rather than run here because running one
    /// under the gate lock would deadlock the moment it touched the gate again —
    /// which it does as soon as it drops its reservation.
    ///
    /// Only waiters within the exclusion radius of the released center can have
    /// been unblocked by it, and those all live in the nine cells around it, so
    /// this never walks the whole waiter set.
    #[must_use]
    fn grant_unblocked(&mut self, released: ChunkPos) -> Vec<(ChunkPos, GrantContinuation)> {
        let mut granted = Vec::new();
        let (cx, cz) = LightWorkWindowGate::grid_cell(released);
        for dx in -1..=1 {
            for dz in -1..=1 {
                let cell = (cx + dx, cz + dz);
                if !self.waiters.contains_key(&cell) {
                    continue;
                }

                let mut index = 0;
                while let Some(bucket) = self.waiters.get(&cell) {
                    let Some(waiter) = bucket.get(index) else {
                        break;
                    };
                    let center = waiter.center;
                    if !LightWorkWindowGate::windows_overlap(released, center)
                        || self.is_blocked(center)
                    {
                        index += 1;
                        continue;
                    }

                    let waiter = self
                        .waiters
                        .get_mut(&cell)
                        .expect("waiter bucket vanished while granting")
                        .swap_remove(index);
                    self.insert_active(center);
                    granted.push((center, waiter.grant));
                }

                if self.waiters.get(&cell).is_some_and(Vec::is_empty) {
                    self.waiters.remove(&cell);
                }
            }
        }
        granted
    }
}

impl Default for LightWorkWindowGate {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LightWorkWindowReservation {
    fn drop(&mut self) {
        let granted = {
            let mut state = self.gate.state.lock();
            let removed = state.remove_active(self.center);
            debug_assert!(removed, "light work reservation missing active center");
            if removed {
                state.grant_unblocked(self.center)
            } else {
                Vec::new()
            }
        };

        for (center, grant) in granted {
            grant(LightWorkWindowReservation {
                gate: Arc::clone(&self.gate),
                center,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::{spawn, task::yield_now};

    use super::*;

    #[test]
    fn overlapping_windows_cannot_be_reserved_together() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let _first = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        assert!(gate.try_reserve_centered(ChunkPos::new(1, 0)).is_none());
        assert!(gate.try_reserve_centered(ChunkPos::new(4, 4)).is_none());
    }

    #[test]
    fn non_overlapping_windows_can_be_reserved_together() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let _first = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        assert!(gate.try_reserve_centered(ChunkPos::new(5, 0)).is_some());
        assert!(gate.try_reserve_centered(ChunkPos::new(0, 5)).is_some());
    }

    #[test]
    fn dropping_reservation_releases_window() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let first = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        assert!(gate.try_reserve_centered(ChunkPos::new(1, 0)).is_none());

        drop(first);
        assert!(gate.try_reserve_centered(ChunkPos::new(1, 0)).is_some());
    }

    #[test]
    fn continuation_runs_holding_the_window_once_it_frees() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let held = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        let granted = Arc::new(SyncMutex::new(None));
        gate.reserve_centered_with(ChunkPos::new(1, 0), {
            let granted = Arc::clone(&granted);
            move |reservation| *granted.lock() = Some(reservation)
        });
        assert!(
            granted.lock().is_none(),
            "continuation ran while a conflicting window was held"
        );

        drop(held);
        assert!(
            granted.lock().is_some(),
            "continuation did not run once its window came free"
        );
        // It was handed the window, not just told to go take it.
        assert!(gate.try_reserve_centered(ChunkPos::new(0, 0)).is_none());

        let reservation = granted.lock().take();
        drop(reservation);
        assert!(gate.try_reserve_centered(ChunkPos::new(0, 0)).is_some());
    }

    #[test]
    fn continuation_dropping_its_reservation_frees_the_window() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let held = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        // Stands in for a waiter that has gone away by the time it is granted:
        // the reservation is handed over and immediately dropped.
        gate.reserve_centered_with(ChunkPos::new(1, 0), drop);

        drop(held);
        assert!(gate.try_reserve_centered(ChunkPos::new(1, 0)).is_some());
    }

    #[test]
    fn a_chain_of_continuations_all_run() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let held = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        // Each waiter releases the window from inside its own grant, so only the
        // first is unblocked by `held`; the rest come off that cascade.
        let ran = Arc::new(SyncMutex::new(Vec::new()));
        for x in 1..=4 {
            let ran = Arc::clone(&ran);
            gate.reserve_centered_with(ChunkPos::new(x, 0), move |reservation| {
                ran.lock().push(x);
                drop(reservation);
            });
        }
        assert!(ran.lock().is_empty());

        drop(held);
        let mut ran = ran.lock().clone();
        ran.sort_unstable();
        assert_eq!(ran, vec![1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn released_window_is_handed_to_a_blocked_waiter() {
        let gate = Arc::new(LightWorkWindowGate::new());
        let held = gate
            .try_reserve_centered(ChunkPos::new(0, 0))
            .expect("first light window should reserve");

        let waiting = spawn({
            let gate = Arc::clone(&gate);
            async move { gate.reserve_centered(ChunkPos::new(1, 0)).await }
        });
        // Let the spawned task block on the conflicting window.
        yield_now().await;
        yield_now().await;

        drop(held);
        let granted = waiting.await.expect("waiter task should not panic");

        // The waiter owns the window now, so an overlapping one must be refused.
        assert!(gate.try_reserve_centered(ChunkPos::new(0, 0)).is_none());
        drop(granted);
        assert!(gate.try_reserve_centered(ChunkPos::new(0, 0)).is_some());
    }
}
