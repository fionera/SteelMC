//! The machine's thread count, read once before this process pins anything.
//!
//! `std::thread::available_parallelism` reports the *affinity mask*, not the
//! machine: on this box it returns 128 normally and 16 under
//! `taskset -c 56-63,120-127`. Every thread pool here is sized from it, and
//! [`crate::cpu::affinity`] exists to narrow masks, so the two combine badly.
//! Measured: with one L3 domain reserved, the gameplay packet pool -- sized
//! from a thread the reservation had pinned -- came up with 8 workers against
//! 64 in the unreserved arm, because `(16 / 2).max(2)` is 8. That is a
//! different server, not a differently-scheduled one, and it silently
//! invalidates the A/B the reservation was built for.
//!
//! So the reading is taken once, while the process still has whatever mask the
//! operator gave it, and every later sizing decision uses that snapshot. An
//! operator's own `taskset` is still honoured -- it is in force before `main`
//! runs -- which is the distinction that matters: this ignores the masks *the
//! server itself* installs, not the ones it was started with.

use std::num::NonZero;
use std::sync::OnceLock;
use std::thread;

/// What to assume when the platform will not say. Matches what every call site
/// used before this module existed.
const FALLBACK: usize = 4;

static MACHINE: OnceLock<usize> = OnceLock::new();

/// Hardware parallelism as of the first call in this process.
///
/// Call [`snapshot`] from `main` first; this is only mask-independent from the
/// point of the first call onwards.
#[must_use]
pub fn machine_parallelism() -> usize {
    *MACHINE.get_or_init(|| thread::available_parallelism().map_or(FALLBACK, NonZero::get))
}

/// Takes the reading now, so that later calls cannot see a mask this process
/// installed on itself.
///
/// Belongs at the top of `main`, before any runtime or pool is built.
pub fn snapshot() {
    let _ = machine_parallelism();
}

#[cfg(test)]
mod tests {
    use super::machine_parallelism;

    #[test]
    fn the_reading_is_stable_and_positive() {
        let first = machine_parallelism();
        assert!(first > 0);
        assert_eq!(machine_parallelism(), first);
    }

    /// The reason this module exists: a thread the server pinned must still
    /// size pools for the machine. Reading `available_parallelism` directly
    /// here would return the pinned thread's CPU count instead.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_pinned_thread_still_sees_the_whole_machine() {
        use super::super::affinity::{current_thread_cpus, pin_current_thread};
        use std::thread;

        // Taken on this unpinned thread, as `main` takes it.
        let machine = machine_parallelism();
        let allowed = current_thread_cpus().expect("this thread has an affinity mask");
        if allowed.len() < 2 {
            // Nothing to narrow to; the assertion below could not discriminate.
            return;
        }
        let one_cpu = allowed[..1].to_vec();

        // In its own thread, which then exits: pinning is per-thread, and the
        // test harness's other threads must not inherit a one-CPU mask.
        let seen = thread::spawn(move || {
            pin_current_thread(&one_cpu).expect("pinning to an allowed CPU should succeed");
            (
                machine_parallelism(),
                thread::available_parallelism().map_or(0, std::num::NonZero::get),
            )
        })
        .join()
        .expect("the pinned thread should not panic");

        assert_eq!(seen.0, machine, "the snapshot followed the affinity mask");
        assert_eq!(seen.1, 1, "the direct call should see only the pinned CPU");
    }
}
