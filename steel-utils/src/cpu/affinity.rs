//! Reading and narrowing the calling thread's CPU affinity mask.

use std::io;
#[cfg(target_os = "linux")]
use std::mem;

/// How many CPU ids a `cpu_set_t` can hold.
///
/// Deliberately derived from the struct rather than from `libc::CPU_SETSIZE`.
/// `cpu_set_t` is `[u64; 16]` on every Linux target libc supports, so the mask
/// always addresses 1024 ids, but libc only reports that through `CPU_SETSIZE`
/// on glibc; for musl it is `if cfg!(musl_v1_2_3) { 1024 } else { 128 }`, and
/// that cfg is off unless the build sets `CARGO_CFG_LIBC_UNSTABLE_MUSL_V1_2_3`.
/// This repository's Dockerfile builds on Alpine, so trusting `CPU_SETSIZE`
/// would throw away every CPU id above 127 in the shipped image while the mask
/// still had room for them.
#[cfg(target_os = "linux")]
const MASK_CAPACITY: usize = 8 * size_of::<libc::cpu_set_t>();

/// The CPUs the calling thread is currently allowed to run on.
///
/// Callers need this before pinning: `sched_setaffinity` does not intersect
/// with the caller's existing mask, it replaces it, and the kernel lets an
/// unprivileged thread widen its own mask back out to the whole machine. So a
/// mask built from sysfs alone would silently escape a `taskset` or
/// `numactl --physcpubind` confinement. Under a cpuset cgroup it does not
/// escape -- the kernel intersects there -- but sysfs is not namespaced by the
/// cgroup either, so a plan built from sysfs names CPUs the cpuset forbids and
/// those threads fail with `EINVAL` while the rest pin, leaving a half-pinned
/// pool. Intersecting a plan with this first avoids both.
///
/// # Errors
///
/// Returns the `sched_getaffinity` error on failure -- notably `EINVAL` on a
/// machine with more than [`MASK_CAPACITY`] CPUs -- and an
/// [`io::ErrorKind::Unsupported`] error on non-Linux targets.
#[cfg(target_os = "linux")]
pub fn current_thread_cpus() -> io::Result<Vec<usize>> {
    // SAFETY: cpu_set_t is a plain fixed-size bit array with no invalid bit
    // patterns; all zero is the documented empty mask.
    let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
    // A pid of 0 means the calling thread, not the process.
    // SAFETY: `set` is initialized and the size passed is its own.
    let result = unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &raw mut set) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    let cpus = (0..MASK_CAPACITY)
        // SAFETY: `cpu` is below the mask's bit capacity, which is CPU_ISSET's
        // only precondition, and `set` was filled by a successful call above.
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect();
    Ok(cpus)
}

/// The CPUs the calling thread is currently allowed to run on.
///
/// # Errors
///
/// Always fails with [`io::ErrorKind::Unsupported`]: only Linux affinity is
/// implemented.
#[cfg(not(target_os = "linux"))]
pub fn current_thread_cpus() -> io::Result<Vec<usize>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "CPU affinity is only implemented on Linux",
    ))
}

/// Restricts the calling thread to `cpus`.
///
/// Deliberately a *set*, not a single CPU: the point of pinning here is cache
/// residency, and every CPU sharing an L3 gives the same residency. Handing the
/// scheduler a whole domain leaves it free to balance load and to use both SMT
/// siblings of a core, which a one-CPU mask would take away for no extra cache
/// benefit.
///
/// This replaces the mask outright and does **not** intersect with what the
/// thread was already allowed; passing a superset of the current mask widens
/// it. Callers that must respect an operator's confinement intersect with
/// [`current_thread_cpus`] themselves -- doing it in here would hide from the
/// caller that its plan was not carried out, and the caller is the one that
/// logs what it did.
///
/// A failure leaves the thread's existing mask untouched, so a caller pinning
/// many threads can treat each failure as "that thread stays where it was".
///
/// # Errors
///
/// Fails if `cpus` is empty or names any id the mask cannot represent -- a
/// partial mask is refused rather than applied, because a thread confined to a
/// silently truncated slice of its domain is worse than an unpinned one while
/// the caller's log line still claims the whole domain. Otherwise returns the
/// `sched_setaffinity` error, or an [`io::ErrorKind::Unsupported`] error on
/// non-Linux targets.
#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpus: &[usize]) -> io::Result<()> {
    if cpus.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "an empty CPU set would not be a valid affinity mask",
        ));
    }
    if let Some(&cpu) = cpus.iter().find(|&&cpu| cpu >= MASK_CAPACITY) {
        // libc's CPU_SET is a plain slice index into the mask's backing array,
        // so this would panic rather than corrupt memory -- and under this
        // workspace's `panic = "abort"` a panic inside a rayon start handler
        // takes the whole server down. Refusing beats both.
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("CPU id {cpu} is beyond the {MASK_CAPACITY} ids a cpu_set_t can hold"),
        ));
    }

    // SAFETY: cpu_set_t is a plain fixed-size bit array with no invalid bit
    // patterns; all zero is the documented empty mask.
    let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
    for &cpu in cpus {
        // SAFETY: `cpu` is below the mask's bit capacity, checked above, which
        // is CPU_SET's only precondition, and `set` is initialized.
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }

    // A pid of 0 means the calling thread, not the process: this must not touch
    // the affinity of any other thread in the server.
    // SAFETY: `set` is an initialized cpu_set_t and the size passed is its own.
    let result =
        unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &raw const set) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Restricts the calling thread to `cpus`.
///
/// # Errors
///
/// Always fails with [`io::ErrorKind::Unsupported`]: only Linux affinity is
/// implemented.
#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread(_cpus: &[usize]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "CPU affinity is only implemented on Linux",
    ))
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{MASK_CAPACITY, current_thread_cpus, pin_current_thread};

    #[test]
    fn rejects_an_empty_mask() {
        assert!(pin_current_thread(&[]).is_err());
    }

    #[test]
    fn refuses_rather_than_truncating_an_out_of_range_domain() {
        // The failure mode this guards is a *partial* pin: a domain straddling
        // the mask's capacity must not be applied as the part that fits, or the
        // thread ends up confined to a fraction of its L3 while the caller
        // reports the whole thing.
        assert!(pin_current_thread(&[MASK_CAPACITY - 1, MASK_CAPACITY]).is_err());
        assert!(pin_current_thread(&[usize::MAX]).is_err());
    }

    #[test]
    fn mask_capacity_is_the_whole_cpu_set_t() {
        // Guards against reintroducing `libc::CPU_SETSIZE`, which is 128 on a
        // musl build even though the struct addresses 1024 ids.
        assert_eq!(MASK_CAPACITY, 1024);
    }

    #[test]
    fn the_current_mask_is_readable_and_ordered() {
        let cpus = current_thread_cpus().expect("a thread always has an affinity mask on Linux");
        assert!(
            !cpus.is_empty(),
            "a runnable thread must be allowed at least one CPU"
        );
        assert!(cpus.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn pinning_to_the_current_mask_is_accepted() {
        // The one affinity change safe to make inside a test process: it asks
        // for exactly what is already in force, so no thread moves.
        let cpus = current_thread_cpus().expect("a thread always has an affinity mask on Linux");
        pin_current_thread(&cpus).expect("re-applying the existing mask should succeed");
        assert_eq!(
            current_thread_cpus().expect("the mask should still be readable"),
            cpus
        );
    }
}
