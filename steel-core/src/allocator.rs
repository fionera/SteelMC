//! Allocator tuning.

use std::env;
use std::ffi::c_long;

/// mimalloc's `mi_option_purge_delay`.
///
/// `libmimalloc-sys` binds the option enum only partially and its constants are
/// stale relative to the mimalloc it vendors, so the value is taken from the
/// vendored headers instead. Both vendored versions agree on it: counting
/// `mi_option_e` in `v2/include/mimalloc.h` and `v3/include/mimalloc.h` puts
/// `purge_delay` at 15 in each, the v3 renames being confined to entries that
/// keep their slot as `deprecated_*`.
///
/// [`tune_for_throughput`] verifies the constant against the documented default
/// before using it, so a version bump that did reshuffle the enum turns into a
/// skipped tuning rather than a wrong option being set.
const MI_OPTION_PURGE_DELAY: i32 = 15;

/// The purge delays mimalloc ships as defaults, in milliseconds.
///
/// v2 uses 10 and v3 uses 1000; `libmimalloc-sys` builds v3 unless its `v2`
/// feature is on, and either is a valid thing to find here.
const MI_DEFAULT_PURGE_DELAYS: [c_long; 2] = [10, 1000];

/// Stops mimalloc returning freed memory to the OS.
///
/// Generation allocates and frees continuously -- a 601x601 pregeneration runs
/// at roughly 239,000 minor page faults per second -- and mimalloc's default
/// ten-millisecond purge delay means much of that memory is decommitted and
/// then immediately faulted back in. The cost lands in the kernel: a
/// machine-wide profile of a 601x601 run attributed 2.5% of cycles to
/// `native_queued_spin_lock_slowpath` and a further 1.3% to TLB shootdowns
/// (`native_flush_tlb_one_user`, `flush_tlb_func`), which is the mmap lock and
/// the IPIs behind those decommits.
///
/// Disabling the purge is worth +2.6% at 601x601 (9,407 -> 9,654 chunks/s over
/// five runs each, with no overlap between the two ranges) and +2.0% at
/// 301x301, for +208 MiB of peak RSS on an 8.4 GiB run -- the process keeps its
/// high-water mark instead of giving pages back and taking them again.
///
/// Must be called before the allocator does significant work. Setting
/// `MIMALLOC_PURGE_DELAY` in the environment is *not* equivalent: mimalloc
/// caches its options on first use, which happens before `main`, so the
/// variable only takes effect when it is already set at process start. This
/// goes through `mi_option_set`, which applies whenever it is called.
///
/// Respects `MIMALLOC_PURGE_DELAY` when the operator has set it, and does
/// nothing if the option does not look like the one this was measured against.
pub fn tune_for_throughput() {
    if env::var_os("MIMALLOC_PURGE_DELAY").is_some() {
        return;
    }

    // SAFETY: `mi_option_get`/`mi_option_set` take a plain option index and an
    // integer value, and are safe to call from any thread at any point after the
    // allocator exists -- which it does, since it served this program's startup.
    unsafe {
        let current = libmimalloc_sys::mi_option_get(MI_OPTION_PURGE_DELAY);
        if !MI_DEFAULT_PURGE_DELAYS.contains(&current) {
            // Not the option this was measured against; leave the allocator alone.
            return;
        }
        libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, -1);
    }
}
