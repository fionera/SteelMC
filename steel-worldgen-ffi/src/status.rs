//! Status codes and the thread-local last-error slot.
//!
//! Nothing but integers and NUL-terminated bytes cross the ABI. Rust errors are
//! flattened to a [`Status`] and a human-readable message the host can fetch with
//! `swg_last_error`.

use std::cell::RefCell;
use std::ffi::CString;

/// Result of an FFI call. `Ok` is zero; every failure is negative so a host can
/// test `< 0` without knowing the full set.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The call succeeded.
    Ok = 0,
    /// A required pointer argument was null.
    NullArgument = -1,
    /// An argument was outside its valid range or failed to decode.
    InvalidArgument = -2,
    /// `swg_runtime_init` has not been called, or it failed.
    RuntimeNotInitialized = -3,
    /// `swg_runtime_init` was called more than once with a different config.
    RuntimeAlreadyInitialized = -4,
    /// The world handle is unknown, or was poisoned by an earlier panic.
    InvalidHandle = -5,
    /// The generator identifier or its config was rejected by Steel.
    GeneratorRejected = -6,
    /// The caller's output buffer was too small; the required size is reported
    /// through the out-parameter so the host can retry.
    BufferTooSmall = -7,
    /// A chunk named a block or biome this build of Steel cannot represent.
    ///
    /// Deliberately distinct from [`Status::Internal`]: Steel's own import path
    /// silently substitutes air for unknown blocks, and this is what makes the
    /// host able to refuse instead.
    UnrepresentableContent = -8,
    /// Generation panicked. The world handle is poisoned and must be closed; the
    /// host should fall back to its own generator.
    Panicked = -9,
    /// Any other failure. See `swg_last_error`.
    Internal = -100,
}

impl Status {
    /// The raw value crossing the ABI.
    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }
}

thread_local! {
    /// Last error message for the calling thread.
    ///
    /// Thread-local rather than global because generation is driven from several
    /// host worker threads concurrently; a shared slot would race and report
    /// another thread's failure.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Records `message` as this thread's last error and returns `status` for the
/// caller to propagate.
pub fn fail(status: Status, message: impl Into<String>) -> Status {
    let message = message.into();
    tracing::error!(status = status.code(), %message, "steel-worldgen-ffi call failed");

    // Interior NULs would truncate the message; replace rather than drop it.
    let sanitized = message.replace('\0', "\u{fffd}");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).ok();
    });
    status
}

/// Clears this thread's last error. Called at the top of every export so a stale
/// message from an earlier call cannot be misread as belonging to this one.
pub fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Runs `f` with this thread's last error message, or `None` if there is none.
///
/// Takes a closure rather than returning the pointer so the borrow cannot outlive
/// the `RefCell` guard.
pub fn with_last_error<R>(f: impl FnOnce(Option<&CString>) -> R) -> R {
    LAST_ERROR.with(|slot| f(slot.borrow().as_ref()))
}
