//! C ABI exposing `SteelMC`'s world generation to a host process.
//!
//! Intended to be loaded into a JVM over Project Panama's Foreign Function &
//! Memory API and driven by a Fabric/NeoForge mod, but the ABI is plain C and
//! assumes nothing about the host.
//!
//! # Contract
//!
//! - Every export returns an [`Status`] code; negative means failure, and
//!   [`swg_last_error`] returns a message for the calling thread.
//! - Nothing but integers, NUL-terminated bytes and caller-owned buffers cross
//!   the boundary. No Rust types, slices, enums or trait objects.
//! - Every export catches unwinds. This only works because the `ffi-release`
//!   profile sets `panic = "unwind"`; the workspace `release` profile sets
//!   `panic = "abort"`, which would strip the landing pads and abort the host
//!   process instead.
//! - Generation runs on this crate's own thread pool, never on a caller thread.
//!   Callers block for the duration.
//!
//! # Safety
//!
//! Exports are `unsafe` because they dereference caller-supplied pointers. Each
//! documents what it requires.

#![allow(
    unsafe_code,
    reason = "this crate exists to expose a C ABI across an FFI boundary"
)]

pub mod engine;
pub mod handle;
pub mod snapshot;
pub mod status;

use std::any::Any;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

use steel_core::chunk::chunk_access::ChunkStatus;
use steel_utils::{ChunkPos, Identifier};

use crate::engine::{GenerationWorld, WorldSpec};
use crate::status::{Status, clear_last_error, fail, with_last_error};

/// ABI version. Bumped on any incompatible change to a signature or struct
/// layout below; the host must refuse to load a library it does not recognise.
pub const ABI_VERSION: u32 = 1;

/// A chunk position, as it crosses the ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwgChunkPos {
    /// Chunk X coordinate.
    pub x: i32,
    /// Chunk Z coordinate.
    pub z: i32,
}

/// Configuration for [`swg_world_open`].
#[repr(C)]
#[derive(Debug)]
pub struct SwgWorldConfig {
    /// NUL-terminated generator identifier, e.g. `minecraft:overworld`.
    pub generator: *const c_char,
    /// World seed.
    pub seed: i64,
    /// Generation worker threads; 0 means one per available core.
    pub threads: u32,
}

/// Returns the ABI version this library implements.
#[unsafe(no_mangle)]
pub const extern "C" fn swg_abi_version() -> u32 {
    ABI_VERSION
}

/// Initializes Steel's process-global registries. Idempotent and safe to call
/// from any thread; only the first call does work.
///
/// Must succeed before [`swg_world_open`].
#[unsafe(no_mangle)]
pub extern "C" fn swg_runtime_init() -> i32 {
    guard(|| match engine::init_runtime() {
        Ok(()) => Status::Ok,
        Err(err) => fail(Status::RuntimeNotInitialized, err),
    })
}

/// Opens a generation world and writes its handle to `out_handle`.
///
/// # Safety
/// `config` must point to a valid [`SwgWorldConfig`] whose `generator` field is
/// a NUL-terminated UTF-8 string. `out_handle` must point to writable storage
/// for one `u64`. Both are only read/written during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn swg_world_open(
    config: *const SwgWorldConfig,
    out_handle: *mut u64,
) -> i32 {
    guard(|| {
        if config.is_null() || out_handle.is_null() {
            return fail(
                Status::NullArgument,
                "config and out_handle must be non-null",
            );
        }

        // SAFETY: checked non-null above; the caller contract requires a valid,
        // initialized `SwgWorldConfig` for the duration of the call.
        let config = unsafe { &*config };

        // SAFETY: `config.generator` comes from the caller, who is required to
        // supply a NUL-terminated string valid for this call.
        let generator = match unsafe { cstr_to_str(config.generator, "generator") } {
            Ok(value) => value,
            Err(status) => return status,
        };
        let generator = match generator.parse::<Identifier>() {
            Ok(value) => value,
            Err(err) => {
                return fail(
                    Status::InvalidArgument,
                    format!("generator {generator:?} is not a valid identifier: {err}"),
                );
            }
        };

        let spec = WorldSpec {
            generator,
            seed: config.seed,
            threads: config.threads as usize,
        };

        match GenerationWorld::open(&spec) {
            Ok(world) => {
                let handle = handle::insert(world);
                // SAFETY: checked non-null above; caller guarantees writable
                // storage for one u64.
                unsafe { out_handle.write(handle) };
                Status::Ok
            }
            Err(err) => fail(Status::GeneratorRejected, err),
        }
    })
}

/// Generates `count` chunks to `target_status` and writes a snapshot of them to
/// `out`.
///
/// Blocks the calling thread while this crate's own pool does the work.
/// Generation must never run on a host thread: the transpiled density functions
/// need far more stack than a typical JVM thread has.
///
/// The required size is always written to `out_len`, so passing a null or short
/// buffer is the way to size one: the call then returns
/// [`Status::BufferTooSmall`] and the chunks stay resident, so retrying with a
/// large enough buffer does not regenerate them.
///
/// # Safety
/// `positions` must point to `count` readable [`SwgChunkPos`] values. `out` must
/// be null or point to `out_capacity` writable bytes. `out_len` must point to
/// writable storage for one `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn swg_generate_batch(
    world: u64,
    positions: *const SwgChunkPos,
    count: usize,
    target_status: u32,
    out: *mut u8,
    out_capacity: usize,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        if positions.is_null() || out_len.is_null() {
            return fail(
                Status::NullArgument,
                "positions and out_len must be non-null",
            );
        }
        if count == 0 {
            return fail(Status::InvalidArgument, "count must be non-zero");
        }

        let Some(target) = ChunkStatus::from_index(target_status as usize) else {
            return fail(
                Status::InvalidArgument,
                format!("unknown chunk status {target_status}"),
            );
        };

        let Some(generation_world) = handle::get(world) else {
            return fail(
                Status::InvalidHandle,
                format!("world handle {world} is unknown or poisoned"),
            );
        };

        // SAFETY: checked non-null above; caller guarantees `count` readable
        // elements, valid for the duration of this call.
        let positions = unsafe { slice::from_raw_parts(positions, count) };
        let centers: Vec<ChunkPos> = positions
            .iter()
            .map(|pos| ChunkPos::new(pos.x, pos.z))
            .collect();

        // A panic here means Steel's own invariants were violated, so the world
        // is retired rather than reused. Caught separately from `guard` so the
        // handle can be poisoned.
        let encoded = catch_unwind(AssertUnwindSafe(|| {
            generation_world.generate_snapshot(&centers, target)
        }));

        let bytes = match encoded {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(err)) => {
                // An unrepresentable block or biome is the host's cue to keep the
                // chunk itself rather than accept a lossy substitute, so it gets
                // its own status.
                let status = if err.contains("no registry entry") {
                    Status::UnrepresentableContent
                } else {
                    Status::Internal
                };
                return fail(status, err);
            }
            Err(payload) => {
                handle::poison(world);
                return fail(
                    Status::Panicked,
                    format!(
                        "generation panicked, world {world} retired: {}",
                        panic_message(&payload)
                    ),
                );
            }
        };

        // SAFETY: checked non-null above; caller guarantees writable storage for
        // one usize.
        unsafe { out_len.write(bytes.len()) };

        if out.is_null() || out_capacity < bytes.len() {
            return fail(
                Status::BufferTooSmall,
                format!(
                    "snapshot needs {} bytes, buffer holds {out_capacity}",
                    bytes.len()
                ),
            );
        }

        // SAFETY: `out` is non-null with at least `bytes.len()` capacity, checked
        // immediately above. Source and destination cannot overlap: `bytes` is a
        // fresh Rust allocation.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
        Status::Ok
    })
}

/// Closes a world handle and releases its threads and chunks.
///
/// Closing an unknown handle is an error, not a no-op, so double-frees surface.
#[unsafe(no_mangle)]
pub extern "C" fn swg_world_close(world: u64) -> i32 {
    guard(|| {
        if handle::remove(world) {
            Status::Ok
        } else {
            fail(
                Status::InvalidHandle,
                format!("world handle {world} is unknown"),
            )
        }
    })
}

/// Returns this thread's last error message, or null if there is none.
///
/// # Safety
/// The returned pointer is owned by this library and valid until the next call
/// on the same thread. Callers must copy it before making another call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn swg_last_error() -> *const c_char {
    with_last_error(|message| message.map_or(ptr::null(), |value| value.as_ptr()))
}

/// Wraps an export body: clears the stale error, catches unwinds, returns a code.
///
/// A panic escaping into the host is undefined behaviour, so this is the outer
/// net for every export. Panics that carry meaning for a specific world handle
/// are caught closer to the call site so the handle can be poisoned first.
fn guard(body: impl FnOnce() -> Status) -> i32 {
    clear_last_error();

    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status.code(),
        Err(payload) => fail(
            Status::Panicked,
            format!(
                "panic crossed the FFI boundary: {}",
                panic_message(&payload)
            ),
        )
        .code(),
    }
}

/// Best-effort text from a panic payload.
fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "<non-string panic payload>".to_owned())
        },
        |message| (*message).to_owned(),
    )
}

/// Borrows a NUL-terminated UTF-8 string from the caller.
///
/// # Safety
/// `pointer` must be null or point to a NUL-terminated byte string that stays
/// valid for the returned reference's lifetime.
unsafe fn cstr_to_str<'a>(pointer: *const c_char, field: &str) -> Result<&'a str, Status> {
    if pointer.is_null() {
        return Err(fail(
            Status::NullArgument,
            format!("{field} must be non-null"),
        ));
    }

    // SAFETY: checked non-null above; caller guarantees NUL termination.
    unsafe { CStr::from_ptr(pointer) }.to_str().map_err(|err| {
        fail(
            Status::InvalidArgument,
            format!("{field} is not UTF-8: {err}"),
        )
    })
}
