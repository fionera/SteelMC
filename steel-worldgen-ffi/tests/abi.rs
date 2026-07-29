//! Exercises the C ABI the way a host will: opaque handles, status codes, and
//! `swg_last_error` for detail.

use std::ffi::{CStr, CString};
use std::ptr;

use steel_worldgen_ffi::status::Status;
use steel_worldgen_ffi::{
    ABI_VERSION, SwgChunkPos, SwgWorldConfig, swg_abi_version, swg_generate_batch, swg_last_error,
    swg_runtime_init, swg_world_close, swg_world_open,
};

/// This thread's last error message, or the empty string.
fn last_error() -> String {
    // SAFETY: the pointer is owned by the library and valid until this thread's
    // next call; it is copied before returning.
    unsafe {
        let pointer = swg_last_error();
        if pointer.is_null() {
            String::new()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    }
}

/// Opens a world, returning its handle.
fn open(generator: &str, seed: i64) -> Result<u64, i32> {
    let generator = CString::new(generator).expect("generator name has no interior NUL");
    let config = SwgWorldConfig {
        generator: generator.as_ptr(),
        seed,
        threads: 2,
    };

    let mut handle = 0_u64;
    // SAFETY: `config` and `handle` are live for the duration of the call, and
    // `generator` outlives `config`.
    let status = unsafe { swg_world_open(&raw const config, &raw mut handle) };
    if status == Status::Ok.code() {
        Ok(handle)
    } else {
        Err(status)
    }
}

#[test]
fn reports_its_abi_version() {
    assert_eq!(swg_abi_version(), ABI_VERSION);
}

#[test]
fn runtime_init_is_idempotent() {
    assert_eq!(swg_runtime_init(), Status::Ok.code());
    assert_eq!(
        swg_runtime_init(),
        Status::Ok.code(),
        "a second init must not re-run the global tables"
    );
}

#[test]
fn rejects_null_arguments() {
    let mut handle = 0_u64;
    // SAFETY: passing null is exactly what this checks; the export is documented
    // to detect it before dereferencing.
    let status = unsafe { swg_world_open(ptr::null(), &raw mut handle) };

    assert_eq!(status, Status::NullArgument.code());
    assert!(
        last_error().contains("non-null"),
        "expected a message naming the problem, got {:?}",
        last_error()
    );
}

#[test]
fn rejects_an_unknown_generator() {
    let status = open("minecraft:not_a_generator", 1).expect_err("should be rejected");

    assert_eq!(status, Status::GeneratorRejected.code());
    assert!(
        last_error().contains("not_a_generator"),
        "error should name the generator, got {:?}",
        last_error()
    );
}

#[test]
fn rejects_a_malformed_identifier() {
    let status = open("not a valid identifier", 1).expect_err("should be rejected");
    assert_eq!(status, Status::InvalidArgument.code());
}

/// Chunk status index for `Features`.
const FEATURES: u32 = 7;

/// Generates into a caller-owned buffer, returning `(status, required_len)`.
fn generate(
    handle: u64,
    positions: &[SwgChunkPos],
    status: u32,
    buffer: &mut [u8],
) -> (i32, usize) {
    let mut needed = 0_usize;
    let out = if buffer.is_empty() {
        ptr::null_mut()
    } else {
        buffer.as_mut_ptr()
    };
    // SAFETY: `positions` and `buffer` are live with the stated lengths, and
    // `needed` is writable storage for one usize.
    let code = unsafe {
        swg_generate_batch(
            handle,
            positions.as_ptr(),
            positions.len(),
            status,
            out,
            buffer.len(),
            &raw mut needed,
        )
    };
    (code, needed)
}

#[test]
fn rejects_an_unknown_handle() {
    let positions = [SwgChunkPos { x: 0, z: 0 }];
    let mut buffer = vec![0_u8; 1 << 20];
    let (status, _) = generate(999_999, &positions, FEATURES, &mut buffer);

    assert_eq!(status, Status::InvalidHandle.code());
}

#[test]
fn rejects_an_unknown_chunk_status() {
    let handle = open("minecraft:overworld", 5).expect("overworld should open");
    let positions = [SwgChunkPos { x: 0, z: 0 }];
    let mut buffer = vec![0_u8; 1 << 20];

    let (status, _) = generate(handle, &positions, 999, &mut buffer);

    assert_eq!(status, Status::InvalidArgument.code());
    assert_eq!(swg_world_close(handle), Status::Ok.code());
}

#[test]
fn reports_the_required_buffer_size() {
    let handle = open("minecraft:overworld", 606).expect("overworld should open");
    let positions = [SwgChunkPos { x: 0, z: 0 }];

    // A null buffer is the documented way to ask how much space is needed.
    let (status, needed) = generate(handle, &positions, FEATURES, &mut []);
    assert_eq!(status, Status::BufferTooSmall.code());
    assert!(needed > 0, "required size should be reported");

    // Retrying with exactly that much must succeed.
    let mut buffer = vec![0_u8; needed];
    let (status, again) = generate(handle, &positions, FEATURES, &mut buffer);
    assert_eq!(status, Status::Ok.code(), "last error: {}", last_error());
    assert_eq!(again, needed, "size should be stable across calls");

    assert_eq!(swg_world_close(handle), Status::Ok.code());
}

#[test]
fn generates_and_closes_through_the_abi() {
    let handle = open("minecraft:overworld", 31337).expect("overworld should open");

    let positions = [SwgChunkPos { x: 0, z: 0 }, SwgChunkPos { x: 1, z: 0 }];
    let mut buffer = vec![0_u8; 8 << 20];
    let (status, length) = generate(handle, &positions, FEATURES, &mut buffer);
    assert_eq!(status, Status::Ok.code(), "last error: {}", last_error());
    assert!(length > 0, "snapshot should not be empty");

    // Magic is "SWGS".
    assert_eq!(
        &buffer[..4],
        b"SWGS",
        "snapshot should start with its magic"
    );

    assert_eq!(swg_world_close(handle), Status::Ok.code());
    assert_eq!(
        swg_world_close(handle),
        Status::InvalidHandle.code(),
        "closing twice must surface, not silently succeed"
    );
}
