//! World handle table.
//!
//! The host holds an opaque `u64`, never a pointer. That keeps the ABI free of
//! lifetime questions and makes a stale handle a clean `InvalidHandle` rather
//! than a use-after-free.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use steel_utils::locks::SyncRwLock;

use crate::engine::GenerationWorld;

/// A registered world plus whether it is still usable.
struct Entry {
    world: Arc<GenerationWorld>,
    /// Set when generation panicked inside this world.
    ///
    /// Steel uses panics as its dependency and write-radius contract, so a
    /// caught panic means invariants were already violated; continuing would
    /// generate silently wrong terrain. The handle is retired and the host is
    /// told to fall back.
    poisoned: bool,
}

/// Process-wide handle table.
static WORLDS: SyncRwLock<Option<FxHashMap<u64, Entry>>> = SyncRwLock::new(None);

/// Next handle to hand out. Starts at 1 so 0 is always invalid.
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

/// Registers `world` and returns its handle.
pub fn insert(world: GenerationWorld) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    let mut guard = WORLDS.write();
    guard.get_or_insert_with(FxHashMap::default).insert(
        handle,
        Entry {
            world: Arc::new(world),
            poisoned: false,
        },
    );
    handle
}

/// Looks up a usable world.
///
/// Returns `None` if the handle is unknown or poisoned. Clones the `Arc` so the
/// table lock is not held while generation runs.
pub fn get(handle: u64) -> Option<Arc<GenerationWorld>> {
    let guard = WORLDS.read();
    let entry = guard.as_ref()?.get(&handle)?;
    if entry.poisoned {
        return None;
    }
    Some(entry.world.clone())
}

/// Marks `handle` unusable. Subsequent lookups fail until it is closed.
pub fn poison(handle: u64) {
    let mut guard = WORLDS.write();
    if let Some(entry) = guard.as_mut().and_then(|worlds| worlds.get_mut(&handle)) {
        entry.poisoned = true;
    }
}

/// Removes `handle`, returning whether it existed.
pub fn remove(handle: u64) -> bool {
    let mut guard = WORLDS.write();
    guard
        .as_mut()
        .is_some_and(|worlds| worlds.remove(&handle).is_some())
}
