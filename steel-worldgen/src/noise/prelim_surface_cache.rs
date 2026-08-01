//! Cross-chunk memo for preliminary surface levels.

use rustc_hash::FxHashMap;
use steel_utils::locks::SyncRwLock;

/// Number of independent shards. Sized so that the lookup rate a pregeneration
/// produces -- on the order of a million per second across the whole machine --
/// lands as a few thousand per second on any one lock.
const SHARDS: usize = 256;

/// Entries a shard keeps before it is cleared.
///
/// The cache exists to serve the generation front, not to remember the world:
/// what matters is holding the columns neighbouring chunks are asking for right
/// now. Clearing a full shard outright rather than evicting entry by entry keeps
/// the hot path free of bookkeeping, and the front refills it within a few
/// chunks. `SHARDS * SHARD_CAPACITY` entries is roughly 50 MB at worst.
const SHARD_CAPACITY: usize = 4096;

/// One shard's table of quart columns to surface levels.
type Shard = SyncRwLock<FxHashMap<(i32, i32), i32>>;

/// Memoizes `preliminary_surface_level` across the chunks of one world.
///
/// The level is a pure function of the quart column and the world seed, so a
/// memo is exact rather than approximate. It is worth having because a chunk's
/// aquifer has to scan a 42-block span on each axis to find its fluid-sampling
/// threshold -- 121 quart columns against the 16 that the chunk actually
/// contains -- and that span overlaps its neighbours', so the same column is
/// computed about seven times over.
///
/// Sharded rather than global, because the last shared structure on this path
/// that took one lock became the throughput ceiling for the entire server.
#[derive(Debug, Default)]
pub struct PrelimSurfaceCache {
    shards: Box<[Shard]>,
}

impl PrelimSurfaceCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| SyncRwLock::new(FxHashMap::default()))
                .collect(),
        }
    }

    /// Which shard owns a quart column.
    ///
    /// Mixes both axes so that a row of columns spreads across shards instead of
    /// queueing on one.
    #[inline]
    const fn shard_of(quart_x: i32, quart_z: i32) -> usize {
        let key = (quart_x as u32 as u64) | ((quart_z as u32 as u64) << 32);
        // splitmix64 finalizer
        let mut hash = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        hash ^= hash >> 30;
        hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 27;
        (hash as usize) % SHARDS
    }

    /// The memoized level for a quart column, if it is known.
    #[inline]
    #[must_use]
    pub fn get(&self, quart_x: i32, quart_z: i32) -> Option<i32> {
        self.shards[Self::shard_of(quart_x, quart_z)]
            .read()
            .get(&(quart_x, quart_z))
            .copied()
    }

    /// Records the level for a quart column.
    #[inline]
    pub fn insert(&self, quart_x: i32, quart_z: i32, level: i32) {
        let mut shard = self.shards[Self::shard_of(quart_x, quart_z)].write();
        if shard.len() >= SHARD_CAPACITY {
            shard.clear();
        }
        shard.insert((quart_x, quart_z), level);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn a_recorded_level_reads_back() {
        let cache = PrelimSurfaceCache::new();
        assert_eq!(cache.get(4, -8), None);
        cache.insert(4, -8, 71);
        assert_eq!(cache.get(4, -8), Some(71));
        // A different column is a miss, not a neighbour's value.
        assert_eq!(cache.get(8, -8), None);
        assert_eq!(cache.get(4, -4), None);
    }

    #[test]
    fn shards_spread_a_row_of_columns() {
        let used: BTreeSet<usize> = (0..64)
            .map(|x| PrelimSurfaceCache::shard_of(x * 4, 0))
            .collect();
        assert!(
            used.len() > 32,
            "a row of 64 columns should not pile onto {} shards",
            used.len()
        );
    }

    #[test]
    fn a_full_shard_is_cleared_rather_than_grown() {
        let cache = PrelimSurfaceCache::new();
        // Fill well past one shard's capacity and confirm nothing is unbounded.
        for i in 0..(SHARDS * SHARD_CAPACITY * 2) as i32 {
            cache.insert(i, 0, i);
        }
        let total: usize = cache.shards.iter().map(|shard| shard.read().len()).sum();
        assert!(
            total <= SHARDS * SHARD_CAPACITY,
            "cache grew past its bound: {total} entries"
        );
    }
}
