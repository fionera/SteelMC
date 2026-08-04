//! CPU topology and thread affinity.
//!
//! Chunk generation misses to cache, not to memory: over a 601x601
//! pregeneration 89.2% of fills came from the local L2, 7.0% from the local
//! CCX's L3 and only 3.8% from DRAM. The kernel meanwhile moved threads about
//! 40,000 times a second, and a move across an L3 boundary throws away both the
//! warm L2 and that CCX's 32 MiB L3 -- including its copy of the ~2.9 MB
//! climate R-tree biome lookup streams 1,536 times per chunk. These modules
//! provide the two halves needed to stop paying for that: reading where the L3
//! boundaries are, and confining a thread to one side of them.

/// Setting the calling thread's CPU affinity mask.
pub mod affinity;
/// The machine's thread count, taken before any mask is narrowed.
pub mod parallelism;
/// Reading cache-sharing domains from Linux sysfs.
pub mod topology;
