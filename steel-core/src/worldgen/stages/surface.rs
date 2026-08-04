use std::sync::Arc;

use glam::IVec3;
use steel_utils::ChunkPos;

use crate::chunk::{
    chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep, static_cache_2d::StaticCache2D,
    status::ChunkStatus,
};
use crate::worldgen::generator::context::WorldGenContext;
use crate::worldgen::generator::{ChunkGenerator, GenerationChunk, SurfacePhase};

/// Quart cells along each axis of a section: sections are 16 blocks, quarts 4.
const QUARTS_PER_SECTION_AXIS: usize = 4;
/// Biome quarts in one section.
const QUARTS_PER_SECTION: usize = QUARTS_PER_SECTION_AXIS.pow(3);
/// Chunks in the 3x3 neighbourhood a surface build can read.
const NEIGHBORHOOD_CHUNKS: usize = 9;

/// The 3x3 neighbourhood's biome quarts, copied out once per surface build.
///
/// Flat and contiguous: `[chunk][section][qy][qz][qx]`, in the same order
/// `PalettedContainer::get(qx, qy, qz)` indexes, so a lookup is one multiply and
/// add. At 24 sections this is ~27 KiB, against the 48 KiB L1D.
struct BiomeNeighborhood {
    quarts: Vec<u16>,
    /// Which of the nine were at `Biomes` and could be copied. A missing
    /// neighbour is not an error here -- `ring_contains_any` treats it as
    /// unprovable -- so it is recorded rather than raised.
    present: [bool; NEIGHBORHOOD_CHUNKS],
    section_count: usize,
}

impl BiomeNeighborhood {
    /// Index of a neighbour offset in `-1..=1` on both axes.
    #[inline]
    const fn slot(chunk_dx: i32, chunk_dz: i32) -> Option<usize> {
        if chunk_dx < -1 || chunk_dx > 1 || chunk_dz < -1 || chunk_dz > 1 {
            return None;
        }
        #[expect(
            clippy::cast_sign_loss,
            reason = "both offsets are bounds-checked to -1..=1 immediately above"
        )]
        Some(((chunk_dz + 1) * 3 + (chunk_dx + 1)) as usize)
    }

    fn snapshot(
        cache: &StaticCache2D<Arc<ChunkHolder>>,
        pos: ChunkPos,
        section_count: usize,
    ) -> Self {
        let mut quarts = vec![0u16; NEIGHBORHOOD_CHUNKS * section_count * QUARTS_PER_SECTION];
        let mut present = [false; NEIGHBORHOOD_CHUNKS];

        for chunk_dz in -1..=1 {
            for chunk_dx in -1..=1 {
                let slot = Self::slot(chunk_dx, chunk_dz).expect("offsets are within -1..=1");
                let neighbor = cache.get(pos.0.x + chunk_dx, pos.0.y + chunk_dz);
                let Some(neighbor_chunk) = neighbor.try_chunk(ChunkStatus::Biomes) else {
                    continue;
                };
                present[slot] = true;

                let sections = neighbor_chunk.sections();
                for (section_idx, section) in
                    sections.sections.iter().take(section_count).enumerate()
                {
                    let guard = section.read();
                    let base = (slot * section_count + section_idx) * QUARTS_PER_SECTION;
                    for local_qy in 0..QUARTS_PER_SECTION_AXIS {
                        for local_qz in 0..QUARTS_PER_SECTION_AXIS {
                            for local_qx in 0..QUARTS_PER_SECTION_AXIS {
                                let offset = (local_qy * QUARTS_PER_SECTION_AXIS + local_qz)
                                    * QUARTS_PER_SECTION_AXIS
                                    + local_qx;
                                quarts[base + offset] = guard.biomes.get(local_qx, local_qy, local_qz);
                            }
                        }
                    }
                }
            }
        }

        Self {
            quarts,
            present,
            section_count,
        }
    }

    /// The biome at a quart, or `None` if that neighbour was not readable.
    #[inline]
    fn get(
        &self,
        chunk_dx: i32,
        chunk_dz: i32,
        section_idx: usize,
        local_qx: usize,
        local_qy: usize,
        local_qz: usize,
    ) -> Option<u16> {
        let slot = Self::slot(chunk_dx, chunk_dz)?;
        if !self.present[slot] || section_idx >= self.section_count {
            return None;
        }
        let base = (slot * self.section_count + section_idx) * QUARTS_PER_SECTION;
        let offset = (local_qy * QUARTS_PER_SECTION_AXIS + local_qz) * QUARTS_PER_SECTION_AXIS
            + local_qx;
        Some(self.quarts[base + offset])
    }
}

/// The one-quart ring around a chunk, split by the neighbour each column is in.
///
/// Fixed geometry: quart offsets from the chunk's own origin run `-1..=4` on
/// both axes, and `0..=3` is the chunk itself, so each side neighbour
/// contributes its facing edge of four quarts and each corner neighbour a single
/// quart -- twenty in all. Written out rather than derived so the walk below is
/// a straight iteration.
type RingNeighbor = ((i32, i32), &'static [(usize, usize)]);
const RING_BY_NEIGHBOR: [RingNeighbor; 8] = [
    ((-1, -1), &[(3, 3)]),
    ((-1, 0), &[(3, 0), (3, 1), (3, 2), (3, 3)]),
    ((-1, 1), &[(3, 0)]),
    ((0, -1), &[(0, 3), (1, 3), (2, 3), (3, 3)]),
    ((0, 1), &[(0, 0), (1, 0), (2, 0), (3, 0)]),
    ((1, -1), &[(0, 3)]),
    ((1, 0), &[(0, 0), (0, 1), (0, 2), (0, 3)]),
    ((1, 1), &[(0, 0)]),
];

#[expect(
    clippy::similar_names,
    reason = "quart coordinate names intentionally mirror x/y/z axes"
)]
pub(crate) fn generate(
    context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let chunk = GenerationChunk::<SurfacePhase>::acquire(&holder);

    let min_qy = chunk.min_y() >> 2;
    let section_count = chunk.section_count();
    let total_quarts_y = (section_count * 4) as i32;
    let pos = holder.get_pos();

    // Both lookups below read the same thing: the biome quarts of the 3x3
    // neighbourhood. Read straight from the neighbours, that is one `read()` per
    // quart in `neighbor_biomes` -- once per surface column, hundreds per chunk
    // -- plus 8 x `section_count` more each time `ring_contains_any` runs. A
    // `SyncRwLock` read is a read-modify-write, so each of those is an atomic
    // *write* to a line inside a neighbour's `Sections` box, contended with
    // every other worker touching that neighbour. The payload behind all of it
    // is only ~27 KiB.
    //
    // So take it once. The neighbours are at `Biomes` by the dependency contract
    // and biomes are fixed from that step onward -- this stage writes block
    // states, not biomes -- so a snapshot taken here cannot go stale under the
    // closures. Copying also resolves the palette indirection once per quart
    // instead of once per lookup, and leaves the data contiguous rather than
    // scattered across nine other chunks' allocations.
    let neighborhood = BiomeNeighborhood::snapshot(cache, pos, section_count);

    let neighbor_biomes = |q: IVec3| -> u16 {
        let chunk_x = q.x >> 2;
        let chunk_z = q.z >> 2;
        let local_qx = (q.x - chunk_x * 4) as usize;
        let local_qz = (q.z - chunk_z * 4) as usize;
        let qy_clamped = (q.y - min_qy).clamp(0, total_quarts_y - 1) as usize;
        let section_idx = qy_clamped / 4;
        let local_qy = qy_clamped % 4;
        neighborhood
            .get(
                chunk_x - pos.0.x,
                chunk_z - pos.0.y,
                section_idx,
                local_qx,
                local_qy,
                local_qz,
            )
            .expect("Neighbor not at Biomes status")
    };

    // Whether any of `biomes` appears in the one-quart ring around this chunk.
    //
    // A fuzzed biome lookup can only return one of the eight quart cells at
    // `{parent_x, +1} x {parent_y, +1} x {parent_z, +1}`, and across a chunk's
    // columns those reach exactly one quart beyond the chunk on each side. So a
    // biome absent from both this chunk and that ring cannot be returned
    // anywhere in the chunk, which is what lets the surface rule drop the
    // branches testing for it. The chunk's own biomes are checked by the caller,
    // which has already read them.
    //
    // Served from the same snapshot, so this walk takes no locks at all.
    let ring_contains_any = |biomes: &[u16]| -> bool {
        if biomes.is_empty() {
            return false;
        }

        RING_BY_NEIGHBOR.iter().any(|&((chunk_dx, chunk_dz), columns)| {
            (0..section_count).any(|section_idx| {
                columns.iter().any(|&(local_qx, local_qz)| {
                    (0..4).any(|local_qy| {
                        // Unreadable means unprovable; report a hit so the
                        // caller keeps the full rule. Matches the pre-snapshot
                        // behaviour, which returned `true` for such a neighbour:
                        // either way the result is
                        // `any_unreadable || any_hit`.
                        neighborhood
                            .get(chunk_dx, chunk_dz, section_idx, local_qx, local_qy, local_qz)
                            .is_none_or(|biome| biomes.contains(&biome))
                    })
                })
            })
        })
    };

    context
        .generator
        .build_surface(chunk, &neighbor_biomes, &ring_contains_any);
}
