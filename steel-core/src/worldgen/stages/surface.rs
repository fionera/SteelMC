use std::sync::Arc;

use glam::IVec3;

use crate::chunk::{
    chunk_generation_task::StaticCache2D, chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep,
    status::ChunkStatus,
};
use crate::worldgen::generator::context::WorldGenContext;
use crate::worldgen::generator::{ChunkGenerator, GenerationChunk, SurfacePhase};

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
    let total_quarts_y = (chunk.section_count() * 4) as i32;

    let neighbor_biomes = |q: IVec3| -> u16 {
        let chunk_x = q.x >> 2;
        let chunk_z = q.z >> 2;
        let neighbor = cache.get(chunk_x, chunk_z);
        let neighbor_chunk = neighbor
            .try_chunk(ChunkStatus::Biomes)
            .expect("Neighbor not at Biomes status");
        let sections = neighbor_chunk.sections();
        let local_qx = (q.x - chunk_x * 4) as usize;
        let local_qz = (q.z - chunk_z * 4) as usize;
        let qy_clamped = (q.y - min_qy).clamp(0, total_quarts_y - 1) as usize;
        let section_idx = qy_clamped / 4;
        let local_qy = qy_clamped % 4;
        sections.sections[section_idx]
            .read()
            .biomes
            .get(local_qx, local_qy, local_qz)
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
    // Walked neighbour by neighbour and section by section so each section lock
    // is taken once for up to sixteen quarts, rather than once per quart as a
    // `neighbor_biomes` call per cell would.
    let pos = holder.get_pos();
    let ring_contains_any = |biomes: &[u16]| -> bool {
        if biomes.is_empty() {
            return false;
        }

        RING_BY_NEIGHBOR.iter().any(|&((chunk_dx, chunk_dz), columns)| {
            let neighbor = cache.get(pos.0.x + chunk_dx, pos.0.y + chunk_dz);
            let Some(neighbor_chunk) = neighbor.try_chunk(ChunkStatus::Biomes) else {
                // Unreadable means unprovable; report a hit so the caller keeps
                // the full rule.
                return true;
            };
            neighbor_chunk.sections().sections.iter().any(|section| {
                let guard = section.read();
                columns.iter().any(|&(local_qx, local_qz)| {
                    (0..4).any(|local_qy| {
                        biomes.contains(&guard.biomes.get(local_qx, local_qy, local_qz))
                    })
                })
            })
        })
    };

    context
        .generator
        .build_surface(chunk, &neighbor_biomes, &ring_contains_any);
}
