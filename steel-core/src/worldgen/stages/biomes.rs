use std::sync::Arc;

use crate::chunk::{
    chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep, static_cache_2d::StaticCache2D,
    status::ChunkStatus,
};
use crate::worldgen::generator::ChunkGenerator;
use crate::worldgen::generator::context::WorldGenContext;

pub(crate) fn generate(
    context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    _cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    let chunk = holder
        .try_chunk(ChunkStatus::StructureReferences)
        .expect("Chunk not found at status StructureReferences");

    context.generator.create_biomes(chunk);
}
