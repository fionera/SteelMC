use std::sync::Arc;

use crate::chunk::{
    chunk_holder::ChunkHolder, chunk_pyramid::ChunkStep, static_cache_2d::StaticCache2D,
};
use crate::worldgen::generator::context::WorldGenContext;

pub(crate) fn generate(
    _context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    _cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    holder: Arc<ChunkHolder>,
) {
    holder.upgrade_to_full();
}
