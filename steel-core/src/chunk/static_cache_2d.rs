//! A pre-filled square cache of chunk neighbours, and the resolver that fills it.
//!
//! Extracted from `chunk_generation_task` so that halo resolution has a home
//! that does not depend on the generation-task model: a per-holder drive
//! resolves a halo per fused run rather than one per task, and needs the same
//! machinery without the task around it.

use std::sync::Arc;

use steel_utils::ChunkPos;

use crate::chunk::{chunk_holder::ChunkHolder, chunk_map::ChunkMap};

/// A pre-filled 2D cache of elements, efficient for async creation.
pub struct StaticCache2D<T> {
    min_x: i32,
    min_z: i32,
    size: i32,
    /// Cache stored in row-major order (Z-then-X).
    cache: Vec<T>,
}

impl<T> StaticCache2D<T> {
    /// Creates a `StaticCache2D` by populating it via a factory.
    pub fn create<F>(center_x: i32, center_z: i32, radius: i32, factory: F) -> Self
    where
        F: Fn(i32, i32) -> T + Send + Sync + 'static,
        T: Send + 'static,
    {
        let size = radius * 2 + 1;
        let min_x = center_x - radius;
        let min_z = center_z - radius;
        let cap = (size * size) as usize;
        let size_usize = size as usize;

        let cache: Vec<T> = (0..cap)
            .map(|index| {
                let x_offset = (index % size_usize) as i32;
                let z_offset = (index / size_usize) as i32;
                factory(min_x + x_offset, min_z + z_offset)
            })
            .collect();

        Self {
            min_x,
            min_z,
            size,
            cache,
        }
    }

    /// Creates a `StaticCache2D`, or returns `None` if any element is missing.
    pub fn try_create<F>(center_x: i32, center_z: i32, radius: i32, factory: F) -> Option<Self>
    where
        F: Fn(i32, i32) -> Option<T>,
    {
        let size = radius * 2 + 1;
        let min_x = center_x - radius;
        let min_z = center_z - radius;
        let cap = (size * size) as usize;
        let size_usize = size as usize;

        let cache = (0..cap)
            .map(|index| {
                let x_offset = (index % size_usize) as i32;
                let z_offset = (index / size_usize) as i32;
                factory(min_x + x_offset, min_z + z_offset)
            })
            .collect::<Option<Vec<T>>>()?;

        Some(Self {
            min_x,
            min_z,
            size,
            cache,
        })
    }

    /// Gets a reference to an element by world coordinates.
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds.
    #[must_use]
    pub fn get(&self, x: i32, z: i32) -> &T {
        let Some(value) = self.try_get(x, z) else {
            panic!(
                "Out of bounds: ({x}, {z}) vs [({}, {}) to ({}, {})]",
                self.min_x,
                self.min_z,
                self.min_x + self.size - 1,
                self.min_z + self.size - 1
            );
        };
        value
    }

    /// Gets a reference to an element by world coordinates.
    #[must_use]
    pub fn try_get(&self, x: i32, z: i32) -> Option<&T> {
        let rel_x = x - self.min_x;
        let rel_z = z - self.min_z;

        if rel_x >= 0 && rel_x < self.size && rel_z >= 0 && rel_z < self.size {
            let index = (rel_z * self.size + rel_x) as usize;
            self.cache.get(index)
        } else {
            None
        }
    }
}


/// Resolves the square of holders centred on `center` out to `radius`.
///
/// Returns `None` if any cell has no holder, which is the caller's signal that
/// the neighbourhood is not ready to be worked on rather than an error: a
/// position can legitimately carry a ticket level while its holder is still
/// being created, or has been taken out for unloading.
pub(crate) fn resolve_halo_at(
    chunk_map: &Arc<ChunkMap>,
    center: ChunkPos,
    radius: i32,
) -> Option<Arc<StaticCache2D<Arc<ChunkHolder>>>> {
    let chunk_map = Arc::clone(chunk_map);
    StaticCache2D::try_create(center.0.x, center.0.y, radius, move |x, y| {
        chunk_map
            .chunks
            .read_sync(&ChunkPos::new(x, y), |_, chunk_holder| chunk_holder.clone())
    })
    .map(Arc::new)
}
