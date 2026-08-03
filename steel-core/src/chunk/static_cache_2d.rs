//! A pre-filled square cache of chunk neighbours.
//!
//! The per-holder generation drive resolves one of these per fused run and
//! hands it to the step it dispatches.

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

    /// Wraps an already-filled row-major square.
    ///
    /// The resolvers that feed the per-holder generation drive cannot use
    /// [`Self::try_create`]: they walk the square in descending Chebyshev
    /// distance so the cells that gate the run are visited before the ones that
    /// merely have to be present, which is not the order the backing vector is
    /// stored in. They fill the vector by index instead and hand it over here.
    ///
    /// # Panics
    /// Panics if `cache` is not exactly `size * size` elements, which would make
    /// every subsequent lookup read the wrong cell.
    pub(crate) fn from_row_major(min_x: i32, min_z: i32, size: i32, cache: Vec<T>) -> Self {
        assert_eq!(
            cache.len(),
            (size * size) as usize,
            "a row-major square must be filled completely before it is published"
        );
        Self {
            min_x,
            min_z,
            size,
            cache,
        }
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
