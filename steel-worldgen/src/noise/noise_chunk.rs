//! `NoiseChunk`: cell-based terrain density evaluation with trilinear interpolation.
//!
//! Matches vanilla's `NoiseChunk` + `NoiseBasedChunkGenerator.doFill()` flow.
//!
//! Vanilla wraps density functions with `Interpolated` markers. Only the inner
//! functions (arguments to `Interpolated`) are evaluated at cell corners; the
//! outer operations (squeeze, min, etc.) are applied per-block after trilinear
//! interpolation. Each `Interpolated` marker gets its own independent channel.
//!
//! Cell dimensions depend on the dimension's noise settings.

use std::marker::PhantomData;
use std::simd::{f64x4, f64x8};

use steel_math::lerp;
use steel_worldgen::density::{ColumnCache, DimensionNoises, NoiseSettings};

use crate::noise::{Beardifier, BeardifierColumn};

/// Maximum number of interpolation channels supported.
/// Overworld uses 8 (1 terrain + 4 noodle caves + 3 vein channels), nether/end use 1.
/// Interpolation channels a slice reserves per cell corner.
///
/// This is the `SoA` stride, so it is also how far apart two corners' channel
/// blocks sit in memory. Overworld uses 8 channels and the other dimensions
/// use 1, so a wider stride buys nothing and costs the trilerp loop a second
/// cache line per corner: at stride 16 a corner's live 64 bytes sat in the
/// first half of a 128-byte span, and the loop sweeps every corner of a cell
/// sixteen times.
const MAX_INTERP: usize = 8;

/// Maximum slice length (`z_corners` * `corners_y`) across all dimensions.
/// Overworld: (16/4+1) * (384/8+1) = 5 * 49 = 245. Rounded up for headroom.
const MAX_SLICE_LEN: usize = 256;

/// SIMD lanes the wide cell-corner fill evaluates at once.
///
/// Matches the vector width the density transpiler emits, so this and the
/// generated `DENSITY_LANES` must move together. Eight `f64` lanes is one
/// AVX-512 register; on narrower targets `std::simd` splits it, which still
/// works and is what non-AVX-512 hosts get.
const DENSITY_LANES: usize = 8;

/// How many leading entries of the blended-noise column actually have to be
/// computed.
///
/// `threshold` is a dimension's
/// [`BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y`], the Y at and above which every
/// interpolated channel multiplies its blended-noise contribution by an
/// exactly-zero top slide. Corners at or above it read a leftover `0.0` from
/// the zero-initialised column and still produce bit-identical channel values,
/// so the column can stop there. `None` keeps the full column and reproduces
/// the unoptimized behaviour exactly.
///
/// `block_ys` is ascending, so the cut is a `partition_point`; the threshold is
/// never a literal here because the y=256 band is an overworld-only fact.
///
/// [`BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y`]: crate::density::DimensionNoises::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y
fn blended_column_len(threshold: Option<i32>, block_ys: &[i32], corners_y: usize) -> usize {
    threshold.map_or(corners_y, |threshold| {
        block_ys.partition_point(|&y| y < threshold)
    })
}

/// How far below zero channel 0 must be before a run is treated as air.
///
/// The trilerp evaluates `a + t*(b - a)` rather than the convex form, so with
/// both endpoints at zero-ish magnitudes the rounded result can sit a few ulps
/// above zero while the exact value is below it. Channel 0 has magnitude on the
/// order of 0.05-0.64 here, so a margin this size costs no measurable hit rate
/// while leaving many orders of magnitude of slack over the ~8 chained lerps.
const AIR_SKIP_CHANNEL0_MARGIN: f64 = -1e-9;

/// Cache-line aligned slice storage.
///
/// With `MAX_INTERP` at 8 a corner's channels occupy exactly 64 bytes, so
/// aligning the allocation makes each corner one whole cache line and keeps
/// either `f64x4` half of it from straddling two lines. `Box` alone only
/// promises 8-byte alignment.
#[repr(C, align(64))]
struct SliceStorage([f64; MAX_INTERP * MAX_SLICE_LEN]);

impl std::ops::Deref for SliceStorage {
    type Target = [f64; MAX_INTERP * MAX_SLICE_LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SliceStorage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Stores density values at cell corners for a single chunk and provides
/// trilinear interpolation between corners for block-level resolution.
///
/// Supports multiple interpolation channels matching vanilla's multi-interpolator
/// system. Each `Interpolated` marker in the density function tree gets its own
/// channel, filled at cell corners and interpolated independently.
///
/// Storage is per-corner `SoA` — `slice[corner_idx * MAX_INTERP + ch]` — so 4
/// adjacent channels' values at a given corner sit in contiguous memory,
/// enabling a single `f64x4` load and SIMD-batched trilinear interpolation
/// across 4 channels per block.
pub struct NoiseChunk<N: DimensionNoises> {
    /// One slice per cell-X boundary, holding density values at the cell
    /// corners on that X-plane. Length is `cell_count_xz + 1`. Indexed as
    /// `slices[cx][corner_idx * MAX_INTERP + ch]` where
    /// `corner_idx = z_corner * corners_y + y_corner` (range `[0, slice_len)`)
    /// and `ch` is the interpolation channel (range `[0, interp_count)`).
    ///
    /// We keep all slices materialized rather than alternating two buffers so
    /// the slice-fill phase can run in parallel: each `cx` boundary's noise
    /// tree evaluation is independent. The per-block trilerp loop then
    /// indexes `slices[cx]` and `slices[cx + 1]` sequentially.
    slices: Vec<Box<SliceStorage>>,
    /// Number of active interpolation channels.
    interp_count: usize,
    /// Number of Y corners per Z column (`cell_count_y` + 1).
    corners_y: usize,

    /// Per-corner block-Y values, precomputed once at construction.
    /// Same for every slice fill (depends only on `cell_min_y`,
    /// `cell_height`, and `corners_y`).
    block_ys: Vec<i32>,

    /// First cell X/Z in world coordinates (cell index, not block).
    first_cell_x: i32,
    first_cell_z: i32,
    /// Minimum cell Y index.
    cell_min_y: i32,
    /// Number of cells in Y direction.
    cell_count_y: usize,
    /// Number of cells per chunk in XZ.
    cell_count_xz: usize,

    _phantom: PhantomData<N>,
}

impl<N: DimensionNoises> NoiseChunk<N> {
    /// Create a new `NoiseChunk` for the given chunk position.
    ///
    /// `chunk_min_block_x` and `chunk_min_block_z` are the world-space block
    /// coordinates of the chunk's northwest corner.
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "panic is a compile-time constant check"
    )]
    pub fn new(chunk_min_block_x: i32, chunk_min_block_z: i32) -> Self {
        let cell_width = N::Settings::CELL_WIDTH;
        let cell_height = N::Settings::CELL_HEIGHT;
        let min_y = N::Settings::MIN_Y;
        let height = N::Settings::HEIGHT;

        let first_cell_x = chunk_min_block_x.div_euclid(cell_width);
        let first_cell_z = chunk_min_block_z.div_euclid(cell_width);
        let cell_min_y = min_y.div_euclid(cell_height);

        let cell_count_xz = (16 / cell_width) as usize;
        let cell_count_y = (height / cell_height) as usize;
        let corners_y = cell_count_y + 1;
        let z_corners = cell_count_xz + 1;
        let slice_len = z_corners * corners_y;

        let interp_count = N::interpolated_count();
        assert!(
            slice_len <= MAX_SLICE_LEN,
            "slice_len {slice_len} exceeds MAX_SLICE_LEN {MAX_SLICE_LEN}"
        );
        assert!(
            interp_count <= MAX_INTERP,
            "interp_count {interp_count} exceeds MAX_INTERP {MAX_INTERP}"
        );

        let block_ys: Vec<i32> = (0..corners_y)
            .map(|cy| (cy as i32 + cell_min_y) * cell_height)
            .collect();

        let n_slices = cell_count_xz + 1;
        let mut slices = Vec::with_capacity(n_slices);
        for _ in 0..n_slices {
            // The boxed fixed-size array keeps the `[f64; N]` type that the SIMD
            // `fill` path and its `get_unchecked` SAFETY proofs rely on. This is a
            // per-chunk constructor, not a hot path, so the stack temporary is fine.
            #[expect(
                clippy::large_stack_arrays,
                reason = "fixed-size boxed array keeps the [f64; N] type the SIMD fill path relies on; cold per-chunk constructor"
            )]
            slices.push(Box::new(SliceStorage([0.0; MAX_INTERP * MAX_SLICE_LEN])));
        }

        Self {
            slices,
            interp_count,
            corners_y,
            block_ys,
            first_cell_x,
            first_cell_z,
            cell_min_y,
            cell_count_y,
            cell_count_xz,
            _phantom: PhantomData,
        }
    }

    /// Fill the slice buffer for the given cell X. Free-standing function so
    /// each parallel slice-fill can run on its own thread with its own
    /// `ColumnCache` clone.
    #[expect(
        clippy::too_many_arguments,
        reason = "slice filling needs the precomputed geometry and per-thread cache"
    )]
    fn fill_slice_into(
        slice: &mut [f64; MAX_INTERP * MAX_SLICE_LEN],
        cell_x: i32,
        block_ys: &[i32],
        blended_column: &mut [f64],
        blended_len: usize,
        interp_count: usize,
        corners_y: usize,
        cell_count_xz: usize,
        first_cell_z: i32,
        noises: &N,
        cache: &mut N::ColumnCache,
    ) {
        let cell_width = N::Settings::CELL_WIDTH;

        let block_x = cell_x * cell_width;

        let mut values = [0.0f64; MAX_INTERP];

        // Scratch buffer for the 4-Y SIMD batch. Lane-major SoA: lane `i`'s
        // `interp_count` channels live at `values_wide[i * interp_count..]`.
        let mut values_wide = [0.0f64; DENSITY_LANES * MAX_INTERP];

        for cz in 0..=cell_count_xz {
            let cell_z = first_cell_z + cz as i32;
            let block_z = cell_z * cell_width;

            // Ensure column cache for this (x, z)
            cache.ensure(block_x, block_z, noises);

            // SIMD-batch blended noise for the Y column, stopping at
            // `blended_len` (see `NoiseChunk::fill`). Entries at and past it are
            // never written by anyone, so they keep the `0.0` they were
            // allocated with — which is exactly what the corner loops below read
            // for those corners, and what the generated channel expressions
            // annihilate anyway.
            noises.compute_noise_column(
                block_x,
                &block_ys[..blended_len],
                block_z,
                &mut blended_column[..blended_len],
            );

            // Wide SIMD-batched corner fill, `DENSITY_LANES` Y values at a
            // time. Tail is handled by the scalar loop below for any remaining
            // `corners_y % DENSITY_LANES` corners.
            let mut cy = 0;
            while cy + DENSITY_LANES <= corners_y {
                let mut ys = [0.0f64; DENSITY_LANES];
                let mut blended = [0.0f64; DENSITY_LANES];
                for lane in 0..DENSITY_LANES {
                    ys[lane] = f64::from(block_ys[cy + lane]);
                    blended[lane] = blended_column[cy + lane];
                }

                noises.fill_cell_corner_densities_4x(
                    cache,
                    block_x,
                    f64x8::from_array(ys),
                    block_z,
                    f64x8::from_array(blended),
                    &mut values_wide[..DENSITY_LANES * interp_count],
                );

                for lane in 0..DENSITY_LANES {
                    let lane_cy = cy + lane;
                    let src = &values_wide[lane * interp_count..(lane + 1) * interp_count];
                    let corner_idx = cz * corners_y + lane_cy;
                    let base = corner_idx * MAX_INTERP;
                    slice[base..base + interp_count].copy_from_slice(src);
                }

                cy += DENSITY_LANES;
            }

            while cy < corners_y {
                let block_y = block_ys[cy];

                noises.fill_cell_corner_densities(
                    cache,
                    block_x,
                    block_y,
                    block_z,
                    blended_column[cy],
                    &mut values[..interp_count],
                );

                let corner_idx = cz * corners_y + cy;
                let base = corner_idx * MAX_INTERP;
                slice[base..base + interp_count].copy_from_slice(&values[..interp_count]);

                cy += 1;
            }
        }
    }

    /// Fill the chunk with terrain blocks using multi-channel trilinear interpolation.
    ///
    /// For each block position:
    /// 1. Trilinearly interpolate each channel independently from cell corners
    /// 2. Apply outer operations (squeeze, min, etc.) via `combine_interpolated`
    /// 3. Call `place_block` with the final density
    #[expect(
        clippy::too_many_lines,
        reason = "single SIMD trilinear-interpolation kernel; splitting the loop nest would scatter the per-corner SAFETY invariants"
    )]
    #[expect(
        clippy::similar_names,
        reason = "factor_{x,y,z}_v vector splats deliberately mirror their scalar factor_{x,y,z} sources"
    )]
    /// `air_only_above_y`: lowest y at which a non-positive density is
    /// guaranteed to place nothing (see `Aquifer::air_only_above_y`). `None`
    /// disables whole-run air skipping.
    pub fn fill<F>(
        &mut self,
        noises: &N,
        cache: &mut N::ColumnCache,
        beardifier: Option<&Beardifier>,
        air_only_above_y: Option<i32>,
        mut place_block: F,
    ) where
        F: FnMut(usize, i32, usize, f64, &[f64], &mut N::ColumnCache),
    {
        let cell_width = N::Settings::CELL_WIDTH;
        let cell_height = N::Settings::CELL_HEIGHT;
        let cell_count_xz = self.cell_count_xz;
        let cell_count_y = self.cell_count_y;
        let interp_count = self.interp_count;
        let corners_y = self.corners_y;
        let first_cell_x = self.first_cell_x;
        let first_cell_z = self.first_cell_z;
        let block_ys: &[i32] = &self.block_ys;

        // Pre-fill ALL slices sequentially. Each `(cell_x boundary)` slice is an
        // independent noise-tree evaluation; the grid in `cache` is set up by the
        // caller via `init_grid` and is read-only here, while each slice only
        // overwrites the cache's per-column active fields — so one cache is reused
        // across slices without cloning. The chunk pipeline already parallelises
        // across chunks, so parallelising the 5 slices here would nest rayon work
        // and add coordination + cache-clone overhead with no spare cores to use.
        let mut beard_column = beardifier.map(BeardifierColumn::new);
        let n_slices = cell_count_xz + 1;
        // `local_blended` stays `corners_y` long — every corner reads it — but
        // only its first `blended_len` entries are ever computed. Above the
        // dimension's threshold the generated channel expressions multiply the
        // blended noise by an exactly-zero top slide, so those corners produce
        // the same bits from the leftover `0.0` as from the real noise.
        let blended_len = blended_column_len(
            N::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y,
            block_ys,
            corners_y,
        );
        let mut local_blended = vec![0.0f64; corners_y];
        for cx_off in 0..n_slices {
            let cell_x = first_cell_x + cx_off as i32;
            Self::fill_slice_into(
                &mut self.slices[cx_off],
                cell_x,
                block_ys,
                &mut local_blended,
                blended_len,
                interp_count,
                corners_y,
                cell_count_xz,
                first_cell_z,
                noises,
                cache,
            );
        }

        let mut interpolated = [0.0f64; MAX_INTERP];

        for cell_x_idx in 0..cell_count_xz {
            for cell_z_idx in 0..cell_count_xz {
                for x_in_cell in 0..cell_width {
                    let factor_x = f64::from(x_in_cell) / f64::from(cell_width);
                    let local_x = (cell_x_idx as i32 * cell_width + x_in_cell) as usize;

                    for z_in_cell in 0..cell_width {
                        let factor_z = f64::from(z_in_cell) / f64::from(cell_width);
                        let local_z = (cell_z_idx as i32 * cell_width + z_in_cell) as usize;

                        // The beardifier's horizontal reach is fixed for the
                        // column, so narrow it here rather than re-testing every
                        // piece for all 384 blocks below.
                        let world_x_col =
                            cell_x_idx as i32 * cell_width + x_in_cell + self.first_cell_x * cell_width;
                        let world_z_col =
                            cell_z_idx as i32 * cell_width + z_in_cell + self.first_cell_z * cell_width;
                        let beard_in_column = beard_column
                            .as_mut()
                            .is_some_and(|column| column.retarget(world_x_col, world_z_col));

                        // Pre-compute flat indices for this Z column
                        let z0_base = cell_z_idx * corners_y;
                        let z1_base = (cell_z_idx + 1) * corners_y;

                        // Process entire Y column at this (x, z)
                        for cell_y_idx in (0..cell_count_y).rev() {
                            // Whole-run air skip.
                            //
                            // Every channel is affine in y between the cell's two
                            // y-corners, so channel 0 over this run is bounded by
                            // its two bilerped endpoints -- and for these
                            // dimensions a non-positive channel 0 forces a
                            // non-positive density (see
                            // `DENSITY_NONPOSITIVE_FROM_CHANNEL0`). Above the
                            // aquifer's air threshold such a block places nothing,
                            // so the entire run can be dropped: no trilerp, no
                            // combine, no aquifer, no placement. Sections start
                            // out homogeneous air, so not writing is correct.
                            //
                            // The margin covers rounding: the trilerp uses
                            // `a + t*(b-a)`, which can land a hair above zero when
                            // both endpoints are zero-ish. Written so NaN falls
                            // through to the slow path.
                            if N::DENSITY_NONPOSITIVE_FROM_CHANNEL0
                                && !beard_in_column
                                && let Some(air_above) = air_only_above_y
                                && (self.cell_min_y + cell_y_idx as i32) * cell_height >= air_above
                            {
                                let i0b = (z0_base + cell_y_idx) * MAX_INTERP;
                                let i1b = (z1_base + cell_y_idx) * MAX_INTERP;
                                let s0 = &*self.slices[cell_x_idx];
                                let s1 = &*self.slices[cell_x_idx + 1];
                                // SAFETY: same indices the trilerp below reads.
                                let (bottom, top) = unsafe {
                                    (
                                        lerp(
                                            factor_z,
                                            lerp(
                                                factor_x,
                                                *s0.get_unchecked(i0b),
                                                *s1.get_unchecked(i0b),
                                            ),
                                            lerp(
                                                factor_x,
                                                *s0.get_unchecked(i1b),
                                                *s1.get_unchecked(i1b),
                                            ),
                                        ),
                                        lerp(
                                            factor_z,
                                            lerp(
                                                factor_x,
                                                *s0.get_unchecked(i0b + MAX_INTERP),
                                                *s1.get_unchecked(i0b + MAX_INTERP),
                                            ),
                                            lerp(
                                                factor_x,
                                                *s0.get_unchecked(i1b + MAX_INTERP),
                                                *s1.get_unchecked(i1b + MAX_INTERP),
                                            ),
                                        ),
                                    )
                                };
                                if bottom.max(top) <= AIR_SKIP_CHANNEL0_MARGIN {
                                    continue;
                                }
                            }

                            for y_in_cell in (0..cell_height).rev() {
                                let factor_y = f64::from(y_in_cell) / f64::from(cell_height);

                                let world_y =
                                    (self.cell_min_y + cell_y_idx as i32) * cell_height + y_in_cell;

                                // Trilinearly interpolate each channel.
                                //
                                // SoA layout puts the 4 (or fewer) channel
                                // values for one corner in contiguous memory,
                                // so a single `f64x4` load per corner replaces
                                // four scattered scalar loads in the legacy
                                // AoS path. Math is per-lane independent and
                                // matches the scalar order exactly, so the
                                // result is bit-identical to vanilla.
                                //
                                // SAFETY: max index = (z1_base + cell_y_idx + 1) * MAX_INTERP + (ch_batch+3)
                                //         ≤ ((cell_count_xz+1)*corners_y - 1) * MAX_INTERP + MAX_INTERP - 1
                                //         < MAX_SLICE_LEN * MAX_INTERP
                                let i0_base = (z0_base + cell_y_idx) * MAX_INTERP;
                                let i1_base = (z1_base + cell_y_idx) * MAX_INTERP;
                                let i0_next = i0_base + MAX_INTERP;
                                let i1_next = i1_base + MAX_INTERP;
                                let s0 = &*self.slices[cell_x_idx];
                                let s1 = &*self.slices[cell_x_idx + 1];
                                let factor_y_v = f64x4::splat(factor_y);
                                let factor_x_v = f64x4::splat(factor_x);
                                let factor_z_v = f64x4::splat(factor_z);

                                let mut ch_batch = 0;
                                while ch_batch + 4 <= interp_count {
                                    // SAFETY: ch_batch+3 < interp_count ≤ MAX_INTERP, all base indices in bounds.
                                    unsafe {
                                        let n000 = f64x4::from_slice(s0.get_unchecked(
                                            i0_base + ch_batch..i0_base + ch_batch + 4,
                                        ));
                                        let n001 = f64x4::from_slice(s0.get_unchecked(
                                            i1_base + ch_batch..i1_base + ch_batch + 4,
                                        ));
                                        let n100 = f64x4::from_slice(s1.get_unchecked(
                                            i0_base + ch_batch..i0_base + ch_batch + 4,
                                        ));
                                        let n101 = f64x4::from_slice(s1.get_unchecked(
                                            i1_base + ch_batch..i1_base + ch_batch + 4,
                                        ));
                                        let n010 = f64x4::from_slice(s0.get_unchecked(
                                            i0_next + ch_batch..i0_next + ch_batch + 4,
                                        ));
                                        let n011 = f64x4::from_slice(s0.get_unchecked(
                                            i1_next + ch_batch..i1_next + ch_batch + 4,
                                        ));
                                        let n110 = f64x4::from_slice(s1.get_unchecked(
                                            i0_next + ch_batch..i0_next + ch_batch + 4,
                                        ));
                                        let n111 = f64x4::from_slice(s1.get_unchecked(
                                            i1_next + ch_batch..i1_next + ch_batch + 4,
                                        ));

                                        let d00 = n000 + factor_y_v * (n010 - n000);
                                        let d10 = n100 + factor_y_v * (n110 - n100);
                                        let d01 = n001 + factor_y_v * (n011 - n001);
                                        let d11 = n101 + factor_y_v * (n111 - n101);
                                        let d0 = d00 + factor_x_v * (d10 - d00);
                                        let d1 = d01 + factor_x_v * (d11 - d01);
                                        let result = d0 + factor_z_v * (d1 - d0);
                                        let arr = result.to_array();
                                        let dst =
                                            interpolated.get_unchecked_mut(ch_batch..ch_batch + 4);
                                        dst.copy_from_slice(&arr);
                                    }
                                    ch_batch += 4;
                                }
                                // Scalar tail (when interp_count is not a multiple of 4).
                                while ch_batch < interp_count {
                                    let ch = ch_batch;
                                    // SAFETY: ch < interp_count ≤ MAX_INTERP; indices in bounds (see comment above).
                                    unsafe {
                                        let n000 = *s0.get_unchecked(i0_base + ch);
                                        let n001 = *s0.get_unchecked(i1_base + ch);
                                        let n100 = *s1.get_unchecked(i0_base + ch);
                                        let n101 = *s1.get_unchecked(i1_base + ch);
                                        let n010 = *s0.get_unchecked(i0_next + ch);
                                        let n011 = *s0.get_unchecked(i1_next + ch);
                                        let n110 = *s1.get_unchecked(i0_next + ch);
                                        let n111 = *s1.get_unchecked(i1_next + ch);

                                        let d00 = lerp(factor_y, n000, n010);
                                        let d10 = lerp(factor_y, n100, n110);
                                        let d01 = lerp(factor_y, n001, n011);
                                        let d11 = lerp(factor_y, n101, n111);
                                        let d0 = lerp(factor_x, d00, d10);
                                        let d1 = lerp(factor_x, d01, d11);
                                        *interpolated.get_unchecked_mut(ch) =
                                            lerp(factor_z, d0, d1);
                                    }
                                    ch_batch += 1;
                                }

                                // Apply outer operations per-block.
                                // x/z are 0 because vanilla's outer operations (squeeze, add, mul,
                                // quarter_negative, blend_alpha, blend_offset) are x/z-independent;
                                // only Y matters for YClampedGradient.
                                let mut density = noises.combine_interpolated(
                                    cache,
                                    &interpolated[..interp_count],
                                    0,
                                    world_y,
                                    0,
                                );

                                // Vanilla integrates beardifier as `add(final_density, beardifier)`
                                // wrapped in `cacheAllInCell` — i.e. evaluated per-block, after the
                                // outer ops on `final_density` have run. Adding it at cell corners
                                // would put it inside the squeeze and trilerp it linearly across
                                // the cell, both of which diverge from vanilla for large beardifier
                                // values inside a structure's pieces.
                                if beard_in_column
                                    && let Some(beard) = beard_column.as_ref()
                                {
                                    density += beard.compute(world_x_col, world_y, world_z_col);
                                }

                                place_block(
                                    local_x,
                                    world_y,
                                    local_z,
                                    density,
                                    &interpolated[..interp_count],
                                    cache,
                                );
                            }
                        }
                    }
                }
            }

            // No swap needed: all slices are pre-filled and indexed directly
            // via `self.slices[cell_x_idx]` / `[cell_x_idx + 1]`.
        }
    }
}

#[cfg(test)]
mod blended_column_tests {
    use super::{DENSITY_LANES, MAX_INTERP, NoiseChunk, blended_column_len};
    use crate::density::{ColumnCache, DimensionNoises, NoiseSettings};
    use crate::density_functions::end::EndNoises;
    use crate::density_functions::nether::NetherNoises;
    use crate::density_functions::overworld::{
        BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y as OVERWORLD_THRESHOLD, OverworldNoiseSettings,
    };
    use crate::noise_parameters::get_noise_parameters;
    use crate::random::{Random, legacy_random::LegacyRandom, xoroshiro::Xoroshiro};

    /// Rebuild the cell-corner Y values exactly as `NoiseChunk::new` does.
    fn block_ys(min_y: i32, height: i32, cell_height: i32) -> Vec<i32> {
        let cell_min_y = min_y.div_euclid(cell_height);
        let corners_y = (height / cell_height) as usize + 1;
        (0..corners_y)
            .map(|cy| (cy as i32 + cell_min_y) * cell_height)
            .collect()
    }

    /// The overworld column geometry the truncation depends on. Asserted rather
    /// than assumed: the y=256 band is an overworld-only fact, and a datapack
    /// change to `min_y` / `height` / `size_vertical` would move the corner the
    /// threshold lands on.
    #[test]
    fn overworld_corner_geometry_puts_the_top_slide_at_corner_40() {
        assert_eq!(OverworldNoiseSettings::MIN_Y, -64);
        assert_eq!(OverworldNoiseSettings::HEIGHT, 384);
        assert_eq!(OverworldNoiseSettings::CELL_HEIGHT, 8);

        let ys = block_ys(
            OverworldNoiseSettings::MIN_Y,
            OverworldNoiseSettings::HEIGHT,
            OverworldNoiseSettings::CELL_HEIGHT,
        );

        assert_eq!(ys.len(), 49, "corners_y");
        assert_eq!(ys[40], 256, "block_ys[40]");
        assert_eq!(*ys.first().unwrap(), -64);
        assert_eq!(*ys.last().unwrap(), 320);
    }

    /// The transpiler must recognise the overworld's `y_clamped_gradient`
    /// top slide (240 -> 256, 1 -> 0) and cut the column at the corner where it
    /// first evaluates to exactly zero.
    #[test]
    fn overworld_threshold_truncates_to_forty_corners() {
        assert_eq!(OVERWORLD_THRESHOLD, Some(256));

        let ys = block_ys(
            OverworldNoiseSettings::MIN_Y,
            OverworldNoiseSettings::HEIGHT,
            OverworldNoiseSettings::CELL_HEIGHT,
        );
        let len = blended_column_len(OVERWORLD_THRESHOLD, &ys, ys.len());

        assert_eq!(len, 40);
        // Every skipped corner is at or above the threshold, so its top slide
        // is exactly zero and the blended noise there cannot reach the output.
        assert!(ys[len..].iter().all(|&y| y >= 256));
        // The last computed corner is still below it, so nothing live is lost.
        assert!(ys[len - 1] < 256);
        // 40 is a whole number of SIMD batches, so the scalar remainder in
        // `BlendedNoise::compute_column` disappears too.
        assert_eq!(len % DENSITY_LANES, 0);
    }

    /// A dimension that opts out must behave exactly as before: the whole
    /// column is computed, whatever its geometry.
    #[test]
    fn none_threshold_keeps_the_whole_column() {
        for (min_y, height, cell_height) in [(-64, 384, 8), (0, 128, 8), (0, 128, 4)] {
            let ys = block_ys(min_y, height, cell_height);
            assert_eq!(
                blended_column_len(None, &ys, ys.len()),
                ys.len(),
                "None must not truncate ({min_y}, {height}, {cell_height})"
            );
        }
    }

    /// Guards the invariant the truncation relies on: the column buffer stays
    /// full length and its tail keeps the `0.0` it was allocated with, so the
    /// corner loops never read uninitialised memory.
    #[test]
    fn untouched_tail_stays_zero() {
        let ys = block_ys(-64, 384, 8);
        let corners_y = ys.len();
        let len = blended_column_len(Some(256), &ys, corners_y);

        let mut column = vec![0.0f64; corners_y];
        // Stand in for `compute_noise_column`, which writes only the prefix.
        for (i, slot) in column[..len].iter_mut().enumerate() {
            *slot = i as f64 + 1.0;
        }

        assert_eq!(column.len(), corners_y, "buffer must stay full length");
        assert!(
            column[len..].iter().all(|&v| v == 0.0),
            "tail must stay 0.0 for the corner loops to read"
        );
        // Reused across slices without re-zeroing: the tail is never written,
        // so a second pass leaves it zero too.
        for (i, slot) in column[..len].iter_mut().enumerate() {
            *slot = i as f64 + 100.0;
        }
        assert!(column[len..].iter().all(|&v| v == 0.0));
    }

    /// Read the threshold back through the trait exactly as `fill` does, so the
    /// generated const is really the one the runtime sees, and check each
    /// dimension truncates only where its own geometry allows.
    #[test]
    fn every_dimension_threshold_is_consistent_with_its_geometry() {
        fn check<N: DimensionNoises>(min_y: i32, height: i32, cell_height: i32, label: &str) {
            let ys = block_ys(min_y, height, cell_height);
            let corners_y = ys.len();
            let len = blended_column_len(N::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y, &ys, corners_y);

            assert!(len <= corners_y, "{label}: cannot exceed the column");

            match N::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y {
                // Opted out: nothing changes.
                None => assert_eq!(len, corners_y, "{label}: None must keep the full column"),
                Some(threshold) => {
                    // Every dropped corner sits at or above the threshold...
                    assert!(
                        ys[len..].iter().all(|&y| y >= threshold),
                        "{label}: dropped a corner below the threshold"
                    );
                    // ...and every kept corner below it is genuinely needed.
                    assert!(
                        ys[..len].iter().all(|&y| y < threshold),
                        "{label}: kept a corner at or above the threshold"
                    );
                }
            }
        }

        check::<crate::density_functions::overworld::OverworldNoises>(-64, 384, 8, "overworld");
        check::<crate::density_functions::nether::NetherNoises>(0, 128, 8, "nether");
        check::<crate::density_functions::end::EndNoises>(0, 128, 4, "end");
    }

    // ── The `None` arm, on real dimensions ──────────────────────────────────
    //
    // Everything above is arithmetic on `blended_column_len`. What follows runs
    // whole dimensions through `NoiseChunk` itself, because nothing else covers
    // the `None` arm: only the overworld emits `Some`, so the overworld parity
    // gates never reach it, and `nether_biome_hashes` / `end_biome_hashes` hash
    // biomes rather than noise columns — they would pass with the nether's
    // blended column truncated to nothing. A runtime-loaded datapack that
    // declines the optimisation takes this same path.

    /// Block coordinates of the chunk every fill below is run for. Off the
    /// origin so the noise is not sampled at a degenerate lattice point.
    const PROBE_CHUNK_X: i32 = 48;
    /// See [`PROBE_CHUNK_X`].
    const PROBE_CHUNK_Z: i32 = -80;
    /// World seed the probe dimensions are built from.
    const PROBE_SEED: u64 = 0x5EED_C0DE_1234_5678;

    /// Every live cell-corner channel value of a filled chunk, as raw bits.
    ///
    /// Bits rather than `f64` because the claim under test is bit-identical
    /// parity: a sign-flipped zero or a 1-ULP drift has to fail.
    fn corner_bits<N: DimensionNoises>(chunk: &NoiseChunk<N>) -> Vec<u64> {
        let corners = (chunk.cell_count_xz + 1) * chunk.corners_y;
        let mut bits = Vec::with_capacity(chunk.slices.len() * corners * chunk.interp_count);
        for slice in &chunk.slices {
            let values: &[f64] = &slice[..];
            for corner in 0..corners {
                let base = corner * MAX_INTERP;
                bits.extend(
                    values[base..base + chunk.interp_count]
                        .iter()
                        .map(|value| value.to_bits()),
                );
            }
        }
        bits
    }

    /// Fill every slice of one chunk with an explicit blended-column length.
    ///
    /// `blended_len == corners_y` is the pre-optimisation body verbatim: before
    /// the parameter existed, `fill_slice_into` handed `compute_noise_column`
    /// the whole column and every corner read a computed entry.
    fn corner_bits_at_len<N: DimensionNoises>(
        noises: &N,
        cache: &mut N::ColumnCache,
        blended_len: usize,
    ) -> Vec<u64> {
        let mut chunk = NoiseChunk::<N>::new(PROBE_CHUNK_X, PROBE_CHUNK_Z);
        let corners_y = chunk.corners_y;
        let interp_count = chunk.interp_count;
        let cell_count_xz = chunk.cell_count_xz;
        let first_cell_x = chunk.first_cell_x;
        let first_cell_z = chunk.first_cell_z;
        let block_ys = chunk.block_ys.clone();
        // Allocated zeroed and never re-zeroed, exactly like `fill`'s
        // `local_blended`: a short column leaves 0.0 in the tail for the corner
        // loops to read.
        let mut column = vec![0.0f64; corners_y];
        for cx_off in 0..=cell_count_xz {
            NoiseChunk::<N>::fill_slice_into(
                &mut chunk.slices[cx_off],
                first_cell_x + cx_off as i32,
                &block_ys,
                &mut column,
                blended_len,
                interp_count,
                corners_y,
                cell_count_xz,
                first_cell_z,
                noises,
                cache,
            );
        }
        corner_bits(&chunk)
    }

    /// Which cell corners' channel values actually change when their
    /// blended-noise input does, probed through the dimension's own generated
    /// corner fill.
    ///
    /// Whether a corner consumes it is a function of the corner's Y alone —
    /// what annihilates it is a `y_clamped_gradient` slide — so one column
    /// decides it for the whole chunk.
    fn corners_consuming_blended<N: DimensionNoises>(
        noises: &N,
        cache: &mut N::ColumnCache,
        block_ys: &[i32],
    ) -> Vec<bool> {
        let interp_count = N::interpolated_count();
        let mut without = vec![0.0f64; interp_count];
        let mut with = vec![0.0f64; interp_count];
        cache.ensure(PROBE_CHUNK_X, PROBE_CHUNK_Z, noises);
        block_ys
            .iter()
            .map(|&y| {
                let (x, z) = (PROBE_CHUNK_X, PROBE_CHUNK_Z);
                noises.fill_cell_corner_densities(cache, x, y, z, 0.0, &mut without);
                noises.fill_cell_corner_densities(cache, x, y, z, 1.0, &mut with);
                without
                    .iter()
                    .zip(&with)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            })
            .collect()
    }

    /// Compare two corner-bit vectors, reporting the first differing value
    /// rather than dumping several hundred `u64`s.
    fn assert_corner_bits_eq(actual: &[u64], expected: &[u64], label: &str, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: {what} (length)");
        if let Some(i) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{label}: {what}; first difference at value {i}: {} vs {}",
                f64::from_bits(actual[i]),
                f64::from_bits(expected[i])
            );
        }
    }

    /// A dimension that opts out must produce exactly the corner densities the
    /// unoptimized code produced, and must be one whose blended noise really
    /// reaches those densities.
    fn check_opted_out<N: DimensionNoises>(label: &str) {
        assert_eq!(
            N::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y,
            None,
            "{label} no longer opts out; this test exists to cover the None arm, \
             so retarget it at a dimension that still takes it"
        );

        let splitter = if N::Settings::LEGACY_RANDOM_SOURCE {
            LegacyRandom::from_seed(PROBE_SEED).next_positional()
        } else {
            Xoroshiro::from_seed(PROBE_SEED).next_positional()
        };
        let noises = N::create(PROBE_SEED, &splitter, &get_noise_parameters());

        let mut cache = N::ColumnCache::default();
        cache.init_grid(PROBE_CHUNK_X, PROBE_CHUNK_Z, &noises);

        // The production path, which is the only thing that reads the const.
        let mut chunk = NoiseChunk::<N>::new(PROBE_CHUNK_X, PROBE_CHUNK_Z);
        chunk.fill(&noises, &mut cache, None, None, |_, _, _, _, _, _| {});
        let production = corner_bits(&chunk);

        let corners_y = chunk.corners_y;
        let block_ys = chunk.block_ys.clone();

        // The pre-optimisation behaviour: the whole column, for every slice.
        let full_column = corner_bits_at_len::<N>(&noises, &mut cache, corners_y);
        assert_corner_bits_eq(
            &production,
            &full_column,
            label,
            "opting out must compute and consume the whole blended column",
        );

        // Non-vacuity. Without this the equality above would hold just as well
        // for a dimension that never looks at its blended noise, which is the
        // one way this test could pass while proving nothing.
        let consuming = corners_consuming_blended::<N>(&noises, &mut cache, &block_ys);
        let Some(last) = consuming.iter().rposition(|&consumed| consumed) else {
            panic!("{label}: no cell corner consumes its blended noise");
        };

        // Corners above the last consuming one are the only ones a threshold
        // could ever drop for free...
        assert_corner_bits_eq(
            &corner_bits_at_len::<N>(&noises, &mut cache, last + 1),
            &full_column,
            label,
            "dropping only corners that ignore their blended noise must change nothing",
        );
        // ...and dropping one more is visible in the densities. So the column
        // this dimension computes is live all the way up to corner `last`, and
        // `None` — which truncates nothing — is what keeps it that way.
        assert!(
            corner_bits_at_len::<N>(&noises, &mut cache, last) != full_column,
            "{label}: corner y={} consumes its blended noise, so cutting the column \
             there must change the densities",
            block_ys[last]
        );
        assert_eq!(
            blended_column_len(
                N::BLENDED_NOISE_IRRELEVANT_AT_OR_ABOVE_Y,
                &block_ys,
                corners_y
            ),
            corners_y,
            "{label}: opting out must leave the column at full length"
        );
    }

    /// The dimensions that take the `None` arm, end to end.
    #[test]
    fn opted_out_dimensions_match_the_full_blended_column() {
        check_opted_out::<NetherNoises>("nether");
        check_opted_out::<EndNoises>("end");
    }
}
