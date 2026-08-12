use super::super::instrumentation::OreFeatureProfile;
use super::super::prelude::*;
use super::super::runner::FeatureDecorationRunner;
use smallvec::SmallVec;
use std::f32::consts::PI;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f64x8;
use std::time::Instant;
use steel_math::trig;
use steel_utils::PackedSectionBlockPos;
use steel_worldgen::state_resolver::WorldgenStateResolver;

impl FeatureDecorationRunner {
    pub(in crate::worldgen::feature) fn place_ore_feature(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        origin: BlockPos,
    ) -> bool {
        if config.size <= 0 {
            return false;
        }
        let direction = random.next_f32() * PI;
        let spread_xz = config.size as f32 / 8.0;
        let spread_xz_ceil = spread_xz.ceil() as i32;
        let max_radius = f32::midpoint(config.size as f32 / 16.0 * 2.0, 1.0).ceil() as i32;
        let sin = f64::from(direction).sin();
        let cos = f64::from(direction).cos();
        let x0 = f64::from(origin.x()) + sin * f64::from(spread_xz);
        let x1 = f64::from(origin.x()) - sin * f64::from(spread_xz);
        let z0 = f64::from(origin.z()) + cos * f64::from(spread_xz);
        let z1 = f64::from(origin.z()) - cos * f64::from(spread_xz);
        let y0 = f64::from(origin.y() + random.next_i32_bounded(3) - 2);
        let y1 = f64::from(origin.y() + random.next_i32_bounded(3) - 2);
        let x_start = origin.x() - spread_xz_ceil - max_radius;
        let y_start = origin.y() - 2 - max_radius;
        let z_start = origin.z() - spread_xz_ceil - max_radius;
        let size_xz = 2 * (spread_xz_ceil + max_radius);
        let size_y = 2 * (2 + max_radius);

        // The probe is a pure boolean OR: `do_place_ore` is called with the same
        // arguments whichever column satisfies it, and no randomness is drawn
        // inside the loop. So the answer is exactly
        // `y_start <= max(height_at)` over the square, and a bound over a whole
        // chunk can reject all of that chunk's columns at once.
        //
        // Worth doing because of where the cost sits. Measured over a 201x201
        // pregeneration: 247.9 probe loops per chunk at 71.6 `height_at` calls
        // each, 17,737 calls per chunk in all, and **63.3% of loops fall
        // through** -- scanning every column of the square and finding nothing.
        // Those fall-through loops are ~95% of the calls, and they are exactly
        // what a per-chunk bound deletes.
        //
        // The bound only ever *rejects*. A chunk that passes is still scanned
        // column by column, because its highest column may lie outside the
        // probe square. Chunks with no bound available (a `ReadOnlyFull`
        // neighbour, whose heights are delegated and never cached) are scanned
        // exactly as before.
        let x_end = x_start + size_xz;
        let z_end = z_start + size_xz;
        let chunk_x0 = SectionPos::block_to_section_coord(x_start);
        let chunk_x1 = SectionPos::block_to_section_coord(x_end);
        let chunk_z0 = SectionPos::block_to_section_coord(z_start);
        let chunk_z1 = SectionPos::block_to_section_coord(z_end);

        for chunk_x in chunk_x0..=chunk_x1 {
            for chunk_z in chunk_z0..=chunk_z1 {
                if let Some(max) =
                    region.worldgen_height_max(HeightmapType::OceanFloorWg, chunk_x, chunk_z)
                    && max < y_start
                {
                    continue;
                }

                let x_lo = x_start.max(chunk_x << 4);
                let x_hi = x_end.min((chunk_x << 4) + 15);
                let z_lo = z_start.max(chunk_z << 4);
                let z_hi = z_end.min((chunk_z << 4) + 15);
                for x_probe in x_lo..=x_hi {
                    for z_probe in z_lo..=z_hi {
                        if y_start
                            <= region.height_at(HeightmapType::OceanFloorWg, x_probe, z_probe)
                        {
                            return Self::do_place_ore(
                        region, registry, random, config, x0, x1, z0, z1, y0, y1, x_start, y_start,
                        z_start, size_xz, size_y,
                    );
                        }
                    }
                }
            }
        }

        false
    }

    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "mirrors vanilla ore vein placement inputs"
    )]
    pub(in crate::worldgen::feature) fn do_place_ore(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        x0: f64,
        x1: f64,
        z0: f64,
        z1: f64,
        y0: f64,
        y1: f64,
        x_start: i32,
        y_start: i32,
        z_start: i32,
        size_xz: i32,
        size_y: i32,
    ) -> bool {
        let Ok(size) = usize::try_from(config.size) else {
            return false;
        };
        let mut vein_nodes = SmallVec::<[[f64; 4]; 32]>::from_elem([0.0; 4], size);

        for i in 0..size {
            let step = i as f32 / config.size as f32;
            let size_factor = random.next_f64() * f64::from(config.size) / 16.0;
            let radius_wave = trig::sin(f64::from(PI * step)) + 1.0;
            let radius = f64::from(radius_wave) * size_factor + 1.0;
            vein_nodes[i] = [
                lerp(f64::from(step), x0, x1),
                lerp(f64::from(step), y0, y1),
                lerp(f64::from(step), z0, z1),
                radius / 2.0,
            ];
        }

        for i1 in 0..size.saturating_sub(1) {
            if vein_nodes[i1][3] <= 0.0 {
                continue;
            }

            for i2 in i1 + 1..size {
                if vein_nodes[i2][3] <= 0.0 {
                    continue;
                }

                let dx = vein_nodes[i1][0] - vein_nodes[i2][0];
                let dy = vein_nodes[i1][1] - vein_nodes[i2][1];
                let dz = vein_nodes[i1][2] - vein_nodes[i2][2];
                let dr = vein_nodes[i1][3] - vein_nodes[i2][3];
                if dr * dr > dx * dx + dy * dy + dz * dz {
                    if dr > 0.0 {
                        vein_nodes[i2][3] = -1.0;
                    } else {
                        vein_nodes[i1][3] = -1.0;
                    }
                }
            }
        }

        let Some(search_volume) = OreSearchVolume::new(size_xz, size_y) else {
            return false;
        };
        let profile = OreFeatureProfile::new(config.size);
        let mut placed = 0_u64;
        let mut tested = OreTestedPositions::with_capacity(search_volume.tested_position_count);
        let targets = ResolvedOreTargets::from_config(registry, config);
        let batch_no_air_exposure = config.discard_chance_on_air_exposure <= 0.0;
        let mut pending_no_air_sections = SmallVec::<[PendingOreSection; 8]>::new();
        let min_y = region.min_y();
        let height = region.height();

        {
            let mut sections = region.bulk_section_access_for_ore(profile.stats());
            let candidate_started_at = profile.stats().map(|_| Instant::now());

            placed += if profile.stats().is_some() {
                Self::collect_ore_candidates::<true>(
                    &mut sections,
                    registry,
                    random,
                    config,
                    &targets,
                    search_volume,
                    vein_nodes,
                    x_start,
                    y_start,
                    z_start,
                    min_y,
                    height,
                    batch_no_air_exposure,
                    &mut tested,
                    &mut pending_no_air_sections,
                )
            } else {
                Self::collect_ore_candidates::<false>(
                    &mut sections,
                    registry,
                    random,
                    config,
                    &targets,
                    search_volume,
                    vein_nodes,
                    x_start,
                    y_start,
                    z_start,
                    min_y,
                    height,
                    batch_no_air_exposure,
                    &mut tested,
                    &mut pending_no_air_sections,
                )
            };
            if let Some(started_at) = candidate_started_at
                && let Some(stats) = profile.stats()
            {
                stats
                    .borrow_mut()
                    .record_candidate_time(started_at.elapsed());
            }

            if batch_no_air_exposure {
                let batch_apply_started_at = profile.stats().map(|_| Instant::now());
                let mut memo = OreReplacementMemo::new();
                for pending_section in &pending_no_air_sections {
                    placed += sections.replace_ore_target_block_states_in_section(
                        pending_section.key.chunk_x,
                        pending_section.key.chunk_z,
                        pending_section.key.section_index,
                        &pending_section.positions,
                        |block_state| memo.replacement(&targets, registry, block_state),
                    );
                }
                if let Some(started_at) = batch_apply_started_at
                    && let Some(stats) = profile.stats()
                {
                    stats
                        .borrow_mut()
                        .record_batch_apply_time(started_at.elapsed());
                }
            }
        }

        profile.finish(placed);
        placed > 0
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "keeps the vanilla ore candidate loop monomorphized without hiding state"
    )]
    fn collect_ore_candidates<const PROFILE: bool>(
        sections: &mut WorldGenBulkSectionAccess<'_, '_, '_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        targets: &ResolvedOreTargets,
        search_volume: OreSearchVolume,
        vein_nodes: SmallVec<[[f64; 4]; 32]>,
        x_start: i32,
        y_start: i32,
        z_start: i32,
        min_y: i32,
        height: i32,
        batch_no_air_exposure: bool,
        tested: &mut OreTestedPositions,
        pending_no_air_sections: &mut SmallVec<[PendingOreSection; 8]>,
    ) -> u64 {
        let mut placed = 0_u64;

        for node in vein_nodes {
            let radius = node[3];
            if radius < 0.0 {
                continue;
            }

            let x_min = fast_floor(node[0] - radius).max(x_start);
            let z_min = fast_floor(node[2] - radius).max(z_start);
            let x_max = fast_floor(node[0] + radius).max(x_min);
            let z_max = fast_floor(node[2] + radius).max(z_min);
            let raw_y_min = fast_floor(node[1] - radius).max(y_start);
            let raw_y_max = fast_floor(node[1] + radius).max(raw_y_min);
            let y_min = raw_y_min.max(min_y);
            let y_max = raw_y_max.min(min_y + height - 1);
            if y_min > y_max {
                continue;
            }

            for x in x_min..=x_max {
                let x_offset = i64::from(x) - i64::from(x_start);
                let x_distance = (f64::from(x) + 0.5 - node[0]) / radius;
                let x_distance_squared = x_distance * x_distance;
                if x_distance_squared >= 1.0 {
                    continue;
                }

                for y in y_min..=y_max {
                    let y_offset = i64::from(y) - i64::from(y_start);
                    let y_distance = (f64::from(y) + 0.5 - node[1]) / radius;
                    let x_y_distance_squared = x_distance_squared + y_distance * y_distance;
                    if x_y_distance_squared >= 1.0 {
                        continue;
                    }

                    let accepted_z =
                        OreNodeZScan::new(z_min, z_max, node[2], radius, x_y_distance_squared);
                    for z in accepted_z {
                        let z_offset = i64::from(z) - i64::from(z_start);

                        if PROFILE {
                            sections.record_ore_candidate_position();
                        }
                        let Some(tested_index) =
                            search_volume.index_from_offsets(x_offset, y_offset, z_offset)
                        else {
                            continue;
                        };
                        if tested.insert(tested_index) {
                            if PROFILE {
                                sections.record_ore_unique_position();
                            }
                            if batch_no_air_exposure {
                                let section_key =
                                    PendingOreSectionKey::from_in_height_coords(min_y, x, y, z);
                                let Some(pos) = PackedSectionBlockPos::from_local_xyz(
                                    (x & 15) as u8,
                                    (y & 15) as u8,
                                    (z & 15) as u8,
                                ) else {
                                    panic!("masked ore section-local position escaped section");
                                };
                                push_pending_ore_position(
                                    pending_no_air_sections,
                                    section_key,
                                    pos,
                                );
                            } else {
                                let pos = BlockPos::new(x, y, z);
                                if sections.can_write_to_pos(pos) {
                                    if PROFILE {
                                        sections.record_ore_write_allowed_position();
                                    }
                                    if Self::try_place_ore_block_in_bulk(
                                        sections, registry, random, config, targets, pos,
                                    ) {
                                        placed += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        placed
    }

    pub(in crate::worldgen::feature) fn place_scattered_ore_feature(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        origin: BlockPos,
    ) -> bool {
        assert!(
            config.size >= 0,
            "scattered ore size {} is negative",
            config.size
        );

        let targets = ResolvedOreTargets::from_config(registry, config);
        let tries = random.next_i32_bounded(config.size + 1);
        for i in 0..tries {
            let max_distance = i.min(7);
            let pos = origin.offset(
                Self::random_scattered_ore_offset(random, max_distance),
                Self::random_scattered_ore_offset(random, max_distance),
                Self::random_scattered_ore_offset(random, max_distance),
            );
            let _ =
                Self::try_place_resolved_ore_block(region, registry, random, config, &targets, pos);
        }

        true
    }

    pub(in crate::worldgen::feature) fn random_scattered_ore_offset(
        random: &mut WorldgenRandom,
        max_distance: i32,
    ) -> i32 {
        Self::java_round_f32((random.next_f32() - random.next_f32()) * max_distance as f32)
    }

    pub(in crate::worldgen::feature) fn java_round_f32(value: f32) -> i32 {
        (value + 0.5).floor() as i32
    }

    fn try_place_resolved_ore_block(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        targets: &ResolvedOreTargets,
        pos: BlockPos,
    ) -> bool {
        let block_state = region.block_state(pos);
        let block_id = ResolvedOreTargets::block_id_for_state(registry, block_state);
        for target in targets.iter() {
            if Self::can_place_resolved_ore(region, registry, random, config, target, pos, block_id)
            {
                return region.set_block_state(pos, target.state, UpdateFlags::UPDATE_CLIENTS);
            }
        }

        false
    }

    fn try_place_ore_block_in_bulk(
        sections: &mut WorldGenBulkSectionAccess<'_, '_, '_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        targets: &ResolvedOreTargets,
        pos: BlockPos,
    ) -> bool {
        if config.discard_chance_on_air_exposure <= 0.0 {
            return sections.replace_ore_target_block_state(pos, |block_state| {
                targets.matching_replacement(registry, block_state)
            });
        }

        let block_state = sections.ore_target_block_state(pos);
        let block_id = ResolvedOreTargets::block_id_for_state(registry, block_state);
        for target in targets.iter() {
            if Self::can_place_resolved_ore_in_bulk(
                sections, registry, random, config, target, pos, block_id,
            ) {
                return sections.set_block_state(pos, target.state);
            }
        }

        false
    }

    fn can_place_resolved_ore(
        region: &WorldGenRegion<'_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        target: &ResolvedOreTarget,
        pos: BlockPos,
        block_id: usize,
    ) -> bool {
        if !target.matches_block_id(block_id) {
            return false;
        }

        if Self::should_skip_air_check(random, config.discard_chance_on_air_exposure) {
            return true;
        }

        !Self::is_adjacent_to_air(region, registry, pos)
    }

    fn can_place_resolved_ore_in_bulk(
        sections: &mut WorldGenBulkSectionAccess<'_, '_, '_>,
        registry: &Registry,
        random: &mut WorldgenRandom,
        config: &OreConfiguration,
        target: &ResolvedOreTarget,
        pos: BlockPos,
        block_id: usize,
    ) -> bool {
        if !target.matches_block_id(block_id) {
            return false;
        }

        if Self::should_skip_air_check(random, config.discard_chance_on_air_exposure) {
            return true;
        }

        !Self::is_adjacent_to_air_in_bulk(sections, registry, pos)
    }

    pub(in crate::worldgen::feature) fn should_skip_air_check(
        random: &mut WorldgenRandom,
        discard_chance_on_air_exposure: f32,
    ) -> bool {
        if discard_chance_on_air_exposure <= 0.0 {
            true
        } else if discard_chance_on_air_exposure >= 1.0 {
            false
        } else {
            random.next_f32() >= discard_chance_on_air_exposure
        }
    }

    pub(in crate::worldgen::feature) fn is_adjacent_to_air(
        region: &WorldGenRegion<'_>,
        registry: &Registry,
        pos: BlockPos,
    ) -> bool {
        Direction::ALL.into_iter().any(|direction| {
            let neighbor = region.block_state(pos.relative(direction));
            Self::is_air_block_state(registry, neighbor)
        })
    }

    pub(in crate::worldgen::feature) fn is_adjacent_to_air_in_bulk(
        sections: &mut WorldGenBulkSectionAccess<'_, '_, '_>,
        registry: &Registry,
        pos: BlockPos,
    ) -> bool {
        Direction::ALL.into_iter().any(|direction| {
            let neighbor = sections.ore_neighbor_block_state(pos.relative(direction));
            Self::is_air_block_state(registry, neighbor)
        })
    }

    pub(in crate::worldgen::feature) fn is_air_block_state(
        registry: &Registry,
        state: BlockStateId,
    ) -> bool {
        let Some(block) = registry.blocks.by_state_id(state) else {
            panic!("feature received invalid block state id {}", state.0);
        };
        block.config.is_air
    }
}

struct OreTestedPositions {
    words: SmallVec<[u64; 16]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PendingOreSectionKey {
    chunk_x: i32,
    chunk_z: i32,
    section_index: usize,
}

struct PendingOreSection {
    key: PendingOreSectionKey,
    positions: SmallVec<[PackedSectionBlockPos; 256]>,
}

struct ResolvedOreTargets {
    targets: SmallVec<[ResolvedOreTarget; 2]>,
}

/// Direct-mapped memo over [`ResolvedOreTargets::matching_replacement`].
///
/// The batch flush asks that question once per unique candidate position -
/// 7,289 times per chunk - but it is a pure function of the block state, and a
/// vein only ever meets a handful of them (the stone family the tags name, plus
/// whatever air or water the caves left behind). Answering from a
/// two-cache-line table skips re-reading the 224 KiB `state_to_block_id` array,
/// the `SmallVec` inline-vs-heap dispatch and the per-target tag scan.
///
/// Measured over 90,601 chunks: 96.49% of lookups hit, so the misses are
/// essentially all compulsory - about 2.5 distinct states per vein, and the
/// table is rebuilt per vein. A larger table cannot improve on that.
struct OreReplacementMemo {
    /// `(state id, answer)` per slot; [`Self::EMPTY`] marks a slot with no answer yet.
    entries: [(u32, Option<BlockStateId>); Self::SLOTS],
}

impl OreReplacementMemo {
    /// Power of two so the slot index is a mask, not a division.
    const SLOTS: usize = 16;
    /// Out of range for any `BlockStateId`, so it can never alias a real key.
    const EMPTY: u32 = u32::MAX;

    const fn new() -> Self {
        Self {
            entries: [(Self::EMPTY, None); Self::SLOTS],
        }
    }

    #[inline]
    fn replacement(
        &mut self,
        targets: &ResolvedOreTargets,
        registry: &Registry,
        state: BlockStateId,
    ) -> Option<BlockStateId> {
        let key = u32::from(state.0);
        let entry = &mut self.entries[(state.0 as usize) & (Self::SLOTS - 1)];
        if entry.0 == key {
            return entry.1;
        }

        let replacement = targets.matching_replacement(registry, state);
        *entry = (key, replacement);
        replacement
    }
}

struct ResolvedOreTarget {
    matcher: ResolvedOreRuleTest,
    state: BlockStateId,
}

enum ResolvedOreRuleTest {
    Block(usize),
    Tag(SmallVec<[usize; 8]>),
}

#[derive(Clone, Copy)]
struct OreSearchVolume {
    size_xz: i64,
    size_xz_y: i64,
    tested_position_count: usize,
}

impl OreSearchVolume {
    fn new(size_xz: i32, size_y: i32) -> Option<Self> {
        let size_xz = i64::from(size_xz);
        let size_y = i64::from(size_y);
        if size_xz <= 0 || size_y <= 0 {
            return None;
        }

        let size_xz_y = size_xz.checked_mul(size_y)?;
        let tested_position_count = usize::try_from(size_xz_y.checked_mul(size_xz)?).ok()?;
        Some(Self {
            size_xz,
            size_xz_y,
            tested_position_count,
        })
    }

    #[inline]
    fn index_from_offsets(self, x_offset: i64, y_offset: i64, z_offset: i64) -> Option<usize> {
        if x_offset < 0 || y_offset < 0 || z_offset < 0 {
            return None;
        }

        // Matches vanilla OreFeature's BitSet index layout.
        let index = x_offset + y_offset * self.size_xz + z_offset * self.size_xz_y;
        usize::try_from(index).ok()
    }
}

/// Walks the `z` positions of one ore vein node that pass the sphere test,
/// eight at a time.
///
/// The innermost ore loop evaluates
/// `x_y_distance_squared + z_distance * z_distance < 1.0` once per `z` in the
/// node's bounding span, and 43.4% of those iterations reject and do nothing
/// (measured over 90,601 chunks: 46,073 z-iterations per chunk against 26,077
/// candidates). Every rejection still pays a convert, an add, a subtract, a
/// divide, a multiply, an add, a compare and the range bookkeeping. That
/// predicate is the largest single block in ore placement -- 34% of its cycles
/// and 10.3% of the whole program's mispredicted branches.
///
/// The scan bounds are unchanged; this evaluates the same expression for eight
/// consecutive `z` per pass and yields the accepted lanes in ascending `z`.
///
/// The result is bit-identical to the scalar form. `z` is an `i32` and `0.5` is
/// a power of two, so `z + 0.5` is exact in every lane; subtract, divide,
/// multiply and add are IEEE-754 basic operations that round per lane, with no
/// reassociation and no fused multiply-add. `simd_ge` is false for NaN exactly
/// as `>=` is, so negating it reproduces `if ... >= 1.0 { continue; }` for a NaN
/// distance too. Ascending order matters because the non-batch body draws from
/// the seeded worldgen RNG.
struct OreNodeZScan {
    /// First `z` of the next block of lanes to evaluate.
    cursor: i32,
    /// Count of `z` from `cursor` to `z_max` inclusive still to evaluate.
    remaining: u64,
    /// First `z` of the block whose accepted lanes are in `accepted`.
    block_first: i32,
    /// Accepted lanes of the current block, one bit per lane, lowest `z` first.
    accepted: u64,
    node_z: f64x8,
    radius: f64x8,
    x_y_distance_squared: f64x8,
}

impl OreNodeZScan {
    /// Lane index as a float, so a block's `z` values are one broadcast add.
    const LANE_INDEX: f64x8 = f64x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    fn new(z_min: i32, z_max: i32, node_z: f64, radius: f64, x_y_distance_squared: f64) -> Self {
        Self {
            cursor: z_min,
            remaining: u64::from(z_max.abs_diff(z_min)) + 1,
            block_first: z_min,
            accepted: 0,
            node_z: f64x8::splat(node_z),
            radius: f64x8::splat(radius),
            x_y_distance_squared: f64x8::splat(x_y_distance_squared),
        }
    }

    /// Evaluates the sphere test for the next eight `z`, dropping lanes past
    /// `z_max`.
    fn evaluate_next_lanes(&mut self) {
        let z = f64x8::splat(f64::from(self.cursor)) + Self::LANE_INDEX;
        let z_distance = (z + f64x8::splat(0.5) - self.node_z) / self.radius;
        let rejected =
            (self.x_y_distance_squared + z_distance * z_distance).simd_ge(f64x8::splat(1.0));
        let lanes = f64x8::LEN as u64;
        let in_span = if self.remaining >= lanes {
            u64::MAX
        } else {
            (1 << self.remaining) - 1
        };

        self.accepted = (!rejected).to_bitmask() & in_span;
        self.block_first = self.cursor;
        self.cursor = self.cursor.wrapping_add(f64x8::LEN as i32);
        self.remaining -= lanes.min(self.remaining);
    }
}

impl Iterator for OreNodeZScan {
    type Item = i32;

    #[expect(
        clippy::inline_always,
        reason = "runs 46,073 times per chunk; out of line every accepted \
                  position would pay a call and reload the three broadcasts"
    )]
    #[inline(always)]
    fn next(&mut self) -> Option<i32> {
        loop {
            if self.accepted != 0 {
                let lane = self.accepted.trailing_zeros();
                self.accepted &= self.accepted - 1;
                return Some(self.block_first + lane.cast_signed());
            }
            if self.remaining == 0 {
                return None;
            }
            self.evaluate_next_lanes();
        }
    }
}

impl ResolvedOreTargets {
    fn from_config(registry: &Registry, config: &OreConfiguration) -> Self {
        let mut targets = SmallVec::with_capacity(config.targets.len());
        for target in &config.targets {
            let matcher = match &target.target {
                RuleTest::BlockMatch { block } => ResolvedOreRuleTest::Block(block.id()),
                RuleTest::TagMatch { tag } => {
                    let block_ids = registry
                        .blocks
                        .iter_tag(tag)
                        .map(steel_registry::RegistryEntry::id)
                        .collect();
                    ResolvedOreRuleTest::Tag(block_ids)
                }
            };
            let state = WorldgenStateResolver::feature_block_state_from_data(
                registry,
                &target.state,
                "ore feature",
            );
            targets.push(ResolvedOreTarget { matcher, state });
        }

        Self { targets }
    }

    fn iter(&self) -> impl Iterator<Item = &ResolvedOreTarget> {
        self.targets.iter()
    }

    fn matching_replacement(
        &self,
        registry: &Registry,
        state: BlockStateId,
    ) -> Option<BlockStateId> {
        let block_id = Self::block_id_for_state(registry, state);
        self.targets
            .iter()
            .find_map(|target| target.matches_block_id(block_id).then_some(target.state))
    }

    fn block_id_for_state(registry: &Registry, state: BlockStateId) -> usize {
        let Some(&block_id) = registry.blocks.state_to_block_id.get(state.0 as usize) else {
            panic!("ore feature received invalid block state id {}", state.0);
        };
        block_id
    }
}

impl ResolvedOreTarget {
    fn matches_block_id(&self, block_id: usize) -> bool {
        match &self.matcher {
            ResolvedOreRuleTest::Block(target_block_id) => block_id == *target_block_id,
            ResolvedOreRuleTest::Tag(block_ids) => block_ids.contains(&block_id),
        }
    }
}

impl PendingOreSectionKey {
    const fn from_in_height_coords(min_y: i32, x: i32, y: i32, z: i32) -> Self {
        Self {
            chunk_x: SectionPos::block_to_section_coord(x),
            chunk_z: SectionPos::block_to_section_coord(z),
            section_index: ((y - min_y) / 16) as usize,
        }
    }
}

fn push_pending_ore_position(
    sections: &mut SmallVec<[PendingOreSection; 8]>,
    key: PendingOreSectionKey,
    pos: PackedSectionBlockPos,
) {
    if let Some(section) = sections.last_mut()
        && section.key == key
    {
        section.positions.push(pos);
        return;
    }

    if let Some(section) = sections.iter_mut().find(|section| section.key == key) {
        section.positions.push(pos);
        return;
    }

    sections.push(PendingOreSection {
        key,
        positions: smallvec::smallvec![pos],
    });
}

impl OreTestedPositions {
    fn with_capacity(bit_count: usize) -> Self {
        Self {
            words: smallvec::smallvec![0; bit_count.div_ceil(u64::BITS as usize)],
        }
    }

    /// Records `index`, returning whether it had not been seen before.
    ///
    /// Kept inlineable, which it was not: the grow branch below pulled `SmallVec`'s
    /// reallocation into the body and the whole probe compiled to an out-of-line
    /// call with seven callee pushes, so every candidate position paid a call,
    /// ten spill stores and twelve reloads -- including reloading the loop's
    /// floating-point constants, since all of xmm is caller-saved -- for what is
    /// otherwise a shift and a test-and-set. The ore candidate loop runs this
    /// 578 million times over a 601x601 pregeneration, roughly 3.7 times per
    /// unique position.
    ///
    /// The grow path is unreachable in practice: the bitset is allocated for the
    /// whole search volume in [`Self::with_capacity`], and every index comes from
    /// [`OreSearchVolume::index_from_offsets`], which returns `None` outside it. It
    /// stays for safety, out of line and marked cold.
    #[expect(
        clippy::inline_always,
        reason = "measured: without it the probe compiles to an out-of-line call \
                  paid once per candidate position, 578M times per 601x601 run"
    )]
    #[inline(always)]
    fn insert(&mut self, index: usize) -> bool {
        let word_index = index / u64::BITS as usize;
        let mask = 1_u64 << (index % u64::BITS as usize);

        let Some(word) = self.words.get_mut(word_index) else {
            return self.insert_beyond_capacity(word_index, mask);
        };
        if *word & mask != 0 {
            return false;
        }

        *word |= mask;
        true
    }

    #[cold]
    #[inline(never)]
    fn insert_beyond_capacity(&mut self, word_index: usize, mask: u64) -> bool {
        self.words.resize(word_index + 1, 0);
        let word = &mut self.words[word_index];
        if *word & mask != 0 {
            return false;
        }

        *word |= mask;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{OreSearchVolume, OreTestedPositions};

    #[test]
    fn ore_tested_position_index_matches_vanilla_layout() {
        let volume = OreSearchVolume::new(4, 6);
        assert_eq!(
            volume.and_then(|volume| volume.index_from_offsets(2, 3, 1)),
            Some(38)
        );
    }

    #[test]
    fn ore_tested_position_index_keeps_vanilla_inclusive_edge_layout() {
        let volume = OreSearchVolume::new(4, 6);
        assert_eq!(
            volume.and_then(|volume| volume.index_from_offsets(4, 0, 0)),
            Some(4)
        );
        assert_eq!(
            volume.and_then(|volume| volume.index_from_offsets(0, 1, 0)),
            Some(4)
        );
    }

    #[test]
    fn ore_tested_positions_deduplicate_and_grow() {
        let mut tested = OreTestedPositions::with_capacity(1);
        assert!(tested.insert(0));
        assert!(!tested.insert(0));
        assert!(tested.insert(130));
        assert!(!tested.insert(130));
    }
}
