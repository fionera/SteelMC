//! Feature-stage structure piece placement boundary.
//!
//! Structure starts are generated before noise, but vanilla emits the piece
//! blocks during biome decoration. This module is the single dispatch point for
//! that pass; individual family placers must fill in exact vanilla behavior
//! before any payload variant starts writing blocks.

mod buried_treasure;
mod desert_pyramid;
mod fortress;
mod jungle_temple;
mod mineshaft;
mod ocean_monument;
mod pool_element;
mod ruined_portal;
mod scattered_feature;
mod stronghold;
mod swamp_hut;
mod template_piece;
mod template_processors;

use simdnbt::owned::{NbtCompound, NbtList, NbtTag};
use steel_registry::block_entity_type::BlockEntityTypeRef;
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::blocks::properties::BlockStateProperties;
use steel_registry::structure::StructureRef;
use steel_registry::template_pool::PoolElement;
use steel_registry::{Registry, vanilla_block_entity_types, vanilla_blocks};
use steel_utils::random::Random;
use steel_utils::random::worldgen_random::WorldgenRandom;
use steel_utils::{
    BlockPos, BlockStateId, BoundingBox, Direction, Identifier, PackedBlockPos, Rotation,
    types::UpdateFlags,
};

use crate::worldgen::region::WorldGenRegion;
use steel_worldgen::structure::mineshaft::MineshaftPieceKind;
use steel_worldgen::structure::{
    ProceduralPieceData, StructureMirror, StructurePiece, StructurePiecePayload,
};

pub(crate) struct StructurePiecePlacer;

impl StructurePiecePlacer {
    /// Vanilla jigsaw pool-element placement flags: `UPDATE_CLIENTS | UPDATE_KNOWN_SHAPE`.
    pub(crate) const JIGSAW_UPDATE_FLAGS: UpdateFlags =
        UpdateFlags::UPDATE_CLIENTS.union(UpdateFlags::UPDATE_KNOWN_SHAPE);
    /// Vanilla template-piece placement flags: `UPDATE_CLIENTS`.
    pub(crate) const TEMPLATE_UPDATE_FLAGS: UpdateFlags = UpdateFlags::UPDATE_CLIENTS;

    /// Whether placing this piece touches state shared with the other chunks
    /// that decorate the same [`StructureStart`](steel_worldgen::structure::StructureStart).
    ///
    /// Feature-stage placement runs once per decorating chunk whose writable box
    /// intersects a piece, so one start is visited by many generation threads.
    /// Most families only read the piece and write blocks into the visiting
    /// chunk's own clip, which is disjoint between visitors, and can therefore
    /// run under a shared borrow of the source chunk's start map. These cannot:
    ///
    /// * `Template`, `BuriedTreasure`, `DesertPyramid`, `JungleTemple` and
    ///   `SwampHut` write [`StructurePiece::bounding_box`] and their payload's
    ///   persisted ground-height adjustment.
    /// * A mineshaft spider corridor reads and writes vanilla's
    ///   `hasPlacedSpider` one-shot flag, and that read gates an RNG draw.
    /// * Stronghold and nether fortress payloads carry vanilla's one-shot chest
    ///   and spawner flags. Only two of their piece kinds actually use one, but
    ///   both dispatches would have to be duplicated to separate them, for well
    ///   under 1% of the measured lock hold, so the families stay exclusive.
    /// * A jigsaw piece whose pool element places a *feature* writes through the
    ///   region without clipping to the decorating chunk, so two visitors would
    ///   no longer be writing disjoint block ranges.
    ///
    /// Every payload this reports `true` for is also one the shared dispatch in
    /// [`Self::try_place_piece_shared`] declines, so a caller that honours this
    /// never reaches that function's `unreachable!`.
    pub(crate) fn piece_needs_exclusive_placement(piece: &StructurePiece) -> bool {
        match &piece.payload {
            StructurePiecePayload::Jigsaw(data) => {
                Self::pool_element_places_feature(&data.pool_element)
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::Mineshaft(data)) => matches!(
                data.kind,
                MineshaftPieceKind::Corridor {
                    spider_corridor: true,
                    ..
                }
            ),
            StructurePiecePayload::Template(_)
            | StructurePiecePayload::Procedural(
                ProceduralPieceData::BuriedTreasure
                | ProceduralPieceData::DesertPyramid(_)
                | ProceduralPieceData::JungleTemple(_)
                | ProceduralPieceData::NetherFortress(_)
                | ProceduralPieceData::Stronghold(_)
                | ProceduralPieceData::SwampHut(_),
            ) => true,
            StructurePiecePayload::Procedural(
                ProceduralPieceData::OceanMonument(_) | ProceduralPieceData::Unimplemented,
            ) => false,
        }
    }

    /// Whether the element tree places a feature rather than a template.
    ///
    /// Template placement clips every write to the settings' bounding box, which
    /// feature-stage placement sets to the decorating chunk. A feature element
    /// runs an ordinary placed feature instead, which writes anywhere the
    /// generation step's write radius allows.
    fn pool_element_places_feature(element: &PoolElement) -> bool {
        match element {
            PoolElement::Feature { .. } => true,
            PoolElement::List { elements, .. } => {
                elements.iter().any(Self::pool_element_places_feature)
            }
            PoolElement::Single { .. } | PoolElement::LegacySingle { .. } | PoolElement::Empty => {
                false
            }
        }
    }

    /// Places one already-clipped structure piece.
    ///
    /// Returns whether the vanilla placement call succeeded. Later milestones
    /// must implement each remaining payload variant completely before it can
    /// return `true`.
    pub(crate) fn place_piece(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        piece: &mut StructurePiece,
        reference_pos: BlockPos,
        clip: BoundingBox,
        random: &mut WorldgenRandom,
        biome_zoom_seed: i64,
    ) -> bool {
        if let Some(placed) = Self::try_place_piece_shared(
            region,
            registry,
            piece,
            reference_pos,
            clip,
            random,
            biome_zoom_seed,
        ) {
            return placed;
        }

        let mut piece_bounding_box = piece.bounding_box;
        let piece_orientation = piece.orientation;
        let placed = match &mut piece.payload {
            StructurePiecePayload::Template(data) => Self::place_template_piece(
                region,
                registry,
                data,
                &mut piece_bounding_box,
                reference_pos,
                clip,
                random,
            ),
            StructurePiecePayload::Procedural(ProceduralPieceData::Mineshaft(data)) => {
                Self::place_mineshaft_spider_corridor(
                    region,
                    registry,
                    piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                    biome_zoom_seed,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::NetherFortress(data)) => {
                Self::place_nether_fortress_piece(
                    region,
                    registry,
                    piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::Stronghold(data)) => {
                Self::place_stronghold_piece(
                    region,
                    registry,
                    piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::BuriedTreasure) => {
                Self::place_buried_treasure_piece(region, &mut piece_bounding_box, clip, random)
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::DesertPyramid(data)) => {
                Self::place_desert_pyramid_piece(
                    region,
                    registry,
                    &mut piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::JungleTemple(data)) => {
                Self::place_jungle_temple_piece(
                    region,
                    registry,
                    &mut piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::SwampHut(data)) => {
                Self::place_swamp_hut_piece(
                    region,
                    registry,
                    &mut piece_bounding_box,
                    piece_orientation,
                    data,
                    clip,
                    random,
                )
            }
            StructurePiecePayload::Jigsaw(_)
            | StructurePiecePayload::Procedural(
                ProceduralPieceData::OceanMonument(_) | ProceduralPieceData::Unimplemented,
            ) => unreachable!("shared payloads are placed by the shared dispatch above"),
        };
        piece.bounding_box = piece_bounding_box;
        placed
    }

    /// Places one already-clipped structure piece through a shared borrow.
    ///
    /// # Panics
    ///
    /// Panics if the payload needs exclusive access. Callers must skip the whole
    /// start when [`Self::piece_needs_exclusive_placement`] reports any of its
    /// pieces, which is strictly more than this dispatch declines.
    pub(crate) fn place_piece_shared(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        piece: &StructurePiece,
        reference_pos: BlockPos,
        clip: BoundingBox,
        random: &mut WorldgenRandom,
        biome_zoom_seed: i64,
    ) -> bool {
        let Some(placed) = Self::try_place_piece_shared(
            region,
            registry,
            piece,
            reference_pos,
            clip,
            random,
            biome_zoom_seed,
        ) else {
            unreachable!(
                "{} needs exclusive structure-start access",
                piece.piece_type
            )
        };
        placed
    }

    /// Placement for the payloads that only need a shared borrow of the piece.
    ///
    /// Returns `None` for the payloads that need `&mut` access, which
    /// [`Self::place_piece`] then handles.
    fn try_place_piece_shared(
        region: &mut WorldGenRegion<'_>,
        registry: &Registry,
        piece: &StructurePiece,
        reference_pos: BlockPos,
        clip: BoundingBox,
        random: &mut WorldgenRandom,
        biome_zoom_seed: i64,
    ) -> Option<bool> {
        match &piece.payload {
            StructurePiecePayload::Jigsaw(data) => Some(Self::place_pool_element(
                region,
                registry,
                &data.pool_element,
                BlockPos::new(data.position.x, data.position.y, data.position.z),
                reference_pos,
                data.rotation,
                clip,
                random,
                data.liquid_settings,
                biome_zoom_seed,
            )),
            StructurePiecePayload::Procedural(ProceduralPieceData::Mineshaft(data)) => {
                Self::try_place_mineshaft_piece(
                    region,
                    registry,
                    piece.bounding_box,
                    piece.orientation,
                    data,
                    clip,
                    random,
                    biome_zoom_seed,
                )
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::OceanMonument(data)) => {
                Some(Self::place_ocean_monument_piece(
                    region,
                    registry,
                    piece.bounding_box,
                    piece.orientation,
                    data,
                    clip,
                    random,
                ))
            }
            StructurePiecePayload::Procedural(ProceduralPieceData::Unimplemented) => Some(false),
            StructurePiecePayload::Template(_)
            | StructurePiecePayload::Procedural(
                ProceduralPieceData::BuriedTreasure
                | ProceduralPieceData::DesertPyramid(_)
                | ProceduralPieceData::JungleTemple(_)
                | ProceduralPieceData::NetherFortress(_)
                | ProceduralPieceData::Stronghold(_)
                | ProceduralPieceData::SwampHut(_),
            ) => None,
        }
    }

    pub(crate) fn after_place_structure(
        region: &mut WorldGenRegion<'_>,
        structure: StructureRef,
        pieces: &[StructurePiece],
        clip: BoundingBox,
    ) {
        if structure.structure_type == Identifier::new_static("minecraft", "desert_pyramid") {
            Self::after_place_desert_pyramid(region, pieces, clip);
        }
    }

    const VANILLA_HORIZONTAL_DIRECTIONS: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    pub(super) fn reorient_chest(
        region: &WorldGenRegion<'_>,
        pos: BlockPos,
        state: BlockStateId,
    ) -> BlockStateId {
        let mut solid_neighbor = None;

        for direction in Self::VANILLA_HORIZONTAL_DIRECTIONS {
            let relative_pos = pos.relative(direction);
            let neighbor = region.block_state(relative_pos);
            if neighbor.get_block() == &vanilla_blocks::CHEST {
                return state;
            }

            if neighbor.is_solid_render() {
                if solid_neighbor.is_some() {
                    solid_neighbor = None;
                    break;
                }
                solid_neighbor = Some(direction);
            }
        }

        if let Some(direction) = solid_neighbor {
            return state.set_value(
                &BlockStateProperties::HORIZONTAL_FACING,
                direction.opposite(),
            );
        }

        let mut lock_dir = state.get_value(&BlockStateProperties::HORIZONTAL_FACING);
        let mut relative_pos = pos.relative(lock_dir);
        if region.block_state(relative_pos).is_solid_render() {
            lock_dir = lock_dir.opposite();
            relative_pos = pos.relative(lock_dir);
        }
        if region.block_state(relative_pos).is_solid_render() {
            lock_dir = lock_dir.rotate_y_clockwise();
            relative_pos = pos.relative(lock_dir);
        }
        if region.block_state(relative_pos).is_solid_render() {
            lock_dir = lock_dir.opposite();
        }
        state.set_value(&BlockStateProperties::HORIZONTAL_FACING, lock_dir)
    }

    pub(super) fn create_loot_chest(
        region: &mut WorldGenRegion<'_>,
        clip: BoundingBox,
        random: &mut WorldgenRandom,
        pos: BlockPos,
        loot_table: &'static str,
    ) -> bool {
        if !clip.contains_blockpos(pos)
            || region.block_state(pos).get_block() == &vanilla_blocks::CHEST
        {
            return false;
        }

        let state = Self::reorient_chest(region, pos, vanilla_blocks::CHEST.default_state());
        if !region.set_block_state(pos, state, UpdateFlags::UPDATE_CLIENTS) {
            return false;
        }

        Self::set_loot_table_block_entity(
            region,
            pos,
            &vanilla_block_entity_types::CHEST,
            state,
            loot_table,
            random.next_i64(),
        )
    }

    pub(super) fn set_loot_table_block_entity(
        region: &WorldGenRegion<'_>,
        pos: BlockPos,
        block_entity_type: BlockEntityTypeRef,
        state: BlockStateId,
        loot_table: &'static str,
        seed: i64,
    ) -> bool {
        let mut nbt = NbtCompound::new();
        nbt.insert("LootTable", loot_table);
        if seed != 0 {
            nbt.insert("LootTableSeed", seed);
        }
        region.set_block_entity_data(pos, block_entity_type, state, nbt)
    }

    pub(super) fn set_spawner_entity(
        region: &WorldGenRegion<'_>,
        pos: BlockPos,
        state: BlockStateId,
        entity_id: &'static str,
    ) -> bool {
        let mut entity = NbtCompound::new();
        entity.insert("id", entity_id);

        let mut spawn_data = NbtCompound::new();
        spawn_data.insert("entity", NbtTag::Compound(entity));

        let mut nbt = NbtCompound::new();
        nbt.insert("Delay", 20_i16);
        nbt.insert("MinSpawnDelay", 200_i16);
        nbt.insert("MaxSpawnDelay", 800_i16);
        nbt.insert("SpawnCount", 4_i16);
        nbt.insert("MaxNearbyEntities", 6_i16);
        nbt.insert("RequiredPlayerRange", 16_i16);
        nbt.insert("SpawnRange", 4_i16);
        nbt.insert("SpawnData", NbtTag::Compound(spawn_data));
        nbt.insert(
            "SpawnPotentials",
            NbtTag::List(NbtList::Compound(Vec::new())),
        );

        region.set_block_entity_data(pos, &vanilla_block_entity_types::MOB_SPAWNER, state, nbt)
    }

    pub(super) fn set_brushable_loot_table(
        region: &WorldGenRegion<'_>,
        pos: BlockPos,
        state: BlockStateId,
        loot_table: &'static str,
    ) -> bool {
        Self::set_loot_table_block_entity(
            region,
            pos,
            &vanilla_block_entity_types::BRUSHABLE_BLOCK,
            state,
            loot_table,
            PackedBlockPos::from(pos).as_raw(),
        )
    }

    pub(super) const fn orientation_transform(
        orientation: Option<Direction>,
    ) -> (StructureMirror, Rotation) {
        match orientation {
            None | Some(Direction::North | Direction::Up | Direction::Down) => {
                (StructureMirror::None, Rotation::None)
            }
            Some(Direction::South) => (StructureMirror::LeftRight, Rotation::None),
            Some(Direction::West) => (StructureMirror::LeftRight, Rotation::Clockwise90),
            Some(Direction::East) => (StructureMirror::None, Rotation::Clockwise90),
        }
    }

    pub(super) fn needs_structure_shape_postprocessing(state: BlockStateId) -> bool {
        let block = state.get_block();
        block == &vanilla_blocks::NETHER_BRICK_FENCE
            || block == &vanilla_blocks::TORCH
            || block == &vanilla_blocks::WALL_TORCH
            || block == &vanilla_blocks::OAK_FENCE
            || block == &vanilla_blocks::SPRUCE_FENCE
            || block == &vanilla_blocks::DARK_OAK_FENCE
            || block == &vanilla_blocks::PALE_OAK_FENCE
            || block == &vanilla_blocks::ACACIA_FENCE
            || block == &vanilla_blocks::BIRCH_FENCE
            || block == &vanilla_blocks::JUNGLE_FENCE
            || block == &vanilla_blocks::LADDER
            || block == &vanilla_blocks::IRON_BARS
    }
}
