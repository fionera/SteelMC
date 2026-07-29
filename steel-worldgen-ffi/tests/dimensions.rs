//! Drives every supported dimension to `Full`, the deepest status the host can
//! ask for.
//!
//! `Full` matters beyond coverage: it is where `InitializeLight`, `Light` and the
//! `ProtoChunk` to `LevelChunk` promotion run, and where `apply_step` has to
//! reserve the light work window.

use steel_core::chunk::chunk_access::ChunkStatus;
use steel_utils::{BlockPos, ChunkPos, Identifier};
use steel_worldgen_ffi::engine::{GenerationWorld, WorldSpec};

/// Counts non-air blocks in a chunk that has reached `status`.
fn solid_blocks(world: &GenerationWorld, center: ChunkPos, status: ChunkStatus) -> u32 {
    let holders = world
        .generate(&[center], status)
        .expect("generation should succeed");
    let chunk = holders[0]
        .try_chunk(status)
        .expect("chunk should have reached the requested status");

    let min_y = world.min_y();
    let max_y = min_y + world.height();

    let mut solid = 0;
    for y in min_y..max_y {
        for x in 0..16 {
            for z in 0..16 {
                if chunk.get_block_state(BlockPos::new(x, y, z)).0 != 0 {
                    solid += 1;
                }
            }
        }
    }
    solid
}

fn open(generator: &str, seed: i64) -> GenerationWorld {
    let spec = WorldSpec {
        generator: generator
            .parse::<Identifier>()
            .expect("generator identifier should parse"),
        seed,
        threads: 4,
    };
    GenerationWorld::open(&spec).unwrap_or_else(|err| panic!("{generator} should open: {err}"))
}

#[test]
fn overworld_reaches_full() {
    let world = open("minecraft:overworld", 4242);
    let solid = solid_blocks(&world, ChunkPos::new(0, 0), ChunkStatus::Full);
    assert!(solid > 10_000, "overworld chunk was mostly air: {solid}");
}

#[test]
fn nether_reaches_full() {
    let world = open("minecraft:the_nether", 4242);
    let solid = solid_blocks(&world, ChunkPos::new(0, 0), ChunkStatus::Full);
    assert!(solid > 10_000, "nether chunk was mostly air: {solid}");
}

#[test]
fn end_reaches_full() {
    // The End is mostly void away from the central island, so generate at the
    // origin where the main island is.
    let world = open("minecraft:the_end", 4242);
    let solid = solid_blocks(&world, ChunkPos::new(0, 0), ChunkStatus::Full);
    assert!(solid > 1_000, "end main island was mostly air: {solid}");
}

#[test]
fn multi_chunk_batch_generates_every_requested_chunk() {
    let world = open("minecraft:overworld", 77);

    let centers: Vec<ChunkPos> = (0..3)
        .flat_map(|x| (0..3).map(move |z| ChunkPos::new(x, z)))
        .collect();

    let holders = world
        .generate(&centers, ChunkStatus::Features)
        .expect("batch generation should succeed");

    assert_eq!(holders.len(), centers.len());
    for (holder, center) in holders.iter().zip(&centers) {
        assert_eq!(
            holder.get_pos(),
            *center,
            "holders come back in request order"
        );
        assert!(
            holder.try_chunk(ChunkStatus::Features).is_some(),
            "chunk {center:?} did not reach Features"
        );
    }
}
