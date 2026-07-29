//! End-to-end check that the headless engine produces real terrain.
//!
//! Uses the Rust API rather than the C ABI; the ABI is exercised separately from
//! the host side.

use steel_core::chunk::chunk_access::ChunkStatus;
use steel_utils::{BlockPos, ChunkPos, Identifier};
use steel_worldgen_ffi::engine::{GenerationWorld, WorldSpec};

/// Opens the vanilla overworld at a fixed seed.
fn overworld(seed: i64) -> GenerationWorld {
    let spec = WorldSpec {
        generator: "minecraft:overworld"
            .parse::<Identifier>()
            .expect("overworld identifier should parse"),
        seed,
        threads: 4,
    };
    GenerationWorld::open(&spec).expect("overworld world should open")
}

#[test]
fn generates_overworld_terrain_through_features() {
    let world = overworld(1234);
    let center = ChunkPos::new(0, 0);

    let holders = world
        .generate(&[center], ChunkStatus::Features)
        .expect("generation should succeed");

    assert_eq!(holders.len(), 1, "one holder per requested position");

    let chunk = holders[0]
        .try_chunk(ChunkStatus::Features)
        .expect("chunk should have reached Features");

    // Bedrock floor is the cheapest unambiguous signal that noise ran: an empty
    // or stubbed chunk would be air all the way down.
    let mut solid = 0_u32;
    for y in -64..320 {
        for x in 0..16 {
            for z in 0..16 {
                let state = chunk.get_block_state(BlockPos::new(x, y, z));
                if state.0 != 0 {
                    solid += 1;
                }
            }
        }
    }

    assert!(
        solid > 10_000,
        "expected a substantially solid chunk, got {solid} non-air blocks"
    );
}

#[test]
fn generation_is_deterministic_for_a_seed() {
    let center = ChunkPos::new(5, -3);

    let sample = |seed: i64| {
        let world = overworld(seed);
        let holders = world
            .generate(&[center], ChunkStatus::Features)
            .expect("generation should succeed");
        let chunk = holders[0]
            .try_chunk(ChunkStatus::Features)
            .expect("chunk should have reached Features");

        (-64..320)
            .step_by(8)
            .map(|y| chunk.get_block_state(BlockPos::new(80 + 7, y, -48 + 9)).0)
            .collect::<Vec<u16>>()
    };

    let first = sample(99);
    let second = sample(99);
    let other = sample(100);

    assert_eq!(first, second, "same seed must produce the same column");
    assert_ne!(
        first, other,
        "different seeds should not produce an identical column"
    );
}
