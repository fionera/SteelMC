//! Checks the snapshot wire format against a hand-written decoder.
//!
//! Deliberately does not reuse the encoder's own helpers: the point is to catch a
//! framing change that would silently break the Java decoder, and a test sharing
//! the encoder's assumptions cannot do that.

use steel_core::chunk::chunk_access::ChunkStatus;
use steel_utils::{ChunkPos, Identifier};
use steel_worldgen_ffi::engine::{GenerationWorld, WorldSpec};
use steel_worldgen_ffi::snapshot::{SNAPSHOT_MAGIC, SNAPSHOT_VERSION};

const SECTION_VOLUME: usize = 4096;
const BIOME_CELLS: usize = 64;

/// Reads the snapshot format independently of the encoder.
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    const fn u8(&mut self) -> u8 {
        let value = self.bytes[self.offset];
        self.offset += 1;
        value
    }

    fn u16(&mut self) -> u16 {
        let value = u16::from_le_bytes(
            self.bytes[self.offset..self.offset + 2]
                .try_into()
                .expect("two bytes"),
        );
        self.offset += 2;
        value
    }

    fn u32(&mut self) -> u32 {
        let value = u32::from_le_bytes(
            self.bytes[self.offset..self.offset + 4]
                .try_into()
                .expect("four bytes"),
        );
        self.offset += 4;
        value
    }

    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }

    fn string(&mut self) -> String {
        let length = self.u16() as usize;
        let value = String::from_utf8(self.bytes[self.offset..self.offset + length].to_vec())
            .expect("palette entries are UTF-8");
        self.offset += length;
        value
    }

    /// Reads a cell array, expanding uniform runs so assertions can index freely.
    fn cells(&mut self, dense_length: usize) -> Vec<u16> {
        match self.u8() {
            0 => vec![self.u16(); dense_length],
            1 => (0..dense_length).map(|_| self.u16()).collect(),
            other => panic!("unknown cell kind {other}"),
        }
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn overworld(seed: i64) -> GenerationWorld {
    let spec = WorldSpec {
        generator: "minecraft:overworld"
            .parse::<Identifier>()
            .expect("overworld identifier should parse"),
        seed,
        threads: 4,
    };
    GenerationWorld::open(&spec).expect("overworld should open")
}

#[test]
fn snapshot_framing_round_trips() {
    let world = overworld(8080);
    let centers = [ChunkPos::new(0, 0), ChunkPos::new(1, 2)];

    let bytes = world
        .generate_snapshot(&centers, ChunkStatus::Features, 0)
        .expect("snapshot should encode");

    let mut reader = Reader::new(&bytes);
    assert_eq!(reader.u32(), SNAPSHOT_MAGIC, "magic");
    assert_eq!(reader.u16(), SNAPSHOT_VERSION, "version");
    assert_eq!(reader.u16(), 0, "reserved");
    assert_eq!(reader.u32(), 2, "chunk count");

    for expected in centers {
        let chunk_x = reader.i32();
        let chunk_z = reader.i32();
        assert_eq!(
            (chunk_x, chunk_z),
            (expected.0.x, expected.0.y),
            "chunks appear in request order"
        );

        assert_eq!(
            reader.u8() as usize,
            ChunkStatus::Features.get_index(),
            "status index"
        );
        for _ in 0..3 {
            assert_eq!(reader.u8(), 0, "reserved");
        }

        let min_y = reader.i32();
        assert_eq!(min_y, -64, "overworld min_y");
        let section_count = reader.u16() as usize;
        assert_eq!(section_count, 24, "overworld section count");
        assert_eq!(reader.u16(), 0, "reserved");

        let block_palette: Vec<String> = (0..reader.u32()).map(|_| reader.string()).collect();
        let biome_palette: Vec<String> = (0..reader.u32()).map(|_| reader.string()).collect();

        assert!(
            block_palette.iter().any(|entry| entry == "minecraft:air"),
            "every overworld chunk has air; palette was {block_palette:?}"
        );
        assert!(
            block_palette
                .iter()
                .any(|entry| entry == "minecraft:bedrock"),
            "every overworld chunk has bedrock; palette was {block_palette:?}"
        );
        assert!(
            biome_palette.iter().all(|entry| entry.contains(':')),
            "biomes are namespaced identifiers: {biome_palette:?}"
        );

        let mut solid = 0_usize;
        for _ in 0..section_count {
            let blocks = reader.cells(SECTION_VOLUME);
            let biomes = reader.cells(BIOME_CELLS);

            assert!(
                blocks
                    .iter()
                    .all(|index| (*index as usize) < block_palette.len()),
                "block indices stay inside the palette"
            );
            assert!(
                biomes
                    .iter()
                    .all(|index| (*index as usize) < biome_palette.len()),
                "biome indices stay inside the palette"
            );

            let air = block_palette
                .iter()
                .position(|entry| entry == "minecraft:air")
                .expect("air is present");
            solid += blocks
                .iter()
                .filter(|index| **index as usize != air)
                .count();
        }

        assert!(
            solid > 10_000,
            "expected substantial terrain, got {solid} non-air blocks"
        );
    }

    assert_eq!(reader.remaining(), 0, "no trailing bytes");
}

#[test]
fn block_states_carry_their_properties() {
    let world = overworld(4242);
    let bytes = world
        .generate_snapshot(&[ChunkPos::new(0, 0)], ChunkStatus::Features, 0)
        .expect("snapshot should encode");

    // Skip the header and the chunk preamble to reach the block palette.
    let mut reader = Reader::new(&bytes);
    reader.u32();
    reader.u16();
    reader.u16();
    reader.u32();
    reader.i32();
    reader.i32();
    reader.u8();
    for _ in 0..3 {
        reader.u8();
    }
    reader.i32();
    reader.u16();
    reader.u16();

    let palette: Vec<String> = (0..reader.u32()).map(|_| reader.string()).collect();

    // Steel's u16 state ids do not agree with the host's, so a bare identifier is
    // not enough: a state with properties has to name them, or the host resolves
    // the wrong state. Which specific blocks appear depends on the biome, so
    // assert the encoding rule rather than a particular block.
    let with_properties: Vec<&String> =
        palette.iter().filter(|entry| entry.contains('[')).collect();

    assert!(
        !with_properties.is_empty(),
        "every overworld chunk contains stateful blocks; palette was {palette:?}"
    );
    for entry in &with_properties {
        assert!(
            entry.ends_with(']') && entry.contains('='),
            "property lists are `name[key=value,...]`, got {entry:?}"
        );
        assert!(
            !entry.contains("[]"),
            "empty property lists must be omitted, got {entry:?}"
        );
    }

    // Bedrock has no properties and must stay bare.
    assert!(
        palette.iter().any(|entry| entry == "minecraft:bedrock"),
        "property-less states stay bare: {palette:?}"
    );

    // Deepslate carries an axis; it is present in every overworld chunk because
    // the deepslate layer spans the whole world at depth.
    let deepslate = palette
        .iter()
        .find(|entry| entry.starts_with("minecraft:deepslate["))
        .unwrap_or_else(|| panic!("expected deepslate with an axis in {palette:?}"));
    assert!(
        deepslate.contains("axis="),
        "deepslate should carry its axis, got {deepslate:?}"
    );
}
