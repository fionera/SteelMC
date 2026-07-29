//! Chunk snapshot encoding.
//!
//! Serializes generated chunks into a compact, self-describing binary form the
//! host decodes into its own chunk objects.
//!
//! # Why not Steel's own format
//!
//! `PersistentChunk` (`steel-core/src/chunk_saver/format.rs`) is a complete chunk
//! model, but it is encoded with `wincode` and its bit-packing uses power-of-two
//! widths that differ from vanilla's. Decoding either in Java means hand-writing a
//! decoder for a schema that already sits at format version 22. This format is
//! deliberately smaller than that: only what the host needs to reconstruct
//! terrain, in an encoding trivial to read from Java.
//!
//! # Why palettes are keyed by identifier
//!
//! Steel's `BlockStateId` is a `u16` allocated by its own registration order and
//! frozen at build time over vanilla's blocks. It does not agree with the host's
//! numeric IDs, and it cannot represent a mod-added block at all. Sending
//! `minecraft:oak_log` plus its properties means the host resolves against its
//! own registry, and an unknown name is a loud failure rather than silent air.
//!
//! # Not yet carried
//!
//! Heightmaps (the host can prime its own), block entities, scheduled ticks,
//! structure starts and references, entities. All are present in
//! `PersistentChunk` and can be added without changing the framing.

use std::fmt::{Display, Formatter, Result as FmtResult};

use rustc_hash::FxHashMap;
use steel_core::chunk::chunk_access::{ChunkAccess, ChunkStatus};
use steel_registry::{REGISTRY, RegistryExt as _};
use steel_utils::BlockStateId;

/// Magic at the start of every snapshot buffer: `SWGS`.
pub const SNAPSHOT_MAGIC: u32 = u32::from_le_bytes(*b"SWGS");

/// Snapshot format version. Bumped on any framing change.
pub const SNAPSHOT_VERSION: u16 = 1;

/// Blocks along one section edge.
const SECTION_SIZE: usize = 16;
/// Blocks in one section.
const SECTION_VOLUME: usize = SECTION_SIZE * SECTION_SIZE * SECTION_SIZE;
/// Biome cells in one section (4x4x4).
const BIOME_CELLS: usize = 64;

/// A palette that is one repeated value.
const KIND_UNIFORM: u8 = 0;
/// A palette written as one index per cell.
const KIND_DENSE: u8 = 1;

/// Something a chunk contained that cannot be encoded.
#[derive(Debug)]
pub enum SnapshotError {
    /// A block state id had no registry entry. Indicates a Steel-side bug rather
    /// than bad input.
    UnknownBlockState(u16),
    /// A biome id had no registry entry.
    UnknownBiome(u16),
    /// More than `u16::MAX` distinct states or biomes in one chunk.
    PaletteOverflow,
}

impl Display for SnapshotError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::UnknownBlockState(id) => {
                write!(formatter, "no registry entry for block state {id}")
            }
            Self::UnknownBiome(id) => write!(formatter, "no registry entry for biome {id}"),
            Self::PaletteOverflow => formatter.write_str("chunk palette exceeded u16"),
        }
    }
}

/// Accumulates chunk-level palettes while sections are encoded.
///
/// Sections reference palette indices, so both palettes are built first and
/// written into the buffer ahead of the section data.
struct Palettes {
    /// Canonical `name[key=value,...]` per state, in insertion order.
    blocks: Vec<String>,
    /// Steel state id to index in `blocks`.
    block_index: FxHashMap<u16, u16>,
    biomes: Vec<String>,
    biome_index: FxHashMap<u16, u16>,
}

impl Palettes {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            block_index: FxHashMap::default(),
            biomes: Vec::new(),
            biome_index: FxHashMap::default(),
        }
    }

    /// Interns a block state, returning its palette index.
    fn block(&mut self, state: BlockStateId) -> Result<u16, SnapshotError> {
        if let Some(index) = self.block_index.get(&state.0) {
            return Ok(*index);
        }

        let block = REGISTRY
            .blocks
            .by_state_id(state)
            .ok_or(SnapshotError::UnknownBlockState(state.0))?;
        let properties = REGISTRY.blocks.get_properties(state);

        // `name[key=value,key=value]`, properties in registry order so the host
        // sees a stable string for a given state.
        let mut encoded = block.key.to_string();
        if !properties.is_empty() {
            encoded.push('[');
            for (position, (key, value)) in properties.iter().enumerate() {
                if position > 0 {
                    encoded.push(',');
                }
                encoded.push_str(key);
                encoded.push('=');
                encoded.push_str(value);
            }
            encoded.push(']');
        }

        let index = u16::try_from(self.blocks.len()).map_err(|_| SnapshotError::PaletteOverflow)?;
        self.blocks.push(encoded);
        self.block_index.insert(state.0, index);
        Ok(index)
    }

    /// Interns a biome id, returning its palette index.
    fn biome(&mut self, biome: u16) -> Result<u16, SnapshotError> {
        if let Some(index) = self.biome_index.get(&biome) {
            return Ok(*index);
        }

        let entry = REGISTRY
            .biomes
            .by_id(biome as usize)
            .ok_or(SnapshotError::UnknownBiome(biome))?;

        let index = u16::try_from(self.biomes.len()).map_err(|_| SnapshotError::PaletteOverflow)?;
        self.biomes.push(entry.key.to_string());
        self.biome_index.insert(biome, index);
        Ok(index)
    }
}

/// Appends a length-prefixed UTF-8 string.
fn put_string(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    // Identifiers and property lists are far below u16::MAX; saturate rather
    // than panic if that ever stops being true.
    let length = u16::try_from(bytes.len()).unwrap_or(u16::MAX);
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&bytes[..length as usize]);
}

/// Appends a cell array as either a uniform value or one index per cell.
///
/// Most sections in a chunk are entirely air or entirely stone, so collapsing
/// those to a single value is most of the size win for very little code.
fn put_cells(out: &mut Vec<u8>, cells: &[u16]) {
    let uniform = cells
        .first()
        .is_some_and(|first| cells.iter().all(|cell| cell == first));

    if uniform {
        out.push(KIND_UNIFORM);
        out.extend_from_slice(&cells[0].to_le_bytes());
        return;
    }

    out.push(KIND_DENSE);
    for cell in cells {
        out.extend_from_slice(&cell.to_le_bytes());
    }
}

/// Encodes one chunk into `out`.
///
/// Cells are written in the host's natural order, `y * 256 + z * 16 + x` for
/// blocks and `y * 16 + z * 4 + x` for biomes, so the host can index directly.
fn encode_chunk(
    out: &mut Vec<u8>,
    chunk: &ChunkAccess,
    status: ChunkStatus,
    min_y: i32,
) -> Result<(), SnapshotError> {
    let sections = chunk.sections();
    let section_count = sections.sections.len();

    let mut palettes = Palettes::new();

    // Read every column once. `read_column_into` takes each section's lock once
    // for 16 vertical reads, which is far cheaper than a lock per block.
    let total_y = section_count * SECTION_SIZE;
    let mut column = Vec::with_capacity(total_y);
    let mut blocks = vec![0_u16; SECTION_SIZE * SECTION_SIZE * total_y];
    for x in 0..SECTION_SIZE {
        for z in 0..SECTION_SIZE {
            sections.read_column_into(x, z, &mut column);
            for (relative_y, state) in column.iter().enumerate() {
                blocks[relative_y * 256 + z * SECTION_SIZE + x] = palettes.block(*state)?;
            }
        }
    }

    // Already indexed `[section * 64 + qy * 16 + qz * 4 + qx]`, which matches the
    // order written below.
    let raw_biomes = sections.read_all_biomes();
    let mut biomes = vec![0_u16; raw_biomes.len()];
    for (slot, raw) in biomes.iter_mut().zip(raw_biomes.iter()) {
        *slot = palettes.biome(*raw)?;
    }

    let position = chunk.pos();
    out.extend_from_slice(&position.0.x.to_le_bytes());
    out.extend_from_slice(&position.0.y.to_le_bytes());
    out.push(u8::try_from(status.get_index()).unwrap_or(0));
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&min_y.to_le_bytes());
    out.extend_from_slice(&u16::try_from(section_count).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());

    out.extend_from_slice(
        &u32::try_from(palettes.blocks.len())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    for entry in &palettes.blocks {
        put_string(out, entry);
    }
    out.extend_from_slice(
        &u32::try_from(palettes.biomes.len())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    for entry in &palettes.biomes {
        put_string(out, entry);
    }

    for section in 0..section_count {
        let start = section * SECTION_VOLUME;
        put_cells(out, &blocks[start..start + SECTION_VOLUME]);

        let biome_start = section * BIOME_CELLS;
        put_cells(out, &biomes[biome_start..biome_start + BIOME_CELLS]);
    }

    Ok(())
}

/// Encodes `chunks` into a single snapshot buffer.
///
/// # Errors
/// Returns an error if a chunk contains a block state or biome with no registry
/// entry, or if one chunk needs more than `u16::MAX` palette entries.
pub fn encode(
    chunks: &[(&ChunkAccess, ChunkStatus)],
    min_y: i32,
) -> Result<Vec<u8>, SnapshotError> {
    let mut out = Vec::new();
    out.extend_from_slice(&SNAPSHOT_MAGIC.to_le_bytes());
    out.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&u32::try_from(chunks.len()).unwrap_or(0).to_le_bytes());

    for (chunk, status) in chunks {
        encode_chunk(&mut out, chunk, *status, min_y)?;
    }

    Ok(out)
}
