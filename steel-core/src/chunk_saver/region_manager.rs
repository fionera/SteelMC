//! Region file manager with seek-based chunk access.
//!
//! Uses a sector-based format where only the header (8KB) is kept in memory.
//! Chunk data is read on-demand from disk and converted directly to runtime
//! format, avoiding memory duplication.

use std::{
    cell::RefCell,
    fmt,
    io::{self},
    path::PathBuf,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use rustc_hash::FxHashMap;
use steel_utils::{
    ChunkPos,
    locks::{AsyncMutex, AsyncRwLock},
};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    sync::oneshot,
};
use zstd::bulk::Compressor;

use crate::chunk::status::ChunkStatus;
use crate::world::World;

/// Compression level for chunk payloads.
const CHUNK_COMPRESSION_LEVEL: i32 = 3;

thread_local! {
    /// One zstd compression context per encoding thread.
    ///
    /// `zstd::encode_all` builds and tears down a `ZSTD_CCtx` on every call, and
    /// that context's workspace is allocated through libc `malloc` rather than
    /// the process's global Rust allocator. Saving one chunk per call across the
    /// encoding pool therefore hammers glibc's arena lock: profiling a 90,601
    /// chunk pregeneration put ~6% of the entire machine in
    /// `__lll_lock_wait_private`, of which 99.4% was under `ZSTD_createCCtx`,
    /// `ZSTD_freeCCtx` and `ZSTD_resetCCtx_internal` -- against 1.3% spent
    /// actually compressing. Keeping one context per thread removes the churn.
    ///
    /// Frame bytes differ slightly from `encode_all` (the bulk API records the
    /// content size in the header), which is fine: nothing depends on the exact
    /// compressed bytes, and `zstd::decode_all` reads either form, so region
    /// files written by older builds still load.
    static CHUNK_COMPRESSOR: RefCell<Option<Compressor<'static>>> =
        const { RefCell::new(None) };
}

/// Compresses a serialized chunk payload, reusing this thread's zstd context.
fn compress_chunk(data: &[u8]) -> io::Result<Vec<u8>> {
    CHUNK_COMPRESSOR.with_borrow_mut(|slot| {
        if slot.is_none() {
            *slot = Some(Compressor::new(CHUNK_COMPRESSION_LEVEL)?);
        }
        slot.as_mut()
            .expect("compressor was just initialized")
            .compress(data)
    })
}

use super::{
    ChunkStorage, LoadedChunk, PersistentChunk,
    format::{
        CHUNK_TABLE_SIZE, ChunkEntry, FILE_HEADER_SIZE, FIRST_DATA_SECTOR, FORMAT_VERSION,
        MAX_CHUNK_SIZE, REGION_MAGIC, RegionHeader, RegionPos, SECTOR_SIZE,
    },
};

#[derive(Debug)]
struct CorruptChunkData(String);

impl fmt::Display for CorruptChunkData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Manages region files with seek-based chunk access.
///
/// Only keeps region headers (8KB each) in memory, not chunk data.
/// Chunks are loaded on-demand and converted directly to runtime format.
pub struct RegionManager {
    /// Base directory for region files (e.g., "world/region").
    base_path: PathBuf,
    /// Open regions, each behind its own lock.
    ///
    /// The map itself is read-mostly: it is written only when a region is first
    /// opened or finally closed, which is once per 1,024 chunks rather than once
    /// per chunk. That distinction is the whole point of the indirection. When
    /// this was a single `RwLock<HashMap<_, RegionHandle>>` every chunk took its
    /// write lock twice -- once to acquire, once to release -- and held it across
    /// file opens and header writes, which serialized chunk generation on one
    /// lock: instrumenting a 601x601 pregeneration found 11,784 chunks queued
    /// there at one sample while the 104-thread generation pool had 36 jobs to
    /// run.
    regions: AsyncRwLock<FxHashMap<RegionPos, Arc<RegionEntry>>>,
}

/// One open region file.
struct RegionEntry {
    /// The file and its header. Operations on a region serialize here, as they
    /// must: they share one file cursor.
    handle: AsyncMutex<RegionHandle>,
    /// Chunks currently counted against this region.
    ///
    /// Outside the handle lock so that dropping a reference does not have to
    /// take it, and so the map's write lock is only needed when the count
    /// actually reaches zero. It is incremented under the map's read lock and
    /// only ever removed under its write lock, which is what stops a reference
    /// taken concurrently with the last release from reviving an entry that has
    /// already been dropped from the map -- two live handles on one region file
    /// would let their headers diverge.
    references: AtomicUsize,
}

/// Prepared chunk data ready to be saved asynchronously.
/// Created by `prepare_chunk_save` during the holder's snapshot-preparation phase.
pub struct PreparedChunkSave {
    /// The chunk position.
    pub pos: ChunkPos,
    /// The highest persisted status captured with the chunk data.
    pub status: ChunkStatus,
    /// The serialized chunk data.
    pub persistent: PersistentChunk<'static>,
    /// Runtime manager entity IDs that were either serialized or explicitly skipped.
    pub handled_runtime_entity_ids: Vec<i32>,
}

/// An open region file with its header.
struct RegionHandle {
    /// File handle for reading/writing.
    file: File,
    /// Chunk location header (8KB).
    header: RegionHeader,
    /// Whether the header has been modified since last save.
    header_dirty: bool,
    /// Current file size in sectors.
    file_sectors: u32,
}

impl RegionManager {
    /// Creates a new region manager.
    ///
    /// # Arguments
    /// * `base_path` - Directory where region files are stored.
    /// * `registry` - The registry for block state and biome conversions.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            regions: AsyncRwLock::new(FxHashMap::default()),
        }
    }

    /// Gets the file path for a region.
    fn region_path(&self, pos: RegionPos) -> PathBuf {
        self.base_path.join(pos.filename())
    }

    /// Returns the region's entry with one reference counted against it.
    ///
    /// Every caller must pair this with [`Self::drop_region_reference`]; the
    /// region file stays open, and its header unflushed, until the last
    /// reference goes.
    async fn acquire_region(&self, pos: RegionPos) -> io::Result<Arc<RegionEntry>> {
        if let Some(entry) = self.regions.read().await.get(&pos) {
            entry.references.fetch_add(1, Ordering::AcqRel);
            return Ok(Arc::clone(entry));
        }

        // Opening has to be serialized -- two creators would truncate each
        // other's file -- so it happens under the map's write lock, with a
        // re-check to settle the race. This is the rare path: once per region
        // file, against the 1,024 chunks that live in it.
        let mut regions = self.regions.write().await;
        if let Some(entry) = regions.get(&pos) {
            entry.references.fetch_add(1, Ordering::AcqRel);
            return Ok(Arc::clone(entry));
        }

        let entry = Arc::new(RegionEntry {
            handle: AsyncMutex::new(self.open_region(pos).await?),
            references: AtomicUsize::new(1),
        });
        regions.insert(pos, Arc::clone(&entry));
        Ok(entry)
    }

    /// Returns the region's entry if it is open, without counting a reference.
    async fn open_region_entry(&self, pos: RegionPos) -> Option<Arc<RegionEntry>> {
        self.regions.read().await.get(&pos).map(Arc::clone)
    }

    /// Drops one reference, closing the region once none are left.
    async fn drop_region_reference(
        &self,
        pos: RegionPos,
        entry: &Arc<RegionEntry>,
    ) -> io::Result<()> {
        {
            let _regions = self.regions.read().await;
            if entry.references.fetch_sub(1, Ordering::AcqRel) != 1 {
                return Ok(());
            }
        }

        // Last reference. Removal needs the write lock, and an acquire may have
        // taken a fresh reference in the meantime, so re-check under it.
        {
            let mut regions = self.regions.write().await;
            if entry.references.load(Ordering::Acquire) != 0 {
                return Ok(());
            }
            match regions.get(&pos) {
                Some(current) if Arc::ptr_eq(current, entry) => regions.remove(&pos),
                // Already replaced by a newer open; that entry owns the file now.
                _ => return Ok(()),
            };
        }

        // Out of the map and unreferenced, so nothing else can reach this
        // handle: flush its header outside both locks.
        let mut handle = entry.handle.lock().await;
        let handle = &mut *handle;
        if handle.header_dirty {
            Self::write_header(&mut handle.file, &handle.header).await?;
            handle.header_dirty = false;
        }
        Ok(())
    }

    /// Opens or creates a region file, loading only the header.
    async fn open_region(&self, pos: RegionPos) -> io::Result<RegionHandle> {
        let path = self.region_path(pos);

        if !path.exists() {
            // Create new region file with empty header
            return self.create_region(pos).await;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .await?;

        // Read and verify magic + version
        let mut header_bytes = [0u8; FILE_HEADER_SIZE];
        file.read_exact(&mut header_bytes).await?;

        let magic = &header_bytes[0..4];
        if magic != REGION_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid region file magic",
            ));
        }

        let version = u16::from_le_bytes([header_bytes[4], header_bytes[5]]);
        if version != FORMAT_VERSION {
            // Version mismatch — backup the old file and create a fresh region.
            drop(file);
            let backup_path = path.with_extension(format!("srg.v{version}.bak"));
            tracing::warn!(
                "Region file {} has version {version} (expected {FORMAT_VERSION}), backing up to {} and recreating",
                path.display(),
                backup_path.display()
            );
            fs::rename(&path, &backup_path).await?;
            return self.create_region(pos).await;
        }

        // Read chunk table
        let mut table_bytes = vec![0u8; CHUNK_TABLE_SIZE];
        file.read_exact(&mut table_bytes).await?;
        let header = RegionHeader::from_bytes(&table_bytes).map_err(|index| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("region chunk table entry {index} has an invalid status byte"),
            )
        })?;

        // Calculate file size in sectors
        let file_size = file.seek(io::SeekFrom::End(0)).await?;
        let file_sectors = file_size.div_ceil(SECTOR_SIZE as u64) as u32;
        Self::validate_region_entries(&header, file_sectors)?;

        Ok(RegionHandle {
            file,
            header,
            header_dirty: false,
            file_sectors,
        })
    }

    /// Creates a new empty region file.
    async fn create_region(&self, pos: RegionPos) -> io::Result<RegionHandle> {
        fs::create_dir_all(&self.base_path).await?;

        let path = self.region_path(pos);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .await?;

        // Write header
        let mut header_bytes = [0u8; FILE_HEADER_SIZE];
        header_bytes[0..4].copy_from_slice(&REGION_MAGIC);
        header_bytes[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        file.write_all(&header_bytes).await?;

        // Write empty chunk table
        let header = RegionHeader::new();
        file.write_all(&header.to_bytes()).await?;
        file.flush().await?;

        Ok(RegionHandle {
            file,
            header,
            header_dirty: false,
            file_sectors: FIRST_DATA_SECTOR,
        })
    }

    /// Writes the header to disk.
    async fn write_header(file: &mut File, header: &RegionHeader) -> io::Result<()> {
        file.seek(io::SeekFrom::Start(FILE_HEADER_SIZE as u64))
            .await?;
        file.write_all(&header.to_bytes()).await?;
        file.flush().await?;
        Ok(())
    }

    /// Reads a chunk's compressed data from disk.
    async fn read_chunk_data(
        file: &mut File,
        sector_offset: u32,
        size: u32,
    ) -> io::Result<Vec<u8>> {
        let byte_offset = u64::from(sector_offset) * SECTOR_SIZE as u64;
        file.seek(io::SeekFrom::Start(byte_offset)).await?;

        let mut compressed = vec![0u8; size as usize];
        file.read_exact(&mut compressed).await?;
        Ok(compressed)
    }

    fn validate_chunk_entry(entry: ChunkEntry, file_sectors: u32) -> io::Result<()> {
        if entry.sector_offset < FIRST_DATA_SECTOR {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chunk table entry points into the region header at sector {}",
                    entry.sector_offset
                ),
            ));
        }
        if entry.size_bytes == 0 || entry.size_bytes as usize > MAX_CHUNK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid chunk table entry size {}", entry.size_bytes),
            ));
        }
        let Some(end_sector) = entry.sector_offset.checked_add(entry.sector_count()) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk table entry sector range overflowed",
            ));
        };
        if end_sector > file_sectors {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chunk table entry ends at sector {end_sector}, past region end {file_sectors}"
                ),
            ));
        }
        Ok(())
    }

    fn validate_region_entries(header: &RegionHeader, file_sectors: u32) -> io::Result<()> {
        let mut occupied = vec![false; file_sectors as usize];
        for sector in occupied.iter_mut().take(FIRST_DATA_SECTOR as usize) {
            *sector = true;
        }
        for (index, &entry) in header.entries.iter().enumerate() {
            if !entry.exists() {
                continue;
            }
            Self::validate_chunk_entry(entry, file_sectors)?;
            let start = entry.sector_offset as usize;
            let end = start + entry.sector_count() as usize;
            if occupied[start..end].iter().any(|is_occupied| *is_occupied) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("chunk table entry {index} overlaps another region allocation"),
                ));
            }
            occupied[start..end].fill(true);
        }
        Ok(())
    }

    async fn clear_corrupt_chunk_if_unchanged(
        &self,
        region_pos: RegionPos,
        index: usize,
        expected_entry: ChunkEntry,
    ) -> io::Result<bool> {
        let Some(entry) = self.open_region_entry(region_pos).await else {
            return Err(io::Error::other(
                "region was released while clearing corrupt chunk data",
            ));
        };
        let mut handle = entry.handle.lock().await;
        let handle = &mut *handle;
        if handle.header.entries[index] != expected_entry {
            return Ok(false);
        }

        handle.header.entries[index] = ChunkEntry::empty();
        if let Err(error) = Self::write_header(&mut handle.file, &handle.header).await {
            handle.header.entries[index] = expected_entry;
            return Err(error);
        }
        handle.header_dirty = false;
        Ok(true)
    }

    /// Writes chunk data to disk at the specified sector offset.
    async fn write_chunk_data(
        file: &mut File,
        sector_offset: u32,
        data: &[u8],
        file_sectors: &mut u32,
    ) -> io::Result<()> {
        let byte_offset = u64::from(sector_offset) * SECTOR_SIZE as u64;
        file.seek(io::SeekFrom::Start(byte_offset)).await?;
        file.write_all(data).await?;

        // Pad to sector boundary
        let padding_needed = (SECTOR_SIZE - (data.len() % SECTOR_SIZE)) % SECTOR_SIZE;
        if padding_needed > 0 {
            file.write_all(&vec![0u8; padding_needed]).await?;
        }

        // Update file sectors if we wrote past the end
        let sectors_used = data.len().div_ceil(SECTOR_SIZE) as u32;
        let end_sector = sector_offset + sectors_used;
        if end_sector > *file_sectors {
            *file_sectors = end_sector;
        }

        file.flush().await?;
        Ok(())
    }

    /// Saves prepared chunk data to disk after the snapshot-preparation phase has ended.
    pub async fn save_chunk_data(
        &self,
        prepared: PreparedChunkSave,
        thread_pool: &rayon::ThreadPool,
    ) -> io::Result<bool> {
        let pos = prepared.pos;
        let status = prepared.status;
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);

        let (sender, receiver) = oneshot::channel();
        thread_pool.spawn(move || {
            let result = Self::encode_chunk(prepared);
            if sender.send(result).is_err() {
                tracing::trace!(
                    chunk = ?pos,
                    "Discarding encoded chunk after its save task was canceled"
                );
            }
        });
        let compressed = receiver.await.map_err(|_| {
            io::Error::other("chunk encode task ended without returning a result")
        })??;

        // Counted for the duration of the write, so a concurrent release cannot
        // close the region out from under it and leave a second handle to open
        // the same file.
        let entry = self.acquire_region(region_pos).await?;
        let result = self
            .write_chunk_into_region(&entry, index, status, &compressed)
            .await;
        let released = self.drop_region_reference(region_pos, &entry).await;
        result?;
        released?;
        Ok(true)
    }

    /// Writes one encoded chunk into an already-referenced region.
    async fn write_chunk_into_region(
        &self,
        entry: &Arc<RegionEntry>,
        index: usize,
        status: ChunkStatus,
        compressed: &[u8],
    ) -> io::Result<()> {
        let mut handle = entry.handle.lock().await;
        let handle = &mut *handle;

        // Find space for the chunk
        let sectors_needed = compressed.len().div_ceil(SECTOR_SIZE) as u32;
        let old_entry = handle.header.entries[index];

        // Try to reuse existing space if it fits
        let sector_offset = if old_entry.exists() && old_entry.sector_count() >= sectors_needed {
            old_entry.sector_offset
        } else {
            handle
                .header
                .find_free_sectors(sectors_needed, handle.file_sectors)
        };

        // Write chunk data
        Self::write_chunk_data(
            &mut handle.file,
            sector_offset,
            compressed,
            &mut handle.file_sectors,
        )
        .await?;

        // Update header entry
        handle.header.entries[index] =
            super::format::ChunkEntry::new(sector_offset, compressed.len() as u32, status);
        handle.header_dirty = true;

        Ok(())
    }

    fn encode_chunk(prepared: PreparedChunkSave) -> io::Result<Vec<u8>> {
        let data = wincode::serialize(&prepared.persistent)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let compressed = compress_chunk(&data)?;

        if compressed.len() > MAX_CHUNK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Chunk too large: {} bytes (max {})",
                    compressed.len(),
                    MAX_CHUNK_SIZE
                ),
            ));
        }

        Ok(compressed)
    }

    /// Loads a chunk from the appropriate region.
    ///
    /// Automatically opens the region if not already open. The region's reference
    /// count is incremented, so you must call `release_chunk` when done with the chunk.
    ///
    /// Returns `Ok(None)` if the chunk doesn't exist on disk.
    ///
    /// # Arguments
    /// * `pos` - The chunk position
    /// * `min_y` - The minimum Y coordinate of the world
    /// * `height` - The total height of the world
    /// * `level` - Weak reference to the world for Full chunk runtime access
    ///
    /// The region must already be acquired via `acquire_chunk` before calling this.
    pub async fn load_chunk(
        &self,
        pos: ChunkPos,
        min_y: i32,
        height: i32,
        level: Weak<World>,
        thread_pool: &rayon::ThreadPool,
    ) -> io::Result<Option<LoadedChunk>> {
        if height <= 0 || height % 16 != 0 || min_y % 16 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "chunk world range must be section-aligned, got min_y={min_y}, height={height}"
                ),
            ));
        }
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);

        let (compressed, entry) = {
            // Should already be open, via acquire_chunk.
            let Some(region) = self.open_region_entry(region_pos).await else {
                log::warn!("load_chunk called without acquire_chunk for region {region_pos:?}");
                return Ok(None);
            };
            let mut handle = region.handle.lock().await;
            let handle = &mut *handle;

            // Check if chunk exists
            let entry = handle.header.entries[index];
            if !entry.exists() {
                return Ok(None);
            }

            // Invalid offsets and sizes indicate damage to the region's location
            // table, not a self-contained chunk payload. Do not discard the slot.
            Self::validate_chunk_entry(entry, handle.file_sectors)?;

            // Read chunk data from disk
            let compressed =
                Self::read_chunk_data(&mut handle.file, entry.sector_offset, entry.size_bytes)
                    .await?;
            (compressed, entry)
        };

        // Keep CPU-heavy decoding off the async runtime. Awaiting the Rayon
        // handoff also lets the region-lock waiter woken above make progress.
        let (sender, receiver) = oneshot::channel();
        thread_pool.spawn(move || {
            let result = Self::decode_chunk(compressed, pos, entry.status, min_y, height, level);
            if sender.send(result).is_err() {
                tracing::trace!(
                    chunk = ?pos,
                    "Discarding decoded chunk after its load task was canceled"
                );
            }
        });

        let decoded = receiver
            .await
            .map_err(|_| io::Error::other("chunk decode task ended without returning a result"))?;
        match decoded {
            Ok(loaded) => Ok(Some(loaded)),
            Err(error) => {
                if !self
                    .clear_corrupt_chunk_if_unchanged(region_pos, index, entry)
                    .await?
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt chunk payload was superseded before it could be removed: {error}"
                        ),
                    ));
                }
                tracing::error!(
                    chunk = ?pos,
                    "Discarded corrupt chunk payload and will regenerate it: {error}",
                );
                Ok(None)
            }
        }
    }

    fn decode_chunk(
        compressed: Vec<u8>,
        pos: ChunkPos,
        status: ChunkStatus,
        min_y: i32,
        height: i32,
        level: Weak<World>,
    ) -> Result<LoadedChunk, CorruptChunkData> {
        let data = zstd::decode_all(&compressed[..])
            .map_err(|error| CorruptChunkData(format!("zstd decode failed: {error}")))?;
        let persistent: PersistentChunk<'_> = wincode::deserialize(&data)
            .map_err(|error| CorruptChunkData(format!("chunk decode failed: {error}")))?;

        ChunkStorage::try_persistent_to_chunk(&persistent, pos, status, min_y, height, level)
            .map_err(|error| CorruptChunkData(format!("chunk materialization failed: {error}")))
    }

    /// Acquires a chunk, incrementing the region's reference count.
    ///
    /// This opens or creates the region file. Call this before loading or
    /// generating a chunk, and call `release_chunk` when done with the chunk.
    ///
    /// Returns `Ok(true)` if the chunk exists on disk, `Ok(false)` if it doesn't.
    pub async fn acquire_chunk(&self, pos: ChunkPos) -> io::Result<bool> {
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);

        let entry = self.acquire_region(region_pos).await?;
        let exists = entry.handle.lock().await.header.entries[index].exists();
        Ok(exists)
    }

    /// Releases a loaded chunk, decrementing the region's reference count.
    ///
    /// When all chunks from a region are released, the header is saved (if dirty)
    /// and the file handle is closed.
    ///
    /// This must be called for each chunk returned by `load_chunk`.
    pub async fn release_chunk(&self, pos: ChunkPos) -> io::Result<()> {
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let Some(entry) = self.open_region_entry(region_pos).await else {
            return Ok(());
        };
        self.drop_region_reference(region_pos, &entry).await
    }

    /// Checks if a chunk exists on disk without loading it.
    pub async fn chunk_exists(&self, pos: ChunkPos) -> io::Result<bool> {
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);

        // Check the cached header first.
        if let Some(region) = self.open_region_entry(region_pos).await {
            return Ok(region.handle.lock().await.header.entries[index].exists());
        }

        // Need to read header from disk
        let path = self.region_path(region_pos);
        if !path.exists() {
            return Ok(false);
        }

        let mut file = File::open(&path).await?;

        // Skip magic + version
        file.seek(io::SeekFrom::Start(FILE_HEADER_SIZE as u64))
            .await?;

        // Read just the one entry we need (8 bytes at index * 8)
        file.seek(io::SeekFrom::Current((index * 8) as i64)).await?;
        let mut entry_bytes = [0u8; 8];
        file.read_exact(&mut entry_bytes).await?;

        let Some(entry) = super::format::ChunkEntry::from_bytes(entry_bytes) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk table entry has an invalid status byte",
            ));
        };
        Ok(entry.exists())
    }

    /// Flushes all dirty headers to disk.
    pub async fn flush_all(&self) -> io::Result<()> {
        let entries: Vec<Arc<RegionEntry>> =
            self.regions.read().await.values().map(Arc::clone).collect();

        for entry in entries {
            let mut handle = entry.handle.lock().await;
            let handle = &mut *handle;
            if handle.header_dirty {
                Self::write_header(&mut handle.file, &handle.header).await?;
                handle.header_dirty = false;
            }
        }

        Ok(())
    }

    /// Flushes all dirty headers and closes all region file handles.
    ///
    /// This should be called during graceful shutdown after all chunks have been saved.
    /// It ensures all data is persisted and file handles are properly closed.
    pub async fn close_all(&self) -> io::Result<()> {
        let entries: Vec<Arc<RegionEntry>> =
            self.regions.write().await.drain().map(|(_, entry)| entry).collect();

        for entry in entries {
            let mut handle = entry.handle.lock().await;
            let handle = &mut *handle;
            if handle.header_dirty {
                Self::write_header(&mut handle.file, &handle.header).await?;
                handle.header_dirty = false;
            }
            // The file is closed when the last reference to the entry goes.
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        path::Path,
        process,
        sync::{
            Weak,
            atomic::{AtomicU64, Ordering},
        },
    };

    use super::*;
    use crate::chunk_saver::{PersistentChunk, PersistentLightData};

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn test_directory(name: &str) -> PathBuf {
        let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!(
            "steel-region-manager-{name}-{}-{sequence}",
            process::id()
        ))
    }

    async fn write_test_region(
        directory: &Path,
        pos: ChunkPos,
        payload: &[u8],
        declared_size: u32,
    ) -> io::Result<()> {
        fs::create_dir_all(directory).await?;
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let path = directory.join(region_pos.filename());
        let mut file = File::create(path).await?;
        let mut file_header = [0u8; FILE_HEADER_SIZE];
        file_header[0..4].copy_from_slice(&REGION_MAGIC);
        file_header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        file.write_all(&file_header).await?;

        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);
        let mut header = RegionHeader::new();
        header.entries[index] =
            ChunkEntry::new(FIRST_DATA_SECTOR, declared_size, ChunkStatus::Empty);
        file.write_all(&header.to_bytes()).await?;
        file.seek(io::SeekFrom::Start(
            u64::from(FIRST_DATA_SECTOR) * SECTOR_SIZE as u64,
        ))
        .await?;
        file.write_all(payload).await?;
        file.flush().await
    }

    fn test_thread_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("test thread pool should build")
    }

    async fn assert_slot_exists_on_disk(directory: &Path, pos: ChunkPos) {
        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let path = directory.join(region_pos.filename());
        let mut file = File::open(path)
            .await
            .expect("test region should remain readable");
        let (local_x, local_z) = RegionPos::local_chunk_pos(pos.0.x, pos.0.y);
        let index = RegionHeader::chunk_index(local_x, local_z);
        file.seek(io::SeekFrom::Start(
            FILE_HEADER_SIZE as u64 + (index * 8) as u64,
        ))
        .await
        .expect("chunk table entry should be seekable");
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)
            .await
            .expect("chunk table entry should be readable");
        assert!(ChunkEntry::from_bytes(bytes).is_some_and(|entry| entry.exists()));
    }

    fn empty_persistent_chunk() -> PersistentChunk<'static> {
        PersistentChunk {
            last_modified: 0,
            block_states: Vec::new(),
            biomes: Vec::new(),
            sections: Vec::new(),
            block_entities: Vec::new(),
            entities: Vec::new(),
            block_ticks: Vec::new(),
            fluid_ticks: Vec::new(),
            heightmaps: Vec::new(),
            light: PersistentLightData::default(),
            carving_mask: None,
            postprocessing: Vec::new(),
            structure_starts: Vec::new(),
            structure_references: Vec::new(),
            pois: Vec::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_acquire_and_release_leave_no_open_regions() {
        // The reference count lives outside the region's own lock so that
        // acquiring and releasing a chunk -- which every chunk does -- need not
        // take it. That only works if a reference taken concurrently with the
        // last release cannot revive an entry already dropped from the map: two
        // live handles on one region file would let their headers diverge.
        let directory = test_directory("concurrent-acquire");
        let pos = ChunkPos::new(0, 0);
        let payload = b"payload";
        write_test_region(&directory, pos, payload, payload.len() as u32)
            .await
            .expect("test region should be written");

        let manager = Arc::new(RegionManager::new(&directory));
        let mut tasks = Vec::new();
        for chunk in 0..32 {
            let manager = Arc::clone(&manager);
            // All 32 chunks are inside the same 32x32 region, so every task
            // contends for the same entry.
            let chunk_pos = ChunkPos::new(chunk % 8, chunk / 8);
            tasks.push(tokio::spawn(async move {
                for _ in 0..64 {
                    manager
                        .acquire_chunk(chunk_pos)
                        .await
                        .expect("region should open");
                    manager
                        .release_chunk(chunk_pos)
                        .await
                        .expect("region should release");
                }
            }));
        }
        for task in tasks {
            task.await.expect("acquire/release task should not panic");
        }

        assert!(
            manager.regions.read().await.is_empty(),
            "every acquire was released, so no region should still be open"
        );
        assert!(
            manager
                .chunk_exists(pos)
                .await
                .expect("header should still be readable"),
            "the chunk table survived the churn"
        );

        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn saving_a_chunk_flushes_its_header_once_released() {
        // save_chunk counts itself against the region for the duration of the
        // write. Releasing that reference is what closes the file and writes the
        // header, so a save into a region nobody else holds must still land.
        let directory = test_directory("save-flush");
        fs::create_dir_all(&directory)
            .await
            .expect("test directory should be creatable");
        let pos = ChunkPos::new(3, 5);
        let manager = RegionManager::new(&directory);

        assert!(
            manager
                .save_chunk_data(
                    PreparedChunkSave {
                        pos,
                        status: ChunkStatus::Full,
                        persistent: empty_persistent_chunk(),
                        handled_runtime_entity_ids: Vec::new(),
                    },
                    &test_thread_pool(),
                )
                .await
                .expect("chunk should save")
        );
        assert!(
            manager.regions.read().await.is_empty(),
            "a save into an otherwise unheld region should close it again"
        );

        let reopened = RegionManager::new(&directory);
        assert!(
            reopened
                .chunk_exists(pos)
                .await
                .expect("header should be readable"),
            "the saved chunk's table entry was flushed to disk"
        );

        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn a_region_held_by_another_chunk_stays_open_across_a_save() {
        let directory = test_directory("save-held");
        fs::create_dir_all(&directory)
            .await
            .expect("test directory should be creatable");
        let held = ChunkPos::new(0, 0);
        let saved = ChunkPos::new(1, 0);
        let manager = RegionManager::new(&directory);

        manager
            .acquire_chunk(held)
            .await
            .expect("region should open");
        manager
            .save_chunk_data(
                PreparedChunkSave {
                    pos: saved,
                    status: ChunkStatus::Full,
                    persistent: empty_persistent_chunk(),
                    handled_runtime_entity_ids: Vec::new(),
                },
                &test_thread_pool(),
            )
            .await
            .expect("chunk should save");

        assert_eq!(
            manager.regions.read().await.len(),
            1,
            "the save must not close a region another chunk is holding"
        );
        manager
            .release_chunk(held)
            .await
            .expect("region should release");
        assert!(manager.regions.read().await.is_empty());

        let reopened = RegionManager::new(&directory);
        assert!(
            reopened
                .chunk_exists(saved)
                .await
                .expect("header should be readable")
        );

        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn invalid_zstd_payload_is_removed_for_regeneration() {
        let directory = test_directory("zstd");
        let pos = ChunkPos::new(0, 0);
        let payload = b"this is not a zstd frame";
        write_test_region(&directory, pos, payload, payload.len() as u32)
            .await
            .expect("test region should be written");

        let manager = RegionManager::new(&directory);
        assert!(
            manager
                .acquire_chunk(pos)
                .await
                .expect("region should open")
        );
        let loaded = manager
            .load_chunk(pos, 0, 16, Weak::new(), &test_thread_pool())
            .await
            .expect("corrupt payload should be handled");
        assert!(loaded.is_none());
        assert!(
            !manager
                .chunk_exists(pos)
                .await
                .expect("header should be readable")
        );
        manager
            .release_chunk(pos)
            .await
            .expect("region should release");

        let reopened = RegionManager::new(&directory);
        assert!(
            !reopened
                .chunk_exists(pos)
                .await
                .expect("header should be flushed")
        );
        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn semantically_invalid_complete_payload_is_removed_for_regeneration() {
        let directory = test_directory("semantic");
        let pos = ChunkPos::new(0, 0);
        let persistent = PersistentChunk {
            last_modified: 0,
            block_states: Vec::new(),
            biomes: Vec::new(),
            sections: Vec::new(),
            block_entities: Vec::new(),
            entities: Vec::new(),
            block_ticks: Vec::new(),
            fluid_ticks: Vec::new(),
            heightmaps: Vec::new(),
            light: PersistentLightData::default(),
            carving_mask: None,
            postprocessing: Vec::new(),
            structure_starts: Vec::new(),
            structure_references: Vec::new(),
            pois: Vec::new(),
        };
        let encoded = wincode::serialize(&persistent).expect("test chunk should encode");
        let payload = zstd::encode_all(encoded.as_slice(), 1).expect("test chunk should compress");
        write_test_region(&directory, pos, &payload, payload.len() as u32)
            .await
            .expect("test region should be written");

        let manager = RegionManager::new(&directory);
        assert!(
            manager
                .acquire_chunk(pos)
                .await
                .expect("region should open")
        );
        let loaded = manager
            .load_chunk(pos, 0, 16, Weak::new(), &test_thread_pool())
            .await
            .expect("semantic corruption should be handled");
        assert!(loaded.is_none());
        assert!(
            !manager
                .chunk_exists(pos)
                .await
                .expect("slot should be cleared")
        );
        manager
            .release_chunk(pos)
            .await
            .expect("region should release");
        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn incomplete_payload_read_is_an_error_and_keeps_slot() {
        let directory = test_directory("short-read");
        let pos = ChunkPos::new(0, 0);
        write_test_region(&directory, pos, &[1, 2, 3], 128)
            .await
            .expect("test region should be written");

        let manager = RegionManager::new(&directory);
        assert!(
            manager
                .acquire_chunk(pos)
                .await
                .expect("region should open")
        );
        let Err(error) = manager
            .load_chunk(pos, 0, 16, Weak::new(), &test_thread_pool())
            .await
        else {
            panic!("short filesystem read must not be treated as payload corruption");
        };
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert!(manager.chunk_exists(pos).await.expect("slot should remain"));
        manager
            .release_chunk(pos)
            .await
            .expect("region should release");
        assert_slot_exists_on_disk(&directory, pos).await;
        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn invalid_world_geometry_is_not_classified_as_chunk_corruption() {
        let directory = test_directory("invalid-world-range");
        let pos = ChunkPos::new(0, 0);
        let payload = b"payload must not be decoded";
        write_test_region(&directory, pos, payload, payload.len() as u32)
            .await
            .expect("test region should be written");

        let manager = RegionManager::new(&directory);
        assert!(
            manager
                .acquire_chunk(pos)
                .await
                .expect("region should open")
        );
        let Err(error) = manager
            .load_chunk(pos, 1, 16, Weak::new(), &test_thread_pool())
            .await
        else {
            panic!("invalid world geometry must fail before decoding the chunk");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(manager.chunk_exists(pos).await.expect("slot should remain"));
        manager
            .release_chunk(pos)
            .await
            .expect("region should release");
        assert_slot_exists_on_disk(&directory, pos).await;
        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }

    #[tokio::test]
    async fn invalid_chunk_status_byte_is_a_structural_error_and_is_preserved() {
        let directory = test_directory("invalid-status");
        let pos = ChunkPos::new(0, 0);
        let payload = b"payload must not be decoded";
        write_test_region(&directory, pos, payload, payload.len() as u32)
            .await
            .expect("test region should be written");

        let region_pos = RegionPos::from_chunk(pos.0.x, pos.0.y);
        let path = directory.join(region_pos.filename());
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .await
            .expect("test region should reopen for corruption");
        file.seek(io::SeekFrom::Start(FILE_HEADER_SIZE as u64 + 7))
            .await
            .expect("status byte should be seekable");
        file.write_all(&[u8::MAX])
            .await
            .expect("status byte should be writable");
        file.flush().await.expect("status byte should be flushed");
        drop(file);

        let manager = RegionManager::new(&directory);
        let Err(error) = manager.acquire_chunk(pos).await else {
            panic!("invalid status byte must reject the region header");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let mut file = File::open(path)
            .await
            .expect("rejected region should remain readable");
        file.seek(io::SeekFrom::Start(FILE_HEADER_SIZE as u64 + 7))
            .await
            .expect("status byte should remain seekable");
        let mut status = [0];
        file.read_exact(&mut status)
            .await
            .expect("status byte should remain readable");
        assert_eq!(status[0], u8::MAX);

        fs::remove_dir_all(directory)
            .await
            .expect("test directory should be removable");
    }
}
