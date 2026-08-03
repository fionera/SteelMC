//! End-to-end pregeneration throughput benchmark, without a server.
//!
//! The other benchmarks in this crate drive `GENERATION_PYRAMID` steps against
//! hand-built holders, which measures generation *compute*. This one measures
//! the whole pipeline -- tickets, scheduling epochs, the generation pool, the
//! unload path and region saves -- by running the production pregeneration
//! driver against a bare [`World`].
//!
//! It exists because the alternative was benchmarking through the real server
//! binary, which binds port 25565 and writes to a fixed `saves/` directory: a
//! leftover process from a previous run makes the next one exit at startup and
//! read as a zero result, and two runs cannot overlap at all. Here every run
//! gets its own storage directory under the target dir and binds nothing.
//!
//! Defaults mirror the server's own thread-count defaults, so numbers are
//! comparable to a `PREGEN_SIZE=<n>` run of the real binary.
//!
//! ```text
//! cargo bench -p steel-core --bench pregen --features benchmark-support -- --size 301 --reps 3
//! ```
//!
//! Criterion is deliberately not used. A run is seconds of wall clock with its
//! own warmup behaviour built in (the pipeline fills, then reaches steady
//! state), and criterion's resampling would multiply an already long run rather
//! than tell us anything its own reps do not.
//!
//! # `--digest`
//!
//! Throughput is not the only thing a scheduler change can move. Generation is
//! supposed to be a pure function of the seed, but two of its inputs are read
//! from a *neighbour's live state* -- `WorldGenAccessMode::capture` keys off a
//! neighbour's published status, and `chunk_for_lighting` admits a distance-2
//! holder only if it has already published `InitializeLight` -- so a chunk's
//! output can in principle depend on when its neighbours happened to run. The
//! stage-hash parity tests cannot see this: they drive pyramid steps against
//! hand-built holders and never run the scheduler at all.
//!
//! `--digest` closes that gap by hashing what a whole scheduled run actually
//! produced, so two runs can be compared. It prints two 128-bit numbers and a
//! chunk count:
//!
//! ```text
//! cargo bench -p steel-core --bench pregen --features benchmark-support -- \
//!     --size 301 --reps 1 --digest
//! ```
//!
//! Two digests rather than one because the interesting answer is not "did it
//! change" but *what* changed: both suspect paths feed the light layer, so
//! "blocks identical, light differs" and "blocks differ" call for completely
//! different responses, and one combined number could not tell them apart.

use std::env;
use std::fmt::{self, Display};
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process;
use std::str::FromStr;
use std::sync::{Arc, Once};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use steel_core::allocator::tune_for_throughput;
use steel_core::behavior::init_behaviors;
use steel_core::block_entity::init_block_entities;
use steel_core::chunk::status::ChunkStatus;
use steel_core::chunk_saver::{
    BLOCKS_PER_SECTION, CHUNK_TABLE_SIZE, FILE_HEADER_SIZE, FORMAT_VERSION, PersistentBlockState,
    PersistentChunk, PersistentLightSection, PersistentSection, REGION_MAGIC, REGION_SIZE,
    RegionHeader, RegionPos, SECTOR_SIZE, unpack_indices,
};
use steel_core::config::WorldStorageConfig;
use steel_core::entity::init_entities;
use steel_core::level_data::WorldGenerationSettings;
use steel_core::server::default_chunk_generation_threads;
use steel_core::server::pregen::pregen_area_for_benchmark;
use steel_core::world::{World, WorldConfig};
use steel_core::worldgen::WorldGeneratorRegistry;
use steel_registry::{REGISTRY, Registry, vanilla_dimension_types};
use steel_utils::types::{Difficulty, GameType};
use steel_utils::{ChunkPos, Identifier};
use tokio::runtime::{Builder, Runtime};
use tokio_util::sync::CancellationToken;
use toml::Value;
use toml::map::Map;

/// The seed the recorded pregeneration figures were measured on.
///
/// Terrain shape drives how much work a chunk is, so comparing two runs on
/// different seeds compares two different workloads. Pinning it keeps a result
/// comparable to every earlier measurement.
const DEFAULT_SEED: i64 = -9_091_483_014_810_473_238;
const DEFAULT_SIZE: i32 = 301;
const DEFAULT_REPS: usize = 3;

/// The allocator the server uses.
///
/// Registered here too because the benchmark is its own binary and would
/// otherwise measure the system allocator. Allocation shows up in this workload
/// -- pregeneration runs at hundreds of thousands of minor page faults per
/// second -- so the two must match for a benchmark result to predict a server
/// result.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

static INIT: Once = Once::new();

struct Options {
    size: i32,
    reps: usize,
    seed: i64,
    generation_threads: usize,
    chunk_workers: usize,
    main_workers: usize,
    encoding_threads: usize,
    window_size: Option<i32>,
    active_windows: Option<usize>,
    ram_only: bool,
    keep_storage: bool,
    digest: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let available = thread::available_parallelism().map_or(4, NonZero::get);
        let mut options = Self {
            size: DEFAULT_SIZE,
            reps: DEFAULT_REPS,
            seed: DEFAULT_SEED,
            // The server's own defaults, so a bench number and a real run number
            // describe the same configuration.
            generation_threads: default_chunk_generation_threads(available),
            // Mirrors the server's drive-aware sizing in `steel/src/main.rs`;
            // the two must agree or a benchmark number does not describe a
            // server run.
            chunk_workers: (available / 5).clamp(4, 24),
            main_workers: (available / 8).clamp(4, 16),
            encoding_threads: (available / 8).clamp(2, 12),
            window_size: None,
            active_windows: None,
            ram_only: false,
            keep_storage: false,
            digest: false,
        };

        // `cargo bench` passes its own flags through; ignore the ones libtest
        // would have consumed rather than failing on them.
        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("{arg} requires a value"))
            };
            match arg.as_str() {
                "--size" => options.size = parse(&value()?, "--size")?,
                "--reps" => options.reps = parse(&value()?, "--reps")?,
                "--seed" => options.seed = parse(&value()?, "--seed")?,
                "--gen-threads" => options.generation_threads = parse(&value()?, "--gen-threads")?,
                "--chunk-workers" => options.chunk_workers = parse(&value()?, "--chunk-workers")?,
                "--main-workers" => options.main_workers = parse(&value()?, "--main-workers")?,
                "--encode-threads" => {
                    options.encoding_threads = parse(&value()?, "--encode-threads")?;
                }
                "--window" => options.window_size = Some(parse(&value()?, "--window")?),
                "--windows" => options.active_windows = Some(parse(&value()?, "--windows")?),
                "--ram-only" => options.ram_only = true,
                "--keep-storage" => options.keep_storage = true,
                "--digest" => options.digest = true,
                "--bench" | "--test" => {}
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => return Err(format!("unknown argument {other} (try --help)")),
            }
        }

        if options.reps == 0 {
            return Err("--reps must be at least 1".to_owned());
        }
        // The digest reads back what the run persisted, and `--ram-only` never
        // persists: `RamOnlyStorage` keeps the prepared saves in a map with no
        // way in from outside the crate. Refusing here beats digesting nothing
        // and printing a chunk count of zero.
        if options.digest && options.ram_only {
            return Err(
                "--digest reads region files back, so it cannot be used with --ram-only".to_owned(),
            );
        }
        Ok(options)
    }
}

fn parse<T: FromStr>(value: &str, flag: &str) -> Result<T, String>
where
    T::Err: Display,
{
    value
        .parse()
        .map_err(|error| format!("{flag} takes a number: {error}"))
}

fn print_usage() {
    println!(
        "\
Headless pregeneration benchmark.

  --size N           chunk side length, odd (default {DEFAULT_SIZE})
  --reps N           runs to perform (default {DEFAULT_REPS})
  --seed N           world seed (default {DEFAULT_SEED})
  --gen-threads N    generation pool threads (default: the server's)
  --chunk-workers N  chunk runtime workers (default: the server's)
  --main-workers N   main runtime workers (default: the server's)
  --encode-threads N encoding pool threads (default: the server's)
  --window N         pregen window side length (default: the server's)
  --windows N        active pregen windows (default: the server's)
  --ram-only         skip region-file persistence entirely
  --keep-storage     do not delete the per-run storage directory
  --digest           hash the generated chunk content (see the module docs)
"
    );
}

/// Reads a MiB value out of `/proc/self/status`.
///
/// Memory is reported because a scheduler change's failure mode is often memory
/// rather than speed: a model that lets blocked chunks accumulate live state
/// shows up here long before it shows up in chunks/s. One such attempt reached
/// 13.6 GiB against a normal 8.4 GiB peak, and the throughput number never
/// arrived at all because no repetition finished.
///
/// **`peak` is process-lifetime and does not reset between repetitions**, and
/// with the allocator's purge disabled ([`tune_for_throughput`]) freed memory is
/// never returned to the OS, so it climbs across repetitions as each world is
/// built and torn down -- 8.5 GiB to 17 GiB over five, against a flat 7.9 GiB
/// with purging left on. That is the allocator holding pages, not a leak. Use
/// `--reps 1` when comparing peak memory between builds, and read `now` for
/// whether a repetition actually released what it allocated.
fn proc_status_mib(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(field))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024)
}

/// Peak resident set size in MiB, as the kernel has tracked it since startup.
fn peak_rss_mib() -> Option<u64> {
    proc_status_mib("VmHWM:")
}

/// Resident set size right now, in MiB.
fn current_rss_mib() -> Option<u64> {
    proc_status_mib("VmRSS:")
}

fn ensure_globals() {
    INIT.call_once(|| {
        let mut registry = Registry::new_vanilla();
        registry.freeze();
        let _ = REGISTRY.init(registry);
        init_behaviors();
        init_block_entities();
        init_entities();
    });
}

/// A storage directory unique to this run, removed when the run ends.
///
/// Under `target/` rather than the system temp dir: a 601x601 run writes
/// gigabytes of region files, and target is the directory already expected to
/// hold large build output on the same filesystem.
fn storage_root(rep: usize) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/bench-pregen")
        .join(format!("{}-{rep}-{unique}", process::id()))
}

struct Harness {
    world: Arc<World>,
    main_runtime: Runtime,
    /// Held so the world's runtime outlives it; dropped last.
    _chunk_runtime: Arc<Runtime>,
    storage: Option<PathBuf>,
}

fn build_harness(options: &Options, rep: usize) -> Result<Harness, String> {
    let chunk_runtime = Arc::new(
        Builder::new_multi_thread()
            .worker_threads(options.chunk_workers)
            .thread_name("chunk-worker")
            .enable_all()
            .build()
            .map_err(|error| format!("chunk runtime should start: {error}"))?,
    );
    let main_runtime = Builder::new_multi_thread()
        .worker_threads(options.main_workers)
        .thread_name("main-worker")
        .enable_all()
        .build()
        .map_err(|error| format!("main runtime should start: {error}"))?;

    let generation_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(options.generation_threads)
            .thread_name(|index| format!("rayon-gen-{index}"))
            .build()
            .map_err(|error| format!("generation pool should start: {error}"))?,
    );
    let encoding_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(options.encoding_threads)
            .thread_name(|index| format!("rayon-chunk-enc-{index}"))
            .build()
            .map_err(|error| format!("encoding pool should start: {error}"))?,
    );

    let generator_key = Identifier::vanilla_static("overworld");
    let generator_registry = WorldGeneratorRegistry::new_with_builtins()
        .map_err(|error| format!("built-in generators should register: {error}"))?;
    let generator_config = generator_registry
        .validate_config(&generator_key, &Value::Table(Map::new()))
        .map_err(|error| format!("overworld generator config should validate: {error}"))?;
    let generator_output = generator_registry
        .create(None, &generator_config, options.seed, generation_pool.clone())
        .map_err(|error| format!("overworld generator should build: {error}"))?;

    let (storage, storage_config) = if options.ram_only {
        (None, WorldStorageConfig::RamOnly)
    } else {
        let root = storage_root(rep);
        fs::create_dir_all(&root)
            .map_err(|error| format!("bench storage directory should be creatable: {error}"))?;
        let path = root.to_string_lossy().into_owned();
        (Some(root), WorldStorageConfig::Disk { path })
    };

    let generation_settings = WorldGenerationSettings::from_generator_config(
        generator_key,
        &generator_output.config,
        generator_output.dimension_type.key.clone(),
        generator_output.dimension_type.min_y,
        generator_output.dimension_type.height,
    );
    let is_flat = generator_output.is_flat;
    let sea_level = generator_output.sea_level;
    let dimension_type = generator_output.dimension_type;

    let world = main_runtime
        .block_on(World::new_with_config_and_encoding_pool(
            Arc::clone(&chunk_runtime),
            vanilla_dimension_types::OVERWORLD.key.clone(),
            dimension_type,
            options.seed,
            WorldConfig {
                storage: storage_config,
                // Ephemeral: the seed is passed explicitly above, and a bench
                // run has no level data worth carrying to the next one.
                level_data_path: None,
                generator: Arc::new(generator_output.generator),
                generation_settings,
                view_distance: 10,
                simulation_distance: 10,
                max_chained_neighbor_updates: 1_000_000,
                compression: None,
                is_flat,
                sea_level,
                default_gamemode: GameType::Survival,
                difficulty: Difficulty::Normal,
            },
            generation_pool,
            encoding_pool,
        ))
        .map_err(|error| format!("bench world should initialize: {error}"))?;

    Ok(Harness {
        world,
        main_runtime,
        _chunk_runtime: chunk_runtime,
        storage,
    })
}

impl Harness {
    fn run(&self, options: &Options) -> Result<Duration, String> {
        let cancel_token = CancellationToken::new();
        self.main_runtime
            .block_on(pregen_area_for_benchmark(
                &self.world,
                ChunkPos::new(0, 0),
                options.size,
                options.window_size,
                options.active_windows,
                &cancel_token,
            ))?
            .ok_or_else(|| "pregeneration was cancelled".to_owned())
    }

    /// Quiesces the world the way server shutdown does.
    ///
    /// Without this the next repetition starts while the previous world's
    /// generation and save tasks are still running, which shows up as a slower
    /// second run rather than as an error.
    ///
    /// Separate from [`Self::shutdown`] because the digest has to run against a
    /// world that is finished but not yet dropped: it needs the save path, and
    /// it needs every in-flight save to have landed first.
    fn quiesce(&self) {
        self.main_runtime.block_on(async {
            self.world.chunk_map.stop_generation_refill_loop();
            self.world.chunk_map.task_tracker.close();
            self.world.chunk_map.task_tracker.wait().await;
        });
    }

    /// Digests the chunk content this run produced.
    ///
    /// Call after [`Self::quiesce`]. The final `save_all_chunks` is what makes
    /// the region files a complete record: a pregeneration ends with the last
    /// windows and their halo still resident, and a resident chunk that was
    /// never dirty-unloaded has never been written. Without this the digest
    /// would silently cover only the chunks the unload path happened to reach,
    /// which is exactly the timing-dependent set the digest exists to rule out.
    /// It also closes the region files, which is what flushes their headers --
    /// an unflushed header is an empty chunk table.
    fn content_digest(&self, options: &Options) -> Result<ContentDigest, String> {
        let Some(storage) = self.storage.as_ref() else {
            return Err("digest needs a disk-backed run".to_owned());
        };
        self.main_runtime
            .block_on(self.world.save_all_chunks())
            .map_err(|error| format!("final chunk save should succeed: {error}"))?;
        digest_storage(storage, options.size / 2)
    }

    fn shutdown(self, keep_storage: bool) {
        self.quiesce();
        let Self {
            world,
            main_runtime,
            _chunk_runtime,
            storage,
        } = self;
        drop(world);
        // Dropped after the world, as scope order did when this method still
        // ran the quiesce itself: the world's tasks live on this runtime.
        drop(main_runtime);
        match storage {
            Some(storage) if keep_storage => {
                println!("  storage kept at {}", storage.display());
            }
            Some(storage) => {
                let _ = fs::remove_dir_all(storage);
            }
            None => {}
        }
    }
}

fn main() {
    // Production does the same before it generates anything; without it the
    // benchmark measures a differently-tuned allocator than the server.
    tune_for_throughput();

    let options = match Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(2);
        }
    };

    ensure_globals();

    let total_chunks = f64::from(options.size) * f64::from(options.size);
    println!(
        "pregen {size}x{size} ({total} chunks), seed {seed}, {gen} generation threads, \
{chunk} chunk workers, {main} main workers, {store}",
        size = options.size,
        total = total_chunks as u64,
        seed = options.seed,
        gen = options.generation_threads,
        chunk = options.chunk_workers,
        main = options.main_workers,
        store = if options.ram_only { "ram-only" } else { "disk" },
    );

    let mut rates = Vec::with_capacity(options.reps);
    for rep in 0..options.reps {
        let harness = match build_harness(&options, rep) {
            Ok(harness) => harness,
            Err(error) => {
                eprintln!("error: {error}");
                process::exit(1);
            }
        };
        let elapsed = match harness.run(&options) {
            Ok(elapsed) => elapsed,
            Err(error) => {
                eprintln!("error: {error}");
                process::exit(1);
            }
        };
        // Everything below runs outside the timed section, and none of it runs
        // at all without `--digest`, so the flag cannot move a throughput
        // number.
        let digest = options.digest.then(|| {
            harness.quiesce();
            match harness.content_digest(&options) {
                Ok(digest) => digest,
                Err(error) => {
                    eprintln!("error: {error}");
                    process::exit(1);
                }
            }
        });
        harness.shutdown(options.keep_storage);

        let rate = total_chunks / elapsed.as_secs_f64();
        rates.push(rate);
        let peak = match (peak_rss_mib(), current_rss_mib()) {
            (Some(peak), Some(now)) => format!("  RSS {now} MiB (peak {peak})"),
            (Some(peak), None) => format!("  peak RSS {peak} MiB"),
            _ => String::new(),
        };
        println!(
            "  rep {}: {:.2}s  {rate:.1} chunks/s{peak}",
            rep + 1,
            elapsed.as_secs_f64(),
        );
        if let Some(digest) = digest {
            println!("{digest}");
        }
    }

    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let min = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let max = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "mean {mean:.1} chunks/s  (min {min:.1}, max {max:.1}, spread {:.2}%)",
        if mean > 0.0 {
            (max - min) / mean * 100.0
        } else {
            0.0
        },
    );
}

// ---------------------------------------------------------------------------
// Content digest
// ---------------------------------------------------------------------------

/// A canonical, order-independent digest of the chunk content a run produced.
///
/// Order-independent because chunks are saved in whatever order the unload path
/// reaches them, which is thread timing, so no sequential hash over the saved
/// stream could ever be stable. Per-chunk hashes are therefore combined with
/// `wrapping_add`, which does not care what order the terms arrive in.
///
/// `wrapping_add` rather than XOR for one reason: XOR cancels. A chunk counted
/// twice -- a region file listed twice, a position digested from two entries --
/// would vanish from an XOR accumulator and leave a digest that still looks
/// like a clean match. Adding cannot hide a duplicate, and `chunks` catches it
/// besides.
#[derive(Clone, Copy, Default)]
struct ContentDigest {
    /// Sum over chunks of a hash of position, persisted status, and every block
    /// state in the chunk.
    blocks: u128,
    /// Sum over chunks of a hash of position, persisted status, and the chunk's
    /// persisted sky and block light.
    light: u128,
    /// How many chunks went into the two sums.
    ///
    /// Reported because a digest is only evidence together with its population:
    /// two runs that both digested nothing agree perfectly.
    chunks: u64,
    /// Digested chunks whose persisted status was below `Full`.
    ///
    /// Expected to be zero: every chunk in the requested area is requested at
    /// `Full` and the run does not finish until all of them report ready. A
    /// non-zero count means the run did not persist what it claimed to, and the
    /// digests describe something other than a finished area.
    below_full: u64,
    /// Chunks found on disk outside the requested area, which are not digested.
    ///
    /// These are the dependency halo. They are generated to whatever status
    /// their neighbours needed and are unloaded when the window that pulled
    /// them in retires, so both their status and their content legitimately
    /// depend on window scheduling. Digesting them would report a difference
    /// that says nothing about whether generation is deterministic.
    halo: u64,
}

impl ContentDigest {
    const fn merge(self, other: Self) -> Self {
        Self {
            blocks: self.blocks.wrapping_add(other.blocks),
            light: self.light.wrapping_add(other.light),
            chunks: self.chunks + other.chunks,
            below_full: self.below_full + other.below_full,
            halo: self.halo + other.halo,
        }
    }

    /// Folds one decoded chunk into the accumulator.
    fn add(&mut self, pos: ChunkPos, status: ChunkStatus, chunk: &PersistentChunk<'_>) {
        self.blocks = self
            .blocks
            .wrapping_add(spread(block_hash(pos, status, chunk)));
        self.light = self
            .light
            .wrapping_add(spread(light_hash(pos, status, chunk)));
        self.chunks += 1;
        if status != ChunkStatus::Full {
            self.below_full += 1;
        }
    }
}

impl Display for ContentDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "  digest blocks {:032x}  light {:032x}\n  digest over {} chunks ({} below Full, {} halo chunks skipped)",
            self.blocks, self.light, self.chunks, self.below_full, self.halo,
        )
    }
}

/// Odd multiplier for [`fold`], the golden-ratio constant.
const FOLD_PRIME: u64 = 0x9E37_79B9_7F4A_7C15;
/// Distinct starting states, so that the same word stream folded as blocks and
/// as light cannot produce the same hash.
const BLOCK_SEED: u64 = 0x5F1D_2C3B_4A59_6877;
const LIGHT_SEED: u64 = 0x1357_9BDF_2468_ACE0;
const STATE_SEED: u64 = 0x0F1E_2D3C_4B5A_6978;
/// Second lane key for [`spread`].
const SPREAD_KEY: u64 = 0xD6E8_FEB8_6659_FD93;
/// Folded in place of a block state a section's palette does not describe, and
/// in place of light bytes a section does not carry. Never a real hash by
/// construction: real ones come out of [`fold`], which cannot be given a value.
const ABSENT: u64 = 0xA95E_47B0_C31D_6E82;

/// One step of a per-chunk hash.
///
/// Order-sensitive on purpose. Within a chunk the digest has to notice a block
/// *moving*, not only a block changing, so every word's position in the stream
/// has to matter -- which is why the per-chunk hash is a fold and only the
/// combination across chunks is commutative.
///
/// Multiplying by an odd constant is a bijection on `u64`, so no word can be
/// swallowed; the rotate brings the entropy the multiply pushes into the high
/// bits back down where the next xor can reach it.
///
/// This is a change detector, not a commitment. There is no adversary here
/// constructing collisions, only two runs of the same generator, so 64 bits of
/// avalanche per chunk is ample.
#[inline]
const fn fold(state: u64, word: u64) -> u64 {
    (state ^ word).wrapping_mul(FOLD_PRIME).rotate_left(29)
}

/// Folds a byte string, length first so that concatenation is not ambiguous.
fn fold_bytes(state: u64, bytes: &[u8]) -> u64 {
    let mut state = fold(state, bytes.len() as u64);
    let (words, rest) = bytes.as_chunks::<8>();
    for word in words {
        state = fold(state, u64::from_le_bytes(*word));
    }
    let mut tail = [0u8; 8];
    tail[..rest.len()].copy_from_slice(rest);
    fold(state, u64::from_le_bytes(tail))
}

/// `SplitMix64`'s finalizer, used to spread one 64-bit hash across 128 bits.
const fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(FOLD_PRIME);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Widens a per-chunk hash to the accumulator's width.
///
/// The fold runs in 64 bits because it runs once per block -- around 10^10
/// times over a 301x301 area -- and a 128-bit multiply there would cost real
/// wall clock for no benefit: at 90,601 chunks the chance that two *different*
/// chunks collide in 64 bits is about 2*10^-10. The accumulator is 128 bits so
/// that summing ~10^5 of those cannot itself lose a difference.
const fn spread(hash: u64) -> u128 {
    ((splitmix64(hash) as u128) << 64) | splitmix64(hash ^ SPREAD_KEY) as u128
}

/// Digests every region file under a run's storage directory.
///
/// Region files are read back rather than hooked into the save path because the
/// save path is the wrong shape for this: a chunk can be saved more than once
/// (dirty, unloaded, revived, re-saved), and a digest that counted a chunk
/// twice would not be a digest of the world. The region file holds exactly one
/// entry per position -- the last one written -- which is the state the run
/// ended in.
///
/// Files are processed in parallel and combined with the same commutative
/// operation used within a file, so the result does not depend on how the work
/// was split.
fn digest_storage(root: &Path, radius: i32) -> Result<ContentDigest, String> {
    let mut regions = Vec::new();
    for entry in fs::read_dir(root)
        .map_err(|error| format!("storage directory should be readable: {error}"))?
    {
        let entry = entry.map_err(|error| format!("storage entry should be readable: {error}"))?;
        let path = entry.path();
        let Some(pos) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(parse_region_name)
        else {
            continue;
        };
        regions.push((path, pos));
    }

    // An empty storage directory is a harness failure, not a result. Reporting
    // it as a digest over zero chunks would be reporting a match with anything.
    if regions.is_empty() {
        return Err(format!("no region files under {}", root.display()));
    }

    regions
        .par_iter()
        .map(|(path, pos)| digest_region(path, *pos, radius))
        .try_reduce(ContentDigest::default, |left, right| Ok(left.merge(right)))
}

/// Parses `r.<x>.<z>.srg`, the region file naming [`RegionPos::filename`] uses.
fn parse_region_name(name: &str) -> Option<RegionPos> {
    let coordinates = name.strip_prefix("r.")?.strip_suffix(".srg")?;
    let (x, z) = coordinates.split_once('.')?;
    Some(RegionPos::new(x.parse().ok()?, z.parse().ok()?))
}

/// Digests one region file.
///
/// Decoding goes through the crate's own format types -- the chunk table, the
/// zstd frame, the wincode schema, the section bit packing -- so the digest
/// sees precisely what the loader would see. Nothing about the *file* is
/// hashed: sector layout follows save order, which follows thread timing, so
/// two runs that generated identical worlds still hold different bytes.
fn digest_region(path: &Path, region: RegionPos, radius: i32) -> Result<ContentDigest, String> {
    let describe = |what: &str, error: &dyn Display| format!("{}: {what}: {error}", path.display());

    let mut file = File::open(path).map_err(|error| describe("open", &error))?;
    let mut file_header = [0u8; FILE_HEADER_SIZE];
    file.read_exact(&mut file_header)
        .map_err(|error| describe("read file header", &error))?;
    if file_header[..4] != REGION_MAGIC {
        return Err(describe("bad magic", &"not a region file"));
    }
    let version = u16::from_le_bytes([file_header[4], file_header[5]]);
    if version != FORMAT_VERSION {
        return Err(describe(
            "format version",
            &format!("file is version {version}, this build writes {FORMAT_VERSION}"),
        ));
    }

    let mut table = vec![0u8; CHUNK_TABLE_SIZE];
    file.read_exact(&mut table)
        .map_err(|error| describe("read chunk table", &error))?;
    let header = RegionHeader::from_bytes(&table).map_err(|index| {
        describe(
            "chunk table",
            &format!("entry {index} has an invalid status"),
        )
    })?;

    let mut digest = ContentDigest::default();
    let mut compressed = Vec::new();
    for (index, entry) in header.entries.iter().enumerate() {
        if !entry.exists() {
            continue;
        }
        let (local_x, local_z) = RegionHeader::index_to_local(index);
        let pos = ChunkPos::new(
            region.x * REGION_SIZE as i32 + local_x as i32,
            region.z * REGION_SIZE as i32 + local_z as i32,
        );
        // The requested area is the square of that radius around (0, 0); see
        // `ContentDigest::halo` for why everything outside it is left out.
        if pos.0.x.abs() > radius || pos.0.y.abs() > radius {
            digest.halo += 1;
            continue;
        }

        compressed.resize(entry.size_bytes as usize, 0);
        file.seek(SeekFrom::Start(
            u64::from(entry.sector_offset) * SECTOR_SIZE as u64,
        ))
        .map_err(|error| describe("seek to chunk", &error))?;
        file.read_exact(&mut compressed)
            .map_err(|error| describe("read chunk", &error))?;
        let raw = zstd::decode_all(&compressed[..])
            .map_err(|error| describe("decompress chunk", &error))?;
        let persistent: PersistentChunk<'_> =
            wincode::deserialize(&raw).map_err(|error| describe("decode chunk", &error))?;
        digest.add(pos, entry.status, &persistent);
    }

    Ok(digest)
}

/// Hashes a chunk's position, persisted status, and every block state it holds.
///
/// Blocks are folded in the order the format packs them, section by section
/// ascending and then in the section's own packed order, which is a fixed
/// function of (x, y, z). Every section contributes exactly
/// `BLOCKS_PER_SECTION` words whichever variant it is stored as, because the
/// storage variant is not canonical and the content is: a section whose blocks
/// happen to be uniform may be written `Homogeneous`, or `Heterogeneous` with a
/// palette entry whose count has fallen to zero. Folding the representation
/// would report those two as different worlds. Folding the decoded blocks
/// reports them as what they are.
///
/// Block states are folded as an identity hashed from the block's name and its
/// sorted properties, never as a palette index: palettes are built in first-seen
/// order, so an index means nothing outside the chunk that produced it.
fn block_hash(pos: ChunkPos, status: ChunkStatus, chunk: &PersistentChunk<'_>) -> u64 {
    let mut state = fold(BLOCK_SEED, pos.0.x as u64);
    state = fold(state, pos.0.y as u64);
    state = fold(state, status.get_index() as u64);
    state = fold(state, chunk.sections.len() as u64);

    let identities: Vec<u64> = chunk
        .block_states
        .iter()
        .map(block_state_identity)
        .collect();
    let identity_of = |index: u16| identities.get(index as usize).copied().unwrap_or(ABSENT);

    let mut section_identities: Vec<u64> = Vec::new();
    for (index, section) in chunk.sections.iter().enumerate() {
        state = fold(state, index as u64);
        match section {
            PersistentSection::Homogeneous { block_state, .. } => {
                let identity = identity_of(*block_state);
                for _ in 0..BLOCKS_PER_SECTION {
                    state = fold(state, identity);
                }
            }
            PersistentSection::Heterogeneous {
                palette,
                bits_per_entry,
                block_data,
                ..
            } => {
                section_identities.clear();
                section_identities.extend(palette.iter().copied().map(identity_of));

                // `unpack_indices` divides by the entry width, so a corrupt one
                // would divide by zero. Only power-of-two widths are ever
                // written.
                let mut folded = 0;
                if matches!(bits_per_entry, 1 | 2 | 4 | 8 | 16) {
                    for packed in
                        unpack_indices(block_data, *bits_per_entry).take(BLOCKS_PER_SECTION)
                    {
                        let identity = section_identities
                            .get(packed as usize)
                            .copied()
                            .unwrap_or(ABSENT);
                        state = fold(state, identity);
                        folded += 1;
                    }
                }
                // A section that decoded short still contributes a full
                // section's worth of words, so the sections after it stay
                // aligned with the same sections of another run.
                for _ in folded..BLOCKS_PER_SECTION {
                    state = fold(state, ABSENT);
                }
            }
        }
    }

    state
}

/// Hashes a block state into a value that means the same thing in every chunk.
///
/// Properties are sorted first because their order in the persisted state comes
/// from how the registry enumerated them, not from the state itself.
fn block_state_identity(state: &PersistentBlockState<'_>) -> u64 {
    let mut hash = fold_bytes(STATE_SEED, state.name.namespace.as_bytes());
    hash = fold_bytes(hash, state.name.path.as_bytes());
    let mut properties = state.properties.clone();
    properties.sort_unstable();
    hash = fold(hash, properties.len() as u64);
    for (key, value) in properties {
        hash = fold_bytes(fold_bytes(hash, key.as_bytes()), value.as_bytes());
    }
    hash
}

/// Hashes a chunk's persisted sky and block light.
///
/// Kept apart from [`block_hash`] because that separation is the whole point:
/// the paths under suspicion -- a neighbour's published status deciding a
/// generation access mode, and light's distance-2 admission rule -- would show
/// up here and not there, and a single number could not say so.
///
/// The layer's own section list is folded as stored, including each section's
/// index and its kind. Kind matters: `Uninitialized` and an all-zero
/// `Initialized` section describe the same light through different
/// serializations, and this reports them as different. That is the safe
/// direction of over-sensitivity -- it can only turn "identical" into
/// "different, go look", never the reverse -- and a chunk's light
/// representation is part of what a scheduler rewrite has to preserve anyway.
fn light_hash(pos: ChunkPos, status: ChunkStatus, chunk: &PersistentChunk<'_>) -> u64 {
    let mut state = fold(LIGHT_SEED, pos.0.x as u64);
    state = fold(state, pos.0.y as u64);
    state = fold(state, status.get_index() as u64);

    for (layer, sections) in [(0u64, &chunk.light.block), (1, &chunk.light.sky)] {
        state = fold(state, layer);
        state = fold(state, sections.len() as u64);
        for section in sections {
            let (kind, data) = match section {
                PersistentLightSection::Uninitialized { .. } => (0u64, None),
                PersistentLightSection::Initialized { data, .. } => (1, Some(data)),
                PersistentLightSection::Internal { data, .. } => (2, Some(data)),
            };
            state = fold(state, kind);
            state = fold(state, u64::from(section.section_index()));
            state = match data {
                Some(data) => fold_bytes(state, data),
                None => fold(state, ABSENT),
            };
        }
    }

    state
}
