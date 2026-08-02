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

use std::env;
use std::fmt::Display;
use std::fs;
use std::num::NonZero;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::{Arc, Once};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use steel_core::behavior::init_behaviors;
use steel_core::block_entity::init_block_entities;
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
            chunk_workers: (available / 8).clamp(4, 16),
            main_workers: (available / 2).max(2),
            encoding_threads: (available / 8).clamp(2, 12),
            window_size: None,
            active_windows: None,
            ram_only: false,
            keep_storage: false,
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
"
    );
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
    fn shutdown(self, keep_storage: bool) {
        let Self {
            world,
            main_runtime,
            _chunk_runtime,
            storage,
        } = self;
        main_runtime.block_on(async {
            world.chunk_map.stop_generation_refill_loop();
            world.chunk_map.task_tracker.close();
            world.chunk_map.task_tracker.wait().await;
        });
        drop(world);
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
        harness.shutdown(options.keep_storage);

        let rate = total_chunks / elapsed.as_secs_f64();
        rates.push(rate);
        println!(
            "  rep {}: {:.2}s  {rate:.1} chunks/s",
            rep + 1,
            elapsed.as_secs_f64(),
        );
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
