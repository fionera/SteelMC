//! Headless generation engine.
//!
//! Drives Steel's chunk pyramid outside the server: no networking, no tick loop,
//! no players. The shape follows `steel-core/benches/worldgen.rs`, which already
//! stands the pipeline up in-process; this is that made reusable.

use std::num::NonZero;
use std::sync::{Arc, OnceLock};
use std::thread::available_parallelism;

use futures::future::join_all;
use steel_core::behavior::init_behaviors;
use steel_core::block_entity::init_block_entities;
use steel_core::chunk::chunk_access::ChunkStatus;
use steel_core::chunk::chunk_generation_task::StaticCache2D;
use steel_core::chunk::chunk_holder::ChunkHolder;
use steel_core::chunk::chunk_map::ChunkMap;
use steel_core::chunk::chunk_pyramid::{ChunkStep, GENERATION_PYRAMID};
use steel_core::chunk::chunk_ticket_manager::ChunkTicketLevel;
use steel_core::entity::init_entities;
use steel_core::level_data::WorldGenerationSettings;
use steel_core::world::{World, WorldConfig, WorldStorageConfig};
use steel_core::worldgen::WorldGeneratorRegistry;
use steel_registry::{REGISTRY, Registry};
use steel_utils::types::{Difficulty, GameType};
use steel_utils::{ChunkPos, Identifier};
use tokio::runtime::{Builder as TokioBuilder, Runtime as TokioRuntime};
use toml::Value;
use toml::map::Map;

use crate::snapshot;

/// Stack size for generation worker threads.
///
/// The transpiled density functions nest deeply enough that Steel's own tests
/// spawn 16 MB threads. Host (JVM) threads are typically 1 MB, which is why
/// generation must never run on one — see `swg_generate_batch`.
const GENERATION_STACK_SIZE: usize = 16 * 1024 * 1024;

/// Guards the process-global registries.
///
/// `init_entities` panics if called twice, and `REGISTRY` is a write-once
/// `OnceLock` whose entries are `Box::leak`ed. A host that reloads the mod or
/// opens a second world must not re-run any of it.
static RUNTIME_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initializes Steel's process-global registries. Idempotent.
///
/// # Errors
/// Returns the original failure message if a previous call failed; the globals
/// cannot be retried once poisoned.
pub fn init_runtime() -> Result<(), String> {
    RUNTIME_INIT
        .get_or_init(|| {
            let mut registry = Registry::new_vanilla();
            registry.freeze();

            // `init` returns Err if something already installed a registry. That
            // is not fatal on its own -- an embedding host may have done it --
            // but the behavior tables below must still run exactly once, which
            // the OnceLock guarantees.
            let _ = REGISTRY.init(registry);

            init_behaviors();
            init_block_entities();
            init_entities();
            Ok(())
        })
        .clone()
}

/// How a world handle was configured.
pub struct WorldSpec {
    /// Generator identifier, e.g. `minecraft:overworld`.
    pub generator: Identifier,
    /// World seed.
    pub seed: i64,
    /// Number of generation worker threads. Zero means one per available core.
    pub threads: usize,
}

/// A live generation world.
///
/// Owns its own rayon pool; generation always runs there and never on a caller
/// thread, so stack size is this crate's problem rather than the host's.
pub struct GenerationWorld {
    /// Drives the step futures. Generation itself runs on `pool`.
    tokio: Arc<TokioRuntime>,
    /// Kept alive so the `Weak<World>` inside the chunk map stays upgradeable.
    _world: Arc<World>,
    chunk_map: Arc<ChunkMap>,
    pool: Arc<rayon::ThreadPool>,
    min_y: i32,
    height: i32,
}

impl GenerationWorld {
    /// Stands up a RAM-only world for `spec`.
    ///
    /// # Errors
    /// Returns a message if the generator identifier is unknown, its config is
    /// rejected, or the world fails to build.
    pub fn open(spec: &WorldSpec) -> Result<Self, String> {
        init_runtime()?;

        let threads = if spec.threads == 0 {
            available_parallelism().map_or(1, NonZero::get)
        } else {
            spec.threads
        };
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(GENERATION_STACK_SIZE)
                .thread_name(|index| format!("steel-worldgen-{index}"))
                .build()
                .map_err(|err| format!("generation pool: {err}"))?,
        );

        let registry = WorldGeneratorRegistry::new_with_builtins()
            .map_err(|err| format!("built-in generators: {err}"))?;
        // The three vanilla noise generators reject any non-empty config, so an
        // empty table is the only thing they accept.
        let config = registry
            .validate_config(&spec.generator, &Value::Table(Map::new()))
            .map_err(|err| format!("generator {}: {err}", spec.generator))?;
        let output = registry
            .create(None, &config, spec.seed, pool.clone())
            .map_err(|err| format!("generator {}: {err}", spec.generator))?;

        let dimension = output.dimension_type;
        let (min_y, height) = (dimension.min_y, dimension.height);

        let generation_settings = WorldGenerationSettings::from_generator_config(
            spec.generator.clone(),
            &output.config,
            dimension.key.clone(),
            min_y,
            height,
        );

        // Multi-thread rather than current-thread: hosts call `generate` from
        // several worker threads at once, and each blocks on this runtime.
        // Generation still happens on `pool`; these workers only drive futures.
        let tokio = Arc::new(
            TokioBuilder::new_multi_thread()
                .worker_threads(2)
                .thread_name("steel-worldgen-rt")
                .enable_all()
                .build()
                .map_err(|err| format!("tokio runtime: {err}"))?,
        );

        let world_config = WorldConfig {
            storage: WorldStorageConfig::RamOnly,
            level_data_path: None,
            generator: Arc::new(output.generator),
            generation_settings,
            view_distance: 10,
            simulation_distance: 10,
            max_chained_neighbor_updates: 1_000_000,
            compression: None,
            is_flat: output.is_flat,
            sea_level: output.sea_level,
            default_gamemode: GameType::Survival,
            difficulty: Difficulty::Normal,
        };

        let world_key = Identifier::new("steel_worldgen_ffi", spec.generator.path.clone());
        let world = tokio
            .block_on(World::new_with_config(
                tokio.clone(),
                world_key,
                dimension,
                spec.seed,
                world_config,
                pool.clone(),
            ))
            .map_err(|err| format!("world: {err}"))?;

        let chunk_map = world.chunk_map.clone();

        Ok(Self {
            tokio,
            _world: world,
            chunk_map,
            pool,
            min_y,
            height,
        })
    }

    /// Drives `centers` to `target`, returning the holder for each requested
    /// position in the order given.
    ///
    /// Neighbours are generated as far as the pyramid requires and then dropped;
    /// only the requested chunks are returned.
    ///
    /// # Errors
    /// Returns a message if `centers` is empty or the region is too large to
    /// address.
    pub fn generate(
        &self,
        centers: &[ChunkPos],
        target: ChunkStatus,
    ) -> Result<Vec<Arc<ChunkHolder>>, String> {
        if centers.is_empty() {
            return Err("no chunk positions requested".to_owned());
        }

        let cache = self.build_cache(centers, target)?;

        // Walk the pyramid one status at a time. Each status is generated across
        // the radius the *target* step accumulated for it, so by the time a step
        // runs, every chunk it may read is already at the required status.
        let target_step = GENERATION_PYRAMID.get_step_to(target);
        let mut status = ChunkStatus::Empty;
        loop {
            let positions = positions_for_status(centers, target_step, status);
            self.run_stage(&cache, &positions, status)?;

            if status == target {
                break;
            }
            match status.next() {
                Some(next) => status = next,
                // `target` is a real variant, so the chain always reaches it
                // before running out. Guard anyway rather than loop forever.
                None => break,
            }
        }

        Ok(centers
            .iter()
            .map(|pos| cache.get(pos.0.x, pos.0.y).clone())
            .collect())
    }

    /// Drives `centers` to `target` and encodes them into a snapshot buffer.
    ///
    /// Encoding happens here rather than in the caller so the chunk read guards
    /// never escape this crate.
    ///
    /// # Errors
    /// Returns a message if generation fails, or if a chunk holds a block state
    /// or biome with no registry entry.
    pub fn generate_snapshot(
        &self,
        centers: &[ChunkPos],
        target: ChunkStatus,
    ) -> Result<Vec<u8>, String> {
        let holders = self.generate(centers, target)?;

        // Hold every read guard for the duration of the encode so a concurrent
        // call cannot advance a chunk mid-snapshot.
        let guards = holders
            .iter()
            .map(|holder| {
                holder
                    .try_chunk(target)
                    .ok_or_else(|| format!("chunk {:?} did not reach {target:?}", holder.get_pos()))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let chunks: Vec<_> = guards.iter().map(|guard| (&**guard, target)).collect();
        snapshot::encode(&chunks, self.min_y).map_err(|err| err.to_string())
    }

    /// Builds the holder cache covering `centers` plus the dependency radius the
    /// target step needs.
    fn build_cache(
        &self,
        centers: &[ChunkPos],
        target: ChunkStatus,
    ) -> Result<Arc<StaticCache2D<Arc<ChunkHolder>>>, String> {
        let target_step = GENERATION_PYRAMID.get_step_to(target);
        let dependency_radius =
            i32::try_from(target_step.get_accumulated_radius_of(ChunkStatus::Empty))
                .map_err(|_| "dependency radius overflow".to_owned())?;

        let (min_x, max_x) = min_max(centers.iter().map(|pos| pos.0.x));
        let (min_z, max_z) = min_max(centers.iter().map(|pos| pos.0.y));

        // StaticCache2D is a square centred on one point, so cover the batch's
        // bounding box and grow by the dependency radius.
        let center_x = min_x.midpoint(max_x);
        let center_z = min_z.midpoint(max_z);
        let half_extent = ((max_x - center_x).max(center_x - min_x))
            .max((max_z - center_z).max(center_z - min_z));
        let radius = half_extent
            .checked_add(dependency_radius)
            .ok_or_else(|| "requested region is too large".to_owned())?;

        let (min_y, height) = (self.min_y, self.height);
        Ok(Arc::new(StaticCache2D::create(
            center_x,
            center_z,
            radius,
            move |x, z| {
                Arc::new(ChunkHolder::new(
                    ChunkPos::new(x, z),
                    ChunkTicketLevel::STRONGEST,
                    None,
                    min_y,
                    height,
                ))
            },
        )))
    }

    /// Runs one pyramid step across `positions` and waits for it to complete.
    ///
    /// Goes through [`ChunkHolder::apply_step`] rather than invoking
    /// `step.task` directly: the task alone stores the chunk without publishing
    /// its status, because the scheduler is what publishes once the step
    /// finishes. Calling the task directly leaves every later step failing with
    /// "Chunk not found at status ...". `apply_step` also reserves the light
    /// work window that the `Light` step requires.
    ///
    /// Steps are dispatched onto the generation pool by `apply_step` itself, so
    /// the chunks in one step run concurrently; this only awaits them.
    ///
    /// # Errors
    /// Returns the positions that did not complete.
    fn run_stage(
        &self,
        cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
        positions: &[ChunkPos],
        status: ChunkStatus,
    ) -> Result<(), String> {
        let step = GENERATION_PYRAMID.get_step_to(status);

        let incomplete = self.tokio.block_on(async {
            let pending = positions
                .iter()
                .filter_map(|pos| {
                    let holder = cache.get(pos.0.x, pos.0.y).clone();
                    holder.apply_step(step, &self.chunk_map, cache, self.pool.clone())
                })
                .collect::<Vec<_>>();

            join_all(pending)
                .await
                .iter()
                .filter(|result| result.is_none())
                .count()
        });

        if incomplete > 0 {
            return Err(format!("{incomplete} chunk(s) failed to reach {status:?}"));
        }
        Ok(())
    }

    /// The tokio runtime backing this world's chunk IO.
    #[must_use]
    pub const fn tokio(&self) -> &Arc<TokioRuntime> {
        &self.tokio
    }

    /// Lowest block Y in this dimension.
    #[must_use]
    pub const fn min_y(&self) -> i32 {
        self.min_y
    }

    /// Build height of this dimension, in blocks.
    #[must_use]
    pub const fn height(&self) -> i32 {
        self.height
    }
}

/// Positions that must reach `status` for `target_step` to succeed at `centers`.
fn positions_for_status(
    centers: &[ChunkPos],
    target_step: &ChunkStep,
    status: ChunkStatus,
) -> Vec<ChunkPos> {
    let radius = i32::try_from(target_step.get_accumulated_radius_of(status)).unwrap_or(0);
    let mut positions = Vec::new();

    for center in centers {
        for z in (center.0.y - radius)..=(center.0.y + radius) {
            for x in (center.0.x - radius)..=(center.0.x + radius) {
                positions.push(ChunkPos::new(x, z));
            }
        }
    }

    // Overlapping centres produce duplicates; running a step twice on one chunk
    // would double-apply it.
    positions.sort_unstable_by_key(|pos| (pos.0.y, pos.0.x));
    positions.dedup();
    positions
}

/// Minimum and maximum of a non-empty iterator.
fn min_max(values: impl Iterator<Item = i32>) -> (i32, i32) {
    values.fold((i32::MAX, i32::MIN), |(lo, hi), value| {
        (lo.min(value), hi.max(value))
    })
}
