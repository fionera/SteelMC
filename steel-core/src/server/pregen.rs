//! Startup pregeneration for the server default world.

use std::collections::VecDeque;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use steel_utils::{ChunkPos, SectionPos};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::chunk::chunk_pyramid::GENERATION_PYRAMID;
use crate::chunk::chunk_map::ChunkMapSchedulingTimings;
use crate::chunk::chunk_request::{
    ChunkRequest, ChunkRequestHandle, ChunkRequestState, ChunkTicketKind,
};
use crate::chunk::status::ChunkStatus;
use crate::server::Server;
use crate::world::World;

#[cfg(feature = "slow_chunk_gen")]
use crate::chunk::chunk_holder::SLOW_CHUNK_GEN;
#[cfg(feature = "slow_chunk_gen")]
use std::sync::atomic::Ordering;

const PREGEN_SIZE_ENV: &str = "PREGEN_SIZE";
const PREGEN_WINDOW_SIZE_ENV: &str = "PREGEN_WINDOW_SIZE";
const PREGEN_ACTIVE_WINDOWS_ENV: &str = "PREGEN_ACTIVE_WINDOWS";
const VANILLA_PLAYER_SPAWN_SIZE_CHUNKS: i32 = 7;
const DEFAULT_PREGEN_WINDOW_SIZE: i32 = 32;
/// How many windows may be generating at once.
///
/// This is the pregeneration pipeline's depth. Each window is
/// `window_size * window_size` target chunks, and a new one is admitted only as
/// an active one finishes, so this bounds how much work the generation threads
/// can see at any moment. Too shallow and the tail of every window -- when a
/// handful of chunks remain and the rest of the pool has nothing to do -- is
/// paid with idle threads.
///
/// It was 2, which starves the pool badly. A scheduler trace of the generation
/// threads showed 90.8% of their context switches were blocking sleeps in
/// rayon's idle path -- they were not contending, they had no work -- while
/// disk waits were 14 events out of 155,697. Raising the depth to 16 measured
/// +18.5% throughput over a 90,601-chunk pregeneration, and just as usefully it
/// collapsed the run-to-run spread: median absolute deviation fell from 2.77s
/// to 0.26s, because the tail stalls a shallow pipeline suffers simply stop
/// happening.
/// How many windows may be generating at once.
///
/// This is the pregeneration pipeline's depth. Each window is
/// `window_size * window_size` target chunks, and a new one is admitted only as
/// an active one finishes, so this bounds how much world is resident at once.
///
/// It was briefly scaled to one window per generation thread, on a measurement
/// that turned out to be invalid: the benchmark generated a 301x301 area, and
/// 127 windows of 32x32 is 130,048 target chunks, so the entire area fit inside
/// the pipeline and *nothing ever unloaded*. That removed all unload and save
/// work from the measurement rather than making it faster. Re-measured on a
/// 601x601 area, where unloading is forced, depth makes no throughput
/// difference worth having -- 16 windows 5,572 chunks/s, 32 -> 5,372, 127 ->
/// 5,698, inside run-to-run spread -- while the peak unload backlog scales with
/// it (30k, 47k, 133k chunks) and with it the worst scheduling epoch (121ms,
/// 220ms, 721ms) and the resident set.
///
/// So depth is a memory and latency knob, not a throughput one. Keep it low.
const DEFAULT_PREGEN_ACTIVE_WINDOWS: usize = 16;
const PREGEN_UNLOAD_BACKPRESSURE_ENV: &str = "PREGEN_UNLOAD_BACKPRESSURE";
/// Unload backlog at which window activation pauses.
///
/// Has to clear `window_size^2 * active_windows`, or the budget check rejects
/// the default configuration outright. At the default 32-chunk window and depth
/// 16 that floor is 16,384; the value here leaves room above it so ordinary
/// backlog growth does not trip backpressure and re-introduce the stalls the
/// deeper pipeline was meant to remove.
const DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH: usize = 65536;
/// Backlog watermarks that pause window activation while unloads drain.
///
/// Deeper pipelines retain more finished-window halo, so the high watermark and
/// the pipeline depth have to move together: raising depth alone just trades
/// generation stalls for backpressure stalls.
/// Unload backlog watermark that leaves the configured pipeline room to run.
///
/// The budget check rejects a depth whose in-flight chunks exceed the high
/// watermark, so a fixed watermark silently caps how deep the pipeline may go.
/// Now that depth tracks the generation pool, the watermark has to track depth,
/// with headroom above the floor so ordinary backlog growth does not trip
/// backpressure and reintroduce the stalls a deep pipeline removes.
fn default_pregen_unload_backpressure_high(active_windows: usize) -> usize {
    let window_size = DEFAULT_PREGEN_WINDOW_SIZE as usize;
    let in_flight_floor = window_size
        .saturating_mul(window_size)
        .saturating_mul(active_windows);
    in_flight_floor
        .saturating_mul(2)
        .max(DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UnloadBackpressure {
    high: usize,
    low: usize,
}

impl UnloadBackpressure {
    const fn from_high(high: usize) -> Self {
        Self {
            high,
            low: high / 2,
        }
    }
}
const FULL_DEPENDENCY_RADIUS: i32 = GENERATION_PYRAMID
    .get_step_to(ChunkStatus::Full)
    .accumulated_dependencies
    .get_radius_of(ChunkStatus::Empty) as i32;

/// Running cost of the scheduling epochs driven by a pregeneration.
///
/// Both halves of an epoch are single-threaded, and every chunk has to pass
/// through them to be created and to retire, so their cost bounds throughput no
/// matter how many generation threads are waiting. Tracking the maximum as well
/// as the total matters: one long epoch stalls task creation outright, and the
/// generation pool drains behind it.
#[derive(Default)]
struct EpochCost {
    epochs: u64,
    ticket_updates: Duration,
    schedule_generation: Duration,
    run_generation: Duration,
    process_unloads: Duration,
    readiness_reconcile: Duration,
    lifecycle_commit: Duration,
    worst_epoch: Duration,
    scheduled: usize,
}

impl EpochCost {
    fn record(&mut self, timings: &ChunkMapSchedulingTimings) {
        let total = timings.ticket_updates
            + timings.schedule_generation
            + timings.run_generation
            + timings.process_unloads
            + timings.readiness_reconcile
            + timings.lifecycle_commit;
        if total.is_zero() {
            return;
        }
        self.epochs += 1;
        self.ticket_updates += timings.ticket_updates;
        self.schedule_generation += timings.schedule_generation;
        self.run_generation += timings.run_generation;
        self.process_unloads += timings.process_unloads;
        self.readiness_reconcile += timings.readiness_reconcile;
        self.lifecycle_commit += timings.lifecycle_commit;
        self.scheduled += timings.scheduled_count;
        self.worst_epoch = self.worst_epoch.max(total);
    }

    fn log(&self, elapsed: Duration) {
        let pct = |part: Duration| part.as_secs_f64() / elapsed.as_secs_f64() * 100.0;
        log::info!(
            "Scheduling epochs: {} epochs, {} chunks scheduled, worst epoch {:.1}ms | \
             share of wall clock: tickets {:.1}%, schedule {:.1}%, refill {:.1}%, unloads {:.1}%, \
             readiness {:.1}%, lifecycle {:.1}%",
            self.epochs,
            self.scheduled,
            self.worst_epoch.as_secs_f64() * 1000.0,
            pct(self.ticket_updates),
            pct(self.schedule_generation),
            pct(self.run_generation),
            pct(self.process_unloads),
            pct(self.readiness_reconcile),
            pct(self.lifecycle_commit),
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct PregenWindow {
    min_x: i32,
    max_x: i32,
    min_z: i32,
    max_z: i32,
}

impl PregenWindow {
    fn positions(self) -> Vec<ChunkPos> {
        let mut positions = Vec::with_capacity(self.chunk_count());
        for z in self.min_z..=self.max_z {
            for x in self.min_x..=self.max_x {
                positions.push(ChunkPos::new(x, z));
            }
        }
        positions
    }

    const fn chunk_count(self) -> usize {
        (self.width() * self.height()) as usize
    }

    const fn width(self) -> i32 {
        self.max_x - self.min_x + 1
    }

    const fn height(self) -> i32 {
        self.max_z - self.min_z + 1
    }

    const fn protected_rect(self) -> PregenRect {
        PregenRect {
            min_x: self.min_x - FULL_DEPENDENCY_RADIUS,
            max_x: self.max_x + FULL_DEPENDENCY_RADIUS,
            min_z: self.min_z - FULL_DEPENDENCY_RADIUS,
            max_z: self.max_z + FULL_DEPENDENCY_RADIUS,
        }
    }
}

#[derive(Clone, Copy)]
struct PregenRect {
    min_x: i32,
    max_x: i32,
    min_z: i32,
    max_z: i32,
}

impl PregenRect {
    const fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }
}

struct ActivePregenWindow {
    window: PregenWindow,
    handle: ChunkRequestHandle,
    ready_chunks: usize,
    ready: bool,
    counted: bool,
}

impl ActivePregenWindow {
    fn new(world: &Arc<World>, window: PregenWindow) -> Self {
        let handle = world.chunk_map.request_chunks(ChunkRequest {
            status: ChunkStatus::Full,
            positions: window.positions(),
            ticket_kind: ChunkTicketKind::Pregen,
        });
        Self {
            window,
            handle,
            ready_chunks: 0,
            ready: false,
            counted: false,
        }
    }

    fn poll(&mut self, world: &Arc<World>) {
        match self.handle.poll() {
            ChunkRequestState::Ready => {
                self.ready_chunks = self.window.chunk_count();
                self.ready = true;
            }
            ChunkRequestState::Pending { ready, .. } => {
                self.ready_chunks = ready;
            }
            ChunkRequestState::Cancelled => {
                self.handle = world.chunk_map.request_chunks(ChunkRequest {
                    status: ChunkStatus::Full,
                    positions: self.window.positions(),
                    ticket_kind: ChunkTicketKind::Pregen,
                });
                self.ready_chunks = 0;
                self.ready = false;
                self.counted = false;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PregenSize {
    side_length: i32,
    radius: i32,
}

impl PregenSize {
    fn from_side_length(side_length: i32) -> Result<Option<Self>, String> {
        if side_length == 0 {
            return Ok(None);
        }
        if side_length < 0 {
            return Err(format!(
                "{PREGEN_SIZE_ENV} must be 0 or a positive odd integer"
            ));
        }
        if side_length % 2 == 0 {
            return Err(format!(
                "{PREGEN_SIZE_ENV} must be odd so the area has a single center chunk"
            ));
        }

        Ok(Some(Self {
            side_length,
            radius: side_length / 2,
        }))
    }
}

impl Server {
    /// Generates the startup spawn area for the server default world.
    ///
    /// Set `PREGEN_SIZE` to an odd chunk side length, or `0` to skip custom pregen.
    /// Set `PREGEN_WINDOW_SIZE` to override the default 32-chunk window side length.
    /// The configured size must fit the active-window unload-backpressure budget.
    pub async fn prepare_spawn_area(&self) -> bool {
        let overworld = self.overworld();
        let pregen_size = match get_pregen_size() {
            Ok(Some(size)) => size,
            Ok(None) => {
                log::info!("Skipping custom startup spawn-area pregeneration");
                return true;
            }
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let active_window_limit = match get_pregen_active_windows() {
            Ok(active_windows) => active_windows,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let backpressure = match get_pregen_unload_backpressure(active_window_limit) {
            Ok(backpressure) => backpressure,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let window_size = match get_pregen_window_size(active_window_limit, backpressure) {
            Ok(window_size) => window_size,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let center_chunk = if pregen_size.side_length > VANILLA_PLAYER_SPAWN_SIZE_CHUNKS {
            ChunkPos::new(0, 0)
        } else {
            let spawn_pos = overworld.level_data.read().data().spawn_pos();
            ChunkPos::new(
                SectionPos::block_to_section_coord(spawn_pos.0.x),
                SectionPos::block_to_section_coord(spawn_pos.0.z),
            )
        };

        pregen_overworld(
            overworld,
            center_chunk,
            pregen_size,
            window_size,
            active_window_limit,
            backpressure,
            &self.cancel_token,
        )
        .await
    }
}

fn get_pregen_size() -> Result<Option<PregenSize>, String> {
    let side_length = match env::var(PREGEN_SIZE_ENV) {
        Ok(value) => value
            .parse::<i32>()
            .map_err(|e| format!("{PREGEN_SIZE_ENV} must be 0 or a positive odd integer: {e}"))?,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(format!("{PREGEN_SIZE_ENV} must be valid unicode"));
        }
    };

    PregenSize::from_side_length(side_length)
}

fn get_pregen_unload_backpressure(active_windows: usize) -> Result<UnloadBackpressure, String> {
    let high = match env::var(PREGEN_UNLOAD_BACKPRESSURE_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            format!("{PREGEN_UNLOAD_BACKPRESSURE_ENV} must be a positive integer: {error}")
        })?,
        Err(env::VarError::NotPresent) => default_pregen_unload_backpressure_high(active_windows),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(format!(
                "{PREGEN_UNLOAD_BACKPRESSURE_ENV} must be valid unicode"
            ));
        }
    };
    if high < 2 {
        return Err(format!(
            "{PREGEN_UNLOAD_BACKPRESSURE_ENV} must be at least 2"
        ));
    }

    Ok(UnloadBackpressure::from_high(high))
}

fn get_pregen_active_windows() -> Result<usize, String> {
    match env::var(PREGEN_ACTIVE_WINDOWS_ENV) {
        Ok(value) => parse_pregen_active_windows(&value),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PREGEN_ACTIVE_WINDOWS),
        Err(env::VarError::NotUnicode(_)) => {
            Err(format!("{PREGEN_ACTIVE_WINDOWS_ENV} must be valid unicode"))
        }
    }
}

fn parse_pregen_active_windows(value: &str) -> Result<usize, String> {
    let active_windows = value.parse::<usize>().map_err(|error| {
        format!("{PREGEN_ACTIVE_WINDOWS_ENV} must be a positive integer: {error}")
    })?;
    if active_windows == 0 {
        return Err(format!(
            "{PREGEN_ACTIVE_WINDOWS_ENV} must be a positive integer"
        ));
    }

    Ok(active_windows)
}

fn get_pregen_window_size(
    active_windows: usize,
    backpressure: UnloadBackpressure,
) -> Result<i32, String> {
    match env::var(PREGEN_WINDOW_SIZE_ENV) {
        Ok(value) => parse_pregen_window_size(&value, active_windows, backpressure),
        Err(env::VarError::NotPresent) => {
            check_pregen_window_budget(DEFAULT_PREGEN_WINDOW_SIZE, active_windows, backpressure)
        }
        Err(env::VarError::NotUnicode(_)) => {
            Err(format!("{PREGEN_WINDOW_SIZE_ENV} must be valid unicode"))
        }
    }
}

fn parse_pregen_window_size(
    value: &str,
    active_windows: usize,
    backpressure: UnloadBackpressure,
) -> Result<i32, String> {
    let window_size = value
        .parse::<i32>()
        .map_err(|error| format!("{PREGEN_WINDOW_SIZE_ENV} must be a positive integer: {error}"))?;
    if window_size <= 0 {
        return Err(format!(
            "{PREGEN_WINDOW_SIZE_ENV} must be a positive integer"
        ));
    }

    check_pregen_window_budget(window_size, active_windows, backpressure)
}

/// Rejects window/depth pairs whose in-flight chunks exceed the unload budget.
///
/// Active windows keep their tickets until their dependency halo is no longer
/// needed, so window area times pipeline depth bounds how many chunks can be
/// waiting to unload. Exceeding the high watermark would make backpressure
/// engage permanently and stall window activation outright.
fn check_pregen_window_budget(
    window_size: i32,
    active_windows: usize,
    backpressure: UnloadBackpressure,
) -> Result<i32, String> {
    let window_size_as_usize = window_size as usize;
    let Some(active_target_chunks) = window_size_as_usize
        .checked_mul(window_size_as_usize)
        .and_then(|chunk_count| chunk_count.checked_mul(active_windows))
    else {
        return Err(format!(
            "{PREGEN_WINDOW_SIZE_ENV} is too large for the pregeneration window budget"
        ));
    };
    if active_target_chunks > backpressure.high {
        return Err(format!(
            "{PREGEN_WINDOW_SIZE_ENV} of {window_size} must keep {active_windows} active windows \
             within the {}-chunk unload-backpressure budget (raise \
             {PREGEN_UNLOAD_BACKPRESSURE_ENV} to allow a deeper pipeline)",
            backpressure.high
        ));
    }

    Ok(window_size)
}

async fn pregen_overworld(
    world: &Arc<World>,
    center_chunk: ChunkPos,
    pregen_size: PregenSize,
    window_size: i32,
    active_window_limit: usize,
    backpressure: UnloadBackpressure,
    cancel_token: &CancellationToken,
) -> bool {
    let total_chunks = total_chunks(pregen_size.side_length);

    log::info!(
        "Preparing spawn area: {} chunks ({}x{}) around chunk ({}, {})",
        total_chunks,
        pregen_size.side_length,
        pregen_size.side_length,
        center_chunk.0.x,
        center_chunk.0.y,
    );

    #[cfg(feature = "slow_chunk_gen")]
    SLOW_CHUNK_GEN.store(true, Ordering::Relaxed);

    let elapsed = {
        let start = Instant::now();
        let completed = generate_pregen(
            world,
            center_chunk,
            pregen_size,
            window_size,
            active_window_limit,
            backpressure,
            cancel_token,
        )
        .await;
        (start.elapsed(), completed)
    };

    #[cfg(feature = "slow_chunk_gen")]
    SLOW_CHUNK_GEN.store(false, Ordering::Relaxed);

    let elapsed_secs = elapsed.0.as_secs_f64();
    let chunks_per_second = if elapsed_secs > 0.0 {
        total_chunks as f64 / elapsed_secs
    } else {
        0.0
    };
    if elapsed.1 {
        log::info!(
            "Spawn area prepared: {total_chunks} chunks in {elapsed_secs:.2}s ({chunks_per_second:.1} chunks/s)",
        );
    } else {
        log::info!("Spawn area preparation cancelled after {elapsed_secs:.2}s");
    }
    elapsed.1
}

fn build_pregen_windows(
    center_chunk: ChunkPos,
    radius: i32,
    window_size: i32,
) -> VecDeque<PregenWindow> {
    let min_x = center_chunk.0.x - radius;
    let max_x = center_chunk.0.x + radius;
    let min_z = center_chunk.0.y - radius;
    let max_z = center_chunk.0.y + radius;
    let x_ranges = pregen_window_ranges(min_x, max_x, window_size);
    let z_ranges = pregen_window_ranges(min_z, max_z, window_size);
    let mut windows = VecDeque::new();

    // A two-row serpentine keeps the bounded active set spatially local while
    // reusing one of every two dependency boundaries between window rows.
    for (strip_index, z_pair) in z_ranges.chunks(2).enumerate() {
        let mut push_column = |&(window_min_x, window_max_x): &(i32, i32)| {
            for &(window_min_z, window_max_z) in z_pair {
                windows.push_back(PregenWindow {
                    min_x: window_min_x,
                    max_x: window_max_x,
                    min_z: window_min_z,
                    max_z: window_max_z,
                });
            }
        };

        if strip_index % 2 == 0 {
            for x_range in &x_ranges {
                push_column(x_range);
            }
        } else {
            for x_range in x_ranges.iter().rev() {
                push_column(x_range);
            }
        }
    }

    windows
}

fn pregen_window_ranges(min: i32, max: i32, window_size: i32) -> Vec<(i32, i32)> {
    let mut ranges = Vec::new();
    let mut start = min;
    while start <= max {
        let end = start.saturating_add(window_size - 1).min(max);
        ranges.push((start, end));
        start = end + 1;
    }
    ranges
}

async fn generate_pregen(
    world: &Arc<World>,
    center_chunk: ChunkPos,
    pregen_size: PregenSize,
    window_size: i32,
    active_window_limit: usize,
    backpressure: UnloadBackpressure,
    cancel_token: &CancellationToken,
) -> bool {
    let total_chunks = total_chunks(pregen_size.side_length);
    let mut pending_windows = build_pregen_windows(center_chunk, pregen_size.radius, window_size);
    let mut active_windows = Vec::with_capacity(active_window_limit + 1);
    let mut last_report = Instant::now();
    let mut last_completed = 0usize;
    let mut completed = 0usize;
    let mut unload_backpressure = false;
    let mut peak_unloading_chunks = 0usize;
    let mut epoch_cost = EpochCost::default();
    let start = Instant::now();

    log::info!(
        "Pregeneration windowing: {window_size}x{window_size} target chunks, {active_window_limit} active windows, dependency halo {FULL_DEPENDENCY_RADIUS} chunks",
    );

    fill_active_windows(
        world,
        &mut pending_windows,
        &mut active_windows,
        active_window_limit,
    );

    while completed < total_chunks {
        if cancel_token.is_cancelled() {
            release_all_windows(world, &mut active_windows);
            return false;
        }

        drain_pregen_broadcasts(world);
        epoch_cost.record(&world.chunk_map.advance_scheduling());
        peak_unloading_chunks = peak_unloading_chunks.max(world.chunk_map.unloading_chunks.len());
        update_unload_backpressure(world, &mut unload_backpressure, backpressure);

        for active in &mut active_windows {
            active.poll(world);
        }

        if !unload_backpressure {
            let newly_ready_count = active_windows
                .iter()
                .filter(|active| active.ready && !active.counted)
                .count();
            for _ in 0..newly_ready_count {
                activate_next_window(world, &mut pending_windows, &mut active_windows);
            }
        }

        for active in &mut active_windows {
            if active.ready && !active.counted {
                completed += active.window.chunk_count();
                active.counted = true;
            }
        }
        if !unload_backpressure {
            fill_active_windows(
                world,
                &mut pending_windows,
                &mut active_windows,
                active_window_limit,
            );
        }
        drain_pregen_broadcasts(world);
        epoch_cost.record(&world.chunk_map.advance_scheduling());
        release_unneeded_completed_windows(world, &mut active_windows);
        peak_unloading_chunks = peak_unloading_chunks.max(world.chunk_map.unloading_chunks.len());
        update_unload_backpressure(world, &mut unload_backpressure, backpressure);

        if completed == total_chunks {
            break;
        }

        if pregen_size.side_length > VANILLA_PLAYER_SPAWN_SIZE_CHUNKS
            && last_report.elapsed() >= Duration::from_secs(5)
        {
            last_completed = report_pregen_progress(
                world,
                &active_windows,
                completed,
                last_completed,
                total_chunks,
                last_report,
                start,
            );
            last_report = Instant::now();
        }

        tokio::select! {
            () = cancel_token.cancelled() => {
                release_all_windows(world, &mut active_windows);
                return false;
            }
            () = sleep(Duration::from_millis(10)) => {}
        }
    }

    release_all_windows(world, &mut active_windows);
    peak_unloading_chunks = peak_unloading_chunks.max(world.chunk_map.unloading_chunks.len());
    log::info!("Pregeneration peak unload backlog: {peak_unloading_chunks} chunks");
    epoch_cost.log(start.elapsed());
    true
}

/// Logs one pregeneration progress line and returns the new completed count.
fn report_pregen_progress(
    world: &Arc<World>,
    active_windows: &[ActivePregenWindow],
    completed: usize,
    last_completed: usize,
    total_chunks: usize,
    last_report: Instant,
    start: Instant,
) -> usize {
    let report_elapsed = last_report.elapsed().as_secs_f64();
    let ready_in_active = active_windows
        .iter()
        .filter(|active| !active.counted)
        .map(|active| active.ready_chunks)
        .sum::<usize>();
    let current_completed = (completed + ready_in_active).min(total_chunks);
    let chunks_per_sec = if start.elapsed().as_secs_f64() > 0.0 {
        (current_completed.saturating_sub(last_completed)) as f64 / report_elapsed
    } else {
        0.0
    };
    let percent = (current_completed as f64 / total_chunks as f64) * 100.0;
    let remaining = total_chunks.saturating_sub(current_completed);
    let eta = if chunks_per_sec > 0.0 && remaining > 0 {
        remaining as f64 / chunks_per_sec
    } else {
        0.0
    };
    log::info!(
        "Progress: {current_completed}/{total_chunks} ({percent:.1}%), {chunks_per_sec:.1} chunks/s, ETA: {eta:.0}s, in-flight tasks {}/{}",
        world.chunk_map.running_generation_task_count(),
        world.chunk_map.generation_task_capacity(),
    );
    current_completed
}

fn update_unload_backpressure(
    world: &Arc<World>,
    unload_backpressure: &mut bool,
    watermarks: UnloadBackpressure,
) {
    let unloading_chunks = world.chunk_map.unloading_chunks.len();
    if *unload_backpressure {
        if unloading_chunks <= watermarks.low {
            *unload_backpressure = false;
            log::info!(
                "Pregen unload backpressure released: unloading_chunks={unloading_chunks}, low_watermark={}",
                watermarks.low,
            );
        }
        return;
    }

    if unloading_chunks >= watermarks.high {
        *unload_backpressure = true;
        log::info!(
            "Pregen unload backpressure active: unloading_chunks={unloading_chunks}, high_watermark={}, low_watermark={}",
            watermarks.high,
            watermarks.low,
        );
    }
}

fn drain_pregen_broadcasts(world: &Arc<World>) {
    world.chunk_map.broadcast_changed_chunks();
}

fn total_chunks(side_length: i32) -> usize {
    let side_length = i64::from(side_length);
    (side_length * side_length) as usize
}

fn fill_active_windows(
    world: &Arc<World>,
    pending_windows: &mut VecDeque<PregenWindow>,
    active_windows: &mut Vec<ActivePregenWindow>,
    active_window_limit: usize,
) {
    while active_windows.iter().filter(|active| !active.ready).count() < active_window_limit {
        if !activate_next_window(world, pending_windows, active_windows) {
            break;
        }
    }
}

fn activate_next_window(
    world: &Arc<World>,
    pending_windows: &mut VecDeque<PregenWindow>,
    active_windows: &mut Vec<ActivePregenWindow>,
) -> bool {
    let Some(window) = pending_windows.pop_front() else {
        return false;
    };

    active_windows.push(ActivePregenWindow::new(world, window));
    true
}

fn release_unneeded_completed_windows(
    world: &Arc<World>,
    active_windows: &mut Vec<ActivePregenWindow>,
) {
    let incomplete_windows = active_windows
        .iter()
        .filter(|active| !active.ready)
        .map(|active| active.window)
        .collect::<Vec<_>>();

    active_windows.retain(|active| {
        if !active.ready {
            return true;
        }

        let protected = active.window.protected_rect();

        incomplete_windows
            .iter()
            .any(|window| protected.overlaps(window.protected_rect()))
    });

    world.chunk_map.advance_scheduling();
}

fn release_all_windows(world: &Arc<World>, active_windows: &mut Vec<ActivePregenWindow>) {
    active_windows.clear();
    world.chunk_map.advance_scheduling();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pregen_size_accepts_zero_as_disabled() {
        assert_eq!(PregenSize::from_side_length(0), Ok(None));
    }

    #[test]
    fn pregen_size_accepts_odd_side_lengths() {
        assert_eq!(
            PregenSize::from_side_length(7),
            Ok(Some(PregenSize {
                side_length: 7,
                radius: 3,
            }))
        );
    }

    #[test]
    fn pregen_size_rejects_even_side_lengths() {
        assert!(PregenSize::from_side_length(2).is_err());
    }

    #[test]
    fn pregen_size_rejects_negative_side_lengths() {
        assert!(PregenSize::from_side_length(-1).is_err());
    }

    #[test]
    fn pregen_window_order_keeps_consecutive_dependency_areas_local() {
        let radius = DEFAULT_PREGEN_WINDOW_SIZE * 2;
        let windows = build_pregen_windows(ChunkPos::new(0, 0), radius, DEFAULT_PREGEN_WINDOW_SIZE);

        assert_eq!(windows.len(), 25);
        assert_eq!(
            windows
                .iter()
                .map(|window| window.chunk_count())
                .sum::<usize>(),
            total_chunks(radius * 2 + 1)
        );
        assert!(
            windows
                .iter()
                .zip(windows.iter().skip(1))
                .all(|(current, next)| current.protected_rect().overlaps(next.protected_rect()))
        );
    }

    #[test]
    fn pregen_window_size_requires_a_positive_integer() {
        let depth = 16;
        let budget = UnloadBackpressure::from_high(DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH);
        assert_eq!(parse_pregen_window_size("1", depth, budget), Ok(1));
        assert_eq!(parse_pregen_window_size("64", depth, budget), Ok(64));
        assert!(parse_pregen_window_size("0", depth, budget).is_err());
        assert!(parse_pregen_window_size("-1", depth, budget).is_err());
        assert!(parse_pregen_window_size("wide", depth, budget).is_err());
    }

    #[test]
    fn pregen_window_size_must_fit_unload_backpressure_budget() {
        // Pinned budgets rather than the shipped constant: the rule under test is
        // `window^2 * depth <= high`, and asserting it against whatever the default
        // happens to be makes the test fail whenever the default is retuned.
        let budget = UnloadBackpressure::from_high(8192);
        assert!(parse_pregen_window_size("64", 2, budget).is_ok());
        assert!(parse_pregen_window_size("65", 2, budget).is_err());
        assert!(parse_pregen_window_size(&i32::MAX.to_string(), 2, budget).is_err());
        // Depth trades against window area for the same budget.
        assert!(parse_pregen_window_size("32", 8, budget).is_ok());
        assert!(parse_pregen_window_size("32", 9, budget).is_err());
        // A larger budget admits a deeper pipeline at the same window size.
        let wide = UnloadBackpressure::from_high(32768);
        assert!(parse_pregen_window_size("32", 9, wide).is_ok());
        assert_eq!(wide.low, 16384);

        assert!(parse_pregen_active_windows("0").is_err());
        assert_eq!(parse_pregen_active_windows("4"), Ok(4));
    }

    #[test]
    fn shipped_pregen_defaults_fit_their_own_budget() {
        let budget = UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(16));
        assert!(
            check_pregen_window_budget(
                DEFAULT_PREGEN_WINDOW_SIZE,
                16,
                budget,
            )
            .is_ok(),
            "default window size and pipeline depth must not trip their own backpressure budget"
        );
    }

    #[test]
    fn default_generation_threads_leave_headroom_for_the_other_pools() {
        use crate::server::default_chunk_generation_threads;
        // Small machines keep every thread; there is nothing to spare.
        assert_eq!(default_chunk_generation_threads(1), 1);
        assert_eq!(default_chunk_generation_threads(4), 4);
        // Larger ones give back a single thread. Holding back a quarter measured
        // faster only while task creation was serialized on the scheduling
        // thread; once that was fixed, generation kept scaling to the top of the
        // machine.
        assert_eq!(default_chunk_generation_threads(8), 6);
        assert_eq!(default_chunk_generation_threads(128), 96);
    }

    #[test]
    fn default_pipeline_depth_fits_its_own_backpressure_budget() {
        // Depth is what keeps the generation pool supplied, so it scales with
        // the pool rather than being a flat count.
        assert_eq!(get_pregen_active_windows(), Ok(DEFAULT_PREGEN_ACTIVE_WINDOWS));
        // And a depth that large must not trip its own backpressure budget.
        for threads in [1, 8, 96, 127] {
            let budget =
                UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(threads));
            assert!(
                check_pregen_window_budget(DEFAULT_PREGEN_WINDOW_SIZE, threads, budget).is_ok(),
                "default depth for {threads} threads must fit its own budget"
            );
        }
    }
}
