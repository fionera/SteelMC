//! Startup pregeneration for the server default world.

use std::collections::VecDeque;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use steel_utils::{ChunkPos, SectionPos};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::chunk::chunk_map::ChunkMapSchedulingTimings;
use crate::chunk::chunk_pyramid::GENERATION_PYRAMID;
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
/// How many windows may be generating at once, on any machine.
///
/// This is the pregeneration pipeline's depth. Each window is
/// `window_size * window_size` target chunks, and a new one is admitted only as
/// an active one finishes, so this bounds both how much work the generation
/// threads can see at any moment and how much world is resident at once. Too
/// shallow and the tail of every window -- when a handful of chunks remain and
/// the rest of the pool has nothing to do -- is paid with idle threads.
///
/// It was 2, which starves the pool badly. A scheduler trace of the generation
/// threads showed 90.8% of their context switches were blocking sleeps in
/// rayon's idle path -- they were not contending, they had no work -- while
/// disk waits were 14 events out of 155,697. Raising the depth to 16 measured
/// +18.5% throughput over a 90,601-chunk pregeneration, and just as usefully it
/// collapsed the run-to-run spread: median absolute deviation fell from 2.77s
/// to 0.26s, because the tail stalls a shallow pipeline suffers simply stop
/// happening.
///
/// Past 16 the return is small and the memory is not, so this stays the floor
/// every machine gets and [`DEEP_PREGEN_ACTIVE_WINDOWS`] is taken only when
/// there is measured room for it. A depth that once looked free was measured on
/// a 301x301 area, where 127 windows of 32x32 is 130,048 target chunks: the
/// whole area fit inside the pipeline and *nothing ever unloaded*, which
/// removed the unload and save work from the measurement rather than making it
/// faster. Depth claims have to come from an area large enough to force
/// unloading.
const DEFAULT_PREGEN_ACTIVE_WINDOWS: usize = 16;
/// Pipeline depth taken instead when the machine has memory to spare.
///
/// 601x601 under the drive, sweeping depth: 16 windows 10,188 chunks/s at 8,873
/// MiB peak RSS, 32 -> 10,527 chunks/s (+3.3%) at 13,007 MiB (+47%), 48 ->
/// 10,476 chunks/s at 16,129 MiB (+82%). The memory side of that is solid --
/// resident set is a monotone function of how many chunks are pinned in flight,
/// and it climbs the way the chunk count says it should.
///
/// The throughput side is not yet. It is one run per point with no spread
/// stated, which is below the standard [`DEFAULT_PREGEN_ACTIVE_WINDOWS`] holds
/// itself to two lines above (reps and a median absolute deviation), and there
/// are two reasons to distrust it at this size. 48 lands *below* 32 while
/// costing another 3 GiB, which is what noise looks like rather than a curve
/// flattening; and the depth-16 point here (10,188) and the branch's quoted
/// 601x601 figure (~10,290), which [`pregen_area_for_benchmark`] produces at
/// this same floor depth, differ by 1% -- so the +3.3% is about three times a
/// drift this workload shows between runs that should be identical to each
/// other. An earlier sweep of the same 601x601 area under the old
/// task model measured 16 -> 5,572, 32 -> 5,372, 127 -> 5,698 and concluded the
/// ordering was inside run-to-run spread; the drive changed the absolute
/// numbers, not the evidence needed to call a 3% difference real.
///
/// So this is provisional: it is worth taking only where the memory is
/// genuinely spare, which is why [`choose_pregen_active_windows`] gates it on a
/// 4x margin and never on a machine that would have to give something up for
/// it. Before the gate is widened -- or this becomes the floor -- the sweep
/// wants repetitions and a spread, and 32 has to clear it.
const DEEP_PREGEN_ACTIVE_WINDOWS: usize = 32;
/// Peak RSS attributable to one extra in-flight target chunk.
///
/// From the same sweep: 13,007 - 8,873 = 4,134 MiB bought 16 more windows of
/// 32x32, i.e. 16,384 more target chunks in flight, which is 258 KiB each. 256
/// KiB is that rounded down to a power of two, and it is what makes the depth
/// choice extrapolate to a window size the sweep never ran -- in-flight chunks
/// are `depth * window_size^2`, so the projection scales with window area.
///
/// This projects the *difference* between the two depths, not the whole
/// resident set: the depth-16 baseline is paid whatever we decide here.
const PREGEN_IN_FLIGHT_CHUNK_BYTES: u64 = 256 * 1024;
/// Share of available memory the deeper pipeline is allowed to project into.
///
/// The deep pipeline is taken only when its projected extra working set is at
/// most a quarter of what [`available_memory_bytes`] reports, so the ~4 GiB a
/// 32-chunk window costs needs 16 GiB free. Depth 32 is worth a provisional
/// +3.3%, which does not justify crowding a co-tenant, and the 256 KiB/chunk
/// figure is a point estimate from one workload -- the 4x margin is what covers
/// being wrong about it. It does not cover being wrong about *which* memory:
/// that is why the reading is capped by the process's cgroup limit rather than
/// taken from the host.
const PREGEN_DEEP_PIPELINE_MEMORY_DIVISOR: u64 = 4;
const PREGEN_UNLOAD_BACKPRESSURE_ENV: &str = "PREGEN_UNLOAD_BACKPRESSURE";
/// Unload backlog at which window activation pauses.
///
/// Has to clear `window_size^2 * active_windows`, or the budget check rejects
/// the default configuration outright. At the default 32-chunk window and depth
/// 16 that floor is 16,384; the value here leaves room above it so ordinary
/// backlog growth does not trip backpressure and re-introduce the stalls the
/// deeper pipeline was meant to remove.
const DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH: usize = 65536;
/// Unload backlog watermark that leaves the configured pipeline room to run.
///
/// Deeper pipelines retain more finished-window halo, and
/// [`check_pregen_window_budget`] rejects a depth whose in-flight chunks exceed
/// the high watermark, so a watermark that could not move with depth would
/// silently cap how deep the pipeline may go.
///
/// It only actually moves above depth 32, and that is deliberate rather than an
/// oversight: at the 32-chunk window both shipped depths land on the
/// [`DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH`] floor -- 16 windows want 32,768
/// and 32 want 65,536, and the floor is 65,536 -- so the two ship with the same
/// 65,536-chunk watermark and the deep pipeline runs at half the floor
/// pipeline's slack (32,768 chunks of backlog above its in-flight count, not
/// 49,152). Backpressure is not what bounds it there: the deepest measured
/// backlog on a 601x601 pass was 47k chunks under the old task model at this
/// depth, below the watermark, and tripping it would pause activation until the
/// backlog drained to `low` -- half the watermark, which at depth 32 is exactly
/// one pipeline's worth of chunks. Raising the floor to restore the slack would
/// change the configuration every RSS figure in this file was measured at, so
/// it waits for a measurement of backlog against depth under the drive.
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

/// The pipeline depth a pregeneration will run at, and what decided it.
///
/// The basis is carried rather than logged where it is computed because the
/// decision is made before the window budget is known to hold, and an operator
/// chasing a memory problem needs the reason next to the number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PregenDepth {
    active_windows: usize,
    basis: PregenDepthBasis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PregenDepthBasis {
    /// `PREGEN_ACTIVE_WINDOWS` was set; nothing else was consulted.
    Explicit,
    /// Available memory covers the deeper pipeline's projected extra chunks.
    MemoryHeadroom {
        available_bytes: u64,
        projected_extra_bytes: u64,
    },
    /// It does not, so the floor depth stands.
    MemoryTight {
        available_bytes: u64,
        projected_extra_bytes: u64,
    },
    /// No way to read available memory here, so assume there is none to spare.
    MemoryUnknown,
    /// The deeper pipeline would not fit the unload-backpressure watermark.
    BudgetTooTight { watermark: usize },
}

/// Picks the pipeline depth from what the machine can afford.
///
/// Depth is bought with resident memory: every extra in-flight window is
/// `window_size^2` more chunks that cannot unload yet (see
/// [`PREGEN_IN_FLIGHT_CHUNK_BYTES`] for the measurement). A server that has to
/// coexist with other work should not spend +47% peak RSS for +3.3% of a
/// one-time startup pass, but a pregeneration on a machine with 16 GiB spare
/// should. So this projects the extra working set and takes the deeper pipeline
/// only when it is a quarter of `available_bytes` or less -- which is what the
/// process may take, not what the host has, or a memory-capped container would
/// deepen off its node's free memory and be killed for it.
///
/// `backpressure_high` is the operator's `PREGEN_UNLOAD_BACKPRESSURE` if they
/// set one; without it the watermark is the one
/// [`default_pregen_unload_backpressure_high`] derives for the deeper depth,
/// which at the shipped window size is the floor either way. It is an input
/// rather than something derived afterwards because a depth the
/// watermark cannot cover is rejected outright by
/// [`check_pregen_window_budget`], and the automatic choice must not be able to
/// turn a configuration that started yesterday into a startup failure today.
fn choose_pregen_active_windows(
    available_bytes: Option<u64>,
    window_size: i32,
    backpressure_high: Option<usize>,
) -> PregenDepth {
    // A `const` and not a runtime subtraction: retuning the deep depth below the
    // floor stops compiling here instead of underflowing into a projection so
    // large that nothing would ever deepen again.
    const EXTRA_WINDOWS: u64 = (DEEP_PREGEN_ACTIVE_WINDOWS - DEFAULT_PREGEN_ACTIVE_WINDOWS) as u64;

    let floor = |basis| PregenDepth {
        active_windows: DEFAULT_PREGEN_ACTIVE_WINDOWS,
        basis,
    };

    let watermark = backpressure_high
        .unwrap_or_else(|| default_pregen_unload_backpressure_high(DEEP_PREGEN_ACTIVE_WINDOWS));
    if check_pregen_window_budget(
        window_size,
        DEEP_PREGEN_ACTIVE_WINDOWS,
        UnloadBackpressure::from_high(watermark),
    )
    .is_err()
    {
        return floor(PregenDepthBasis::BudgetTooTight { watermark });
    }

    let Some(available_bytes) = available_bytes else {
        return floor(PregenDepthBasis::MemoryUnknown);
    };
    // A window size that cannot be a chunk count projects an unaffordable
    // working set rather than a free one, so nonsense never reads as headroom.
    let window_area = u64::try_from(window_size).map_or(u64::MAX, |size| size.saturating_mul(size));
    let projected_extra_bytes = window_area
        .saturating_mul(EXTRA_WINDOWS)
        .saturating_mul(PREGEN_IN_FLIGHT_CHUNK_BYTES);

    if projected_extra_bytes.saturating_mul(PREGEN_DEEP_PIPELINE_MEMORY_DIVISOR) <= available_bytes
    {
        PregenDepth {
            active_windows: DEEP_PREGEN_ACTIVE_WINDOWS,
            basis: PregenDepthBasis::MemoryHeadroom {
                available_bytes,
                projected_extra_bytes,
            },
        }
    } else {
        floor(PregenDepthBasis::MemoryTight {
            available_bytes,
            projected_extra_bytes,
        })
    }
}

/// Memory that can be handed out without pushing the machine into reclaim.
///
/// `MemAvailable` rather than `MemFree`: page cache is reclaimable, and a
/// machine that has been serving for a while has almost no free memory and
/// plenty available. No crate is pulled in for this -- it is one line of
/// `/proc/meminfo` -- and everywhere else returns `None`, which reads as "no
/// headroom" and keeps the conservative depth. That is the right way to be
/// wrong: the cost is 3.3% of one startup pass.
///
/// `/proc/meminfo` alone is the wrong number in exactly the deployment the
/// margin exists to protect. Inside a memory-capped container it describes the
/// node, not the cap: on this development box a `docker run -m 4g` reads
/// `MemAvailable` of ~1.1 TiB, clears the 16 GiB the deep pipeline wants
/// several hundred times over, and then gets OOM-killed at the cap. So the
/// cgroup's own headroom caps it (see [`cgroup_headroom`]).
fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::fs::read_to_string;

        let host_available = read_to_string("/proc/meminfo")
            .ok()
            .as_deref()
            .and_then(parse_mem_available);
        available_within_cgroup(host_available, cgroup_headroom())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// What a cgroup memory limit leaves this process, if one applies.
#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CgroupHeadroom {
    /// No memory limit applies at any level, so the host figure stands alone.
    Unlimited,
    /// A limit applies and leaves this much of itself unused.
    Limited(u64),
    /// A limit applies but its numbers could not be read, so how much it leaves
    /// is unknown.
    Unreadable,
}

/// Narrows the host's `MemAvailable` to what this process may actually take.
///
/// A cgroup limit is the number the OOM killer measures against, so it caps the
/// host figure rather than replacing it: a 4 GiB container on a 1 TiB host has
/// 4 GiB, and a 1 TiB container on a host with 4 GiB left has 4 GiB.
///
/// A limit whose numbers cannot be read yields `None` -- "no headroom" -- and
/// not the host figure. Falling back to the host there is precisely the bug
/// this exists to remove: the fallback is only safe when there is nothing
/// capping the process, and here something demonstrably is.
#[cfg(any(target_os = "linux", test))]
fn available_within_cgroup(host_available: Option<u64>, cgroup: CgroupHeadroom) -> Option<u64> {
    match cgroup {
        CgroupHeadroom::Unlimited => host_available,
        CgroupHeadroom::Limited(headroom) => {
            Some(host_available.map_or(headroom, |host| host.min(headroom)))
        }
        CgroupHeadroom::Unreadable => None,
    }
}

/// A memory limit as one cgroup file states it.
#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CgroupLimit {
    /// cgroup v2 `max`, or v1's saturated sentinel: this level caps nothing.
    Unlimited,
    Bytes(u64),
    /// The file exists and says something this cannot read.
    Unparseable,
}

/// cgroup v1 writes an unset limit as a page-aligned saturation of `i64`
/// (`0x7ffffffffffff000` on 4 KiB pages), and kernels have varied the low bits,
/// so any limit above this threshold is "unset" rather than a real cap. It is
/// 1 EiB: a real limit anywhere near it would be a limit on nothing.
#[cfg(any(target_os = "linux", test))]
const CGROUP_V1_UNLIMITED_FLOOR: u64 = 1 << 60;

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_limit(value: &str) -> CgroupLimit {
    let value = value.trim();
    // One function for both generations: a v2 `memory.max` never holds the v1
    // sentinel and a v1 `memory.limit_in_bytes` never holds `max`, so accepting
    // both spellings cannot misread either file.
    if value == "max" {
        return CgroupLimit::Unlimited;
    }
    match value.parse::<u64>() {
        Ok(bytes) if bytes >= CGROUP_V1_UNLIMITED_FLOOR => CgroupLimit::Unlimited,
        Ok(bytes) => CgroupLimit::Bytes(bytes),
        Err(_) => CgroupLimit::Unparseable,
    }
}

/// Headroom one cgroup level leaves, from its limit and current charge.
///
/// The charge includes page cache, which is reclaimable, so this understates
/// headroom on a container that has been reading from disk. That is the
/// direction to be wrong in: overstating it is what gets the process killed,
/// and understating it costs the 3.3% the deeper pipeline is worth.
#[cfg(any(target_os = "linux", test))]
const fn cgroup_level_headroom(limit: CgroupLimit, current_bytes: Option<u64>) -> CgroupHeadroom {
    match (limit, current_bytes) {
        (CgroupLimit::Unlimited, _) => CgroupHeadroom::Unlimited,
        (CgroupLimit::Bytes(limit), Some(current)) => {
            CgroupHeadroom::Limited(limit.saturating_sub(current))
        }
        // A limit that is there but whose numbers are not readable -- either
        // file -- is the case that must not silently read as the host's figure.
        (CgroupLimit::Unparseable, _) | (CgroupLimit::Bytes(_), None) => CgroupHeadroom::Unreadable,
    }
}

/// Combines two levels of the cgroup hierarchy into the one that binds.
///
/// Limits nest: a pod's limit and its parent slice's limit both apply, and the
/// process is killed by whichever it hits first, so the tightest wins.
#[cfg(any(target_os = "linux", test))]
fn narrower_headroom(left: CgroupHeadroom, right: CgroupHeadroom) -> CgroupHeadroom {
    match (left, right) {
        // Unreadable dominates: a level whose headroom is unknown could be the
        // binding one, so the answer cannot be "the other level's number".
        (CgroupHeadroom::Unreadable, _) | (_, CgroupHeadroom::Unreadable) => {
            CgroupHeadroom::Unreadable
        }
        (CgroupHeadroom::Limited(left), CgroupHeadroom::Limited(right)) => {
            CgroupHeadroom::Limited(left.min(right))
        }
        (CgroupHeadroom::Limited(bytes), CgroupHeadroom::Unlimited)
        | (CgroupHeadroom::Unlimited, CgroupHeadroom::Limited(bytes)) => {
            CgroupHeadroom::Limited(bytes)
        }
        (CgroupHeadroom::Unlimited, CgroupHeadroom::Unlimited) => CgroupHeadroom::Unlimited,
    }
}

/// The controller-relative cgroup path this process is in, v2 first.
///
/// `/proc/self/cgroup` is `hierarchy:controllers:path` per line; the unified
/// (v2) hierarchy is the line with an empty controller list and id 0. A v1
/// memory limit lives on the line whose controller list contains `memory`,
/// which may be co-mounted with others (`cpu,memory`), hence the split.
#[cfg(any(target_os = "linux", test))]
fn cgroup_path<'a>(proc_self_cgroup: &'a str, controller: Option<&str>) -> Option<&'a str> {
    proc_self_cgroup.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let _hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let matches = match controller {
            None => controllers.is_empty(),
            Some(wanted) => controllers.split(',').any(|entry| entry == wanted),
        };
        matches.then_some(path)
    })
}

/// How much of its memory limit the cgroup this process runs in has left.
///
/// Both hierarchies are consulted and the tighter answer wins, because on a
/// systemd hybrid host the unified hierarchy exists while the memory controller
/// is still on v1: reading only the one the process has a unified path for
/// would miss the limit on exactly the machines that have one.
///
/// Absent cgroup files read as [`CgroupHeadroom::Unlimited`]: a kernel with no
/// cgroup filesystem, a hierarchy that is not mounted, or a level without the
/// memory controller enabled is not evidence of a limit. Files that exist and
/// cannot be read are, and yield [`CgroupHeadroom::Unreadable`].
#[cfg(target_os = "linux")]
fn cgroup_headroom() -> CgroupHeadroom {
    use std::fs::read_to_string;

    let Ok(proc_self_cgroup) = read_to_string("/proc/self/cgroup") else {
        return CgroupHeadroom::Unlimited;
    };

    let unified = cgroup_path(&proc_self_cgroup, None).map_or(CgroupHeadroom::Unlimited, |path| {
        cgroup_hierarchy_headroom("/sys/fs/cgroup", path, "memory.max", "memory.current")
    });
    let legacy =
        cgroup_path(&proc_self_cgroup, Some("memory")).map_or(CgroupHeadroom::Unlimited, |path| {
            cgroup_hierarchy_headroom(
                "/sys/fs/cgroup/memory",
                path,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            )
        });

    narrower_headroom(unified, legacy)
}

/// Headroom left by one cgroup hierarchy, walking the whole ancestry.
///
/// From this process's own cgroup up to the mount root, because a limit set on
/// an ancestor binds just as hard as one set on the leaf -- Kubernetes puts the
/// pod limit on the parent of the container's own cgroup, so reading only the
/// leaf misses it entirely.
#[cfg(target_os = "linux")]
fn cgroup_hierarchy_headroom(
    root: &str,
    relative: &str,
    limit_file: &str,
    current_file: &str,
) -> CgroupHeadroom {
    use std::fs::read_to_string;
    use std::path::{Path, PathBuf};

    let root = Path::new(root);
    let mut level = PathBuf::from(root);
    // A namespaced container sees its own cgroup as `/`, so the walk is often a
    // single level; on the host it is the full slice/scope chain.
    level.push(relative.trim_start_matches('/'));

    let mut headroom = CgroupHeadroom::Unlimited;
    loop {
        let limit = match read_to_string(level.join(limit_file)) {
            Ok(value) => parse_cgroup_limit(&value),
            // No such file: this hierarchy is not mounted, or the controller is
            // not enabled at this level. Neither caps anything.
            Err(_) => CgroupLimit::Unlimited,
        };
        let current = read_to_string(level.join(current_file))
            .ok()
            .and_then(|value| parse_cgroup_current(&value));
        headroom = narrower_headroom(headroom, cgroup_level_headroom(limit, current));

        if level == root || !level.pop() {
            return headroom;
        }
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_current(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

#[cfg(any(target_os = "linux", test))]
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let field = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    // The kernel writes this field in kB (KiB, despite the spelling) and has
    // done since 3.14; a unit that is not that is a kernel this cannot read.
    let (value, unit) = field.trim().split_once(char::is_whitespace)?;
    if !unit.trim().eq_ignore_ascii_case("kB") {
        return None;
    }

    value.parse::<u64>().ok()?.checked_mul(1024)
}

/// Describes a depth choice the way an operator debugging memory needs it.
fn describe_pregen_depth(depth: PregenDepth) -> String {
    let mib = |bytes: u64| bytes / (1024 * 1024);
    let windows = depth.active_windows;
    match depth.basis {
        PregenDepthBasis::Explicit => {
            format!("depth {windows} windows, set explicitly by {PREGEN_ACTIVE_WINDOWS_ENV}")
        }
        PregenDepthBasis::MemoryHeadroom {
            available_bytes,
            projected_extra_bytes,
        } => format!(
            "depth {windows} windows: its extra in-flight chunks project {} MiB against {} MiB \
             available to this process (MemAvailable, capped by any cgroup memory limit), clearing \
             the {PREGEN_DEEP_PIPELINE_MEMORY_DIVISOR}x margin required to deepen (set \
             {PREGEN_ACTIVE_WINDOWS_ENV}={DEFAULT_PREGEN_ACTIVE_WINDOWS} to keep the resident set \
             down instead)",
            mib(projected_extra_bytes),
            mib(available_bytes),
        ),
        PregenDepthBasis::MemoryTight {
            available_bytes,
            projected_extra_bytes,
        } => format!(
            "depth {windows} windows: {DEEP_PREGEN_ACTIVE_WINDOWS} would project {} MiB of extra \
             in-flight chunks, which wants {} MiB available and this process has {} MiB \
             (MemAvailable, capped by any cgroup memory limit)",
            mib(projected_extra_bytes),
            mib(projected_extra_bytes.saturating_mul(PREGEN_DEEP_PIPELINE_MEMORY_DIVISOR)),
            mib(available_bytes),
        ),
        PregenDepthBasis::MemoryUnknown => format!(
            "depth {windows} windows: available memory cannot be read here -- no readable \
             MemAvailable, or a cgroup memory limit applies whose numbers could not be read -- so \
             the deeper {DEEP_PREGEN_ACTIVE_WINDOWS}-window pipeline is not assumed to fit"
        ),
        PregenDepthBasis::BudgetTooTight { watermark } => format!(
            "depth {windows} windows: {DEEP_PREGEN_ACTIVE_WINDOWS} would not fit the \
             {watermark}-chunk unload-backpressure watermark"
        ),
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
    readiness_demotions: Duration,
    block_entity_unloads: Duration,
    ticking_snapshot_rebuild: Duration,
    worst_epoch: Duration,
    /// Worst single occurrence of each phase, in the same order as the log.
    /// The mean epoch is far below the pool's drain time, so it is the tail that
    /// starves the generation pool and the tail that has to be attributed.
    worst_phases: [Duration; 9],
    scheduled: usize,
    /// Largest single epoch's scheduling batch, which is what the tail measures.
    worst_scheduled_batch: usize,
    ticking_chunks: usize,
}

impl EpochCost {
    fn record(&mut self, timings: &ChunkMapSchedulingTimings) {
        // Every phase, not a subset. Three of these used to be left out, which
        // made both the per-phase shares and `worst_epoch` undercounts -- and
        // hid the snapshot rebuild entirely, which is a full scan of the holder
        // map on every boundary.
        let total = timings.ticket_updates
            + timings.schedule_generation
            + timings.run_generation
            + timings.process_unloads
            + timings.readiness_reconcile
            + timings.lifecycle_commit
            + timings.readiness_demotions
            + timings.block_entity_unloads
            + timings.ticking_snapshot_rebuild;
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
        self.readiness_demotions += timings.readiness_demotions;
        self.block_entity_unloads += timings.block_entity_unloads;
        self.ticking_snapshot_rebuild += timings.ticking_snapshot_rebuild;
        self.scheduled += timings.scheduled_count;
        self.worst_scheduled_batch = self.worst_scheduled_batch.max(timings.scheduled_count);
        self.ticking_chunks = self.ticking_chunks.max(timings.rebuilt_ticking_chunk_count);
        for (worst, phase) in self.worst_phases.iter_mut().zip([
            timings.ticket_updates,
            timings.schedule_generation,
            timings.run_generation,
            timings.process_unloads,
            timings.readiness_reconcile,
            timings.lifecycle_commit,
            timings.readiness_demotions,
            timings.block_entity_unloads,
            timings.ticking_snapshot_rebuild,
        ]) {
            *worst = (*worst).max(phase);
        }
        self.worst_epoch = self.worst_epoch.max(total);
    }

    fn log(&self, elapsed: Duration) {
        let pct = |part: Duration| part.as_secs_f64() / elapsed.as_secs_f64() * 100.0;
        let total = self.ticket_updates
            + self.schedule_generation
            + self.run_generation
            + self.process_unloads
            + self.readiness_reconcile
            + self.lifecycle_commit
            + self.readiness_demotions
            + self.block_entity_unloads
            + self.ticking_snapshot_rebuild;
        log::info!(
            "Scheduling epochs: {} epochs, {} chunks scheduled, peak {} ticking chunks, \
             worst epoch {:.1}ms, all phases {:.1}% of wall clock | \
             tickets {:.1}%, schedule {:.1}%, refill {:.1}%, unloads {:.1}%, \
             readiness {:.1}%, lifecycle {:.1}%, demotions {:.1}%, block-entities {:.1}%, \
             ticking-snapshot {:.1}%",
            self.epochs,
            self.scheduled,
            self.ticking_chunks,
            self.worst_epoch.as_secs_f64() * 1000.0,
            pct(total),
            pct(self.ticket_updates),
            pct(self.schedule_generation),
            pct(self.run_generation),
            pct(self.process_unloads),
            pct(self.readiness_reconcile),
            pct(self.lifecycle_commit),
            pct(self.readiness_demotions),
            pct(self.block_entity_unloads),
            pct(self.ticking_snapshot_rebuild),
        );
        let names = [
            "tickets",
            "schedule",
            "refill",
            "unloads",
            "readiness",
            "lifecycle",
            "demotions",
            "block-entities",
            "ticking-snapshot",
        ];
        let mut worst: Vec<String> = names
            .iter()
            .zip(self.worst_phases)
            .map(|(name, phase)| format!("{name} {:.1}ms", phase.as_secs_f64() * 1000.0))
            .collect();
        worst.sort_by(|left, right| right.len().cmp(&left.len()));
        log::info!(
            "Scheduling epochs: mean epoch {:.2}ms, mean batch {} holders, worst batch {} holders \
             | worst single phase: {}",
            total.as_secs_f64() * 1000.0 / self.epochs.max(1) as f64,
            self.scheduled / self.epochs.max(1) as usize,
            self.worst_scheduled_batch,
            names
                .iter()
                .zip(self.worst_phases)
                .map(|(name, phase)| format!("{name} {:.1}ms", phase.as_secs_f64() * 1000.0))
                .collect::<Vec<_>>()
                .join(", "),
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
    ///
    /// The pipeline depth is chosen from available memory unless
    /// `PREGEN_ACTIVE_WINDOWS` pins it, and the choice with its basis is logged
    /// here -- this runs once per startup, so that log line is the record of why
    /// the run has the resident set it has.
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
        // Window size first: the depth choice projects a working set from the
        // window area, and the watermark override has to be known before the
        // depth so that an operator-pinned watermark constrains the automatic
        // choice instead of being contradicted by it.
        let requested_window_size = match get_pregen_window_size() {
            Ok(window_size) => window_size,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let backpressure_high = match get_pregen_unload_backpressure_override() {
            Ok(high) => high,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let depth = match get_pregen_active_windows(requested_window_size, backpressure_high) {
            Ok(depth) => depth,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        let active_window_limit = depth.active_windows;
        let backpressure = UnloadBackpressure::from_high(
            backpressure_high
                .unwrap_or_else(|| default_pregen_unload_backpressure_high(active_window_limit)),
        );
        let window_size = match check_pregen_window_budget(
            requested_window_size,
            active_window_limit,
            backpressure,
        ) {
            Ok(window_size) => window_size,
            Err(error) => {
                log::error!("{error}");
                return false;
            }
        };
        log::info!("Pregeneration pipeline {}", describe_pregen_depth(depth));
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

/// Runs one pregeneration pass with every parameter supplied explicitly.
///
/// The server path takes its parameters from the `PREGEN_*` environment
/// variables and needs a fully built [`Server`], which binds a listener as soon
/// as it starts. This entry point takes a bare world instead, so a benchmark can
/// drive the real scheduler -- tickets, epochs, generation, unload and save --
/// without a network port to contend for or a shared save directory to collide
/// over. Everything below it is the production path, unchanged.
///
/// `window_size` and `active_window_limit` default to the server's values when
/// `None`. Returns the wall time of the pass, or `None` if it was cancelled.
///
/// # Errors
/// Returns an error if `side_length` is not a positive odd integer, or if the
/// window size and pipeline depth exceed the unload-backpressure budget.
#[cfg(feature = "benchmark-support")]
pub async fn pregen_area_for_benchmark(
    world: &Arc<World>,
    center_chunk: ChunkPos,
    side_length: i32,
    window_size: Option<i32>,
    active_window_limit: Option<usize>,
    cancel_token: &CancellationToken,
) -> Result<Option<Duration>, String> {
    let Some(pregen_size) = PregenSize::from_side_length(side_length)? else {
        return Ok(Some(Duration::ZERO));
    };
    // Deliberately the floor depth and not the server's automatic choice: a
    // benchmark number that moves with how much RAM happened to be free is not
    // comparable across runs, and the harness installs no logger, so the chosen
    // depth would not even be reported. Sweep depth with `--windows`; on a
    // machine with headroom the server now ships `--windows 32`.
    let active_windows = active_window_limit.unwrap_or(DEFAULT_PREGEN_ACTIVE_WINDOWS);
    if active_windows == 0 {
        return Err("active window limit must be a positive integer".to_owned());
    }
    let backpressure =
        UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(active_windows));
    let window_size = check_pregen_window_budget(
        window_size.unwrap_or(DEFAULT_PREGEN_WINDOW_SIZE),
        active_windows,
        backpressure,
    )?;

    let start = Instant::now();
    let completed = generate_pregen(
        world,
        center_chunk,
        pregen_size,
        window_size,
        active_windows,
        backpressure,
        cancel_token,
    )
    .await;

    Ok(completed.then(|| start.elapsed()))
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

/// Reads an operator-pinned unload watermark, `None` if they pinned none.
///
/// Returned unresolved so the depth choice can see whether the watermark is
/// fixed; the default is only known once a depth has been picked.
fn get_pregen_unload_backpressure_override() -> Result<Option<usize>, String> {
    let high = match env::var(PREGEN_UNLOAD_BACKPRESSURE_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            format!("{PREGEN_UNLOAD_BACKPRESSURE_ENV} must be a positive integer: {error}")
        })?,
        Err(env::VarError::NotPresent) => return Ok(None),
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

    Ok(Some(high))
}

fn get_pregen_active_windows(
    window_size: i32,
    backpressure_high: Option<usize>,
) -> Result<PregenDepth, String> {
    match env::var(PREGEN_ACTIVE_WINDOWS_ENV) {
        Ok(value) => Ok(PregenDepth {
            active_windows: parse_pregen_active_windows(&value)?,
            basis: PregenDepthBasis::Explicit,
        }),
        Err(env::VarError::NotPresent) => Ok(choose_pregen_active_windows(
            available_memory_bytes(),
            window_size,
            backpressure_high,
        )),
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

/// Reads the requested window side length, before the budget check.
///
/// Only well-formedness is checked here: whether the window fits depends on the
/// depth, and the depth is now chosen from the window area, so the budget check
/// has to run after both are known.
fn get_pregen_window_size() -> Result<i32, String> {
    match env::var(PREGEN_WINDOW_SIZE_ENV) {
        Ok(value) => parse_pregen_window_size(&value),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PREGEN_WINDOW_SIZE),
        Err(env::VarError::NotUnicode(_)) => {
            Err(format!("{PREGEN_WINDOW_SIZE_ENV} must be valid unicode"))
        }
    }
}

fn parse_pregen_window_size(value: &str) -> Result<i32, String> {
    let window_size = value
        .parse::<i32>()
        .map_err(|error| format!("{PREGEN_WINDOW_SIZE_ENV} must be a positive integer: {error}"))?;
    if window_size <= 0 {
        return Err(format!(
            "{PREGEN_WINDOW_SIZE_ENV} must be a positive integer"
        ));
    }

    Ok(window_size)
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

    /// Parses a window size and then budget-checks it, the way startup does.
    fn checked_window_size(
        value: &str,
        active_windows: usize,
        backpressure: UnloadBackpressure,
    ) -> Result<i32, String> {
        check_pregen_window_budget(
            parse_pregen_window_size(value)?,
            active_windows,
            backpressure,
        )
    }

    #[test]
    fn pregen_window_size_requires_a_positive_integer() {
        assert_eq!(parse_pregen_window_size("1"), Ok(1));
        assert_eq!(parse_pregen_window_size("64"), Ok(64));
        assert!(parse_pregen_window_size("0").is_err());
        assert!(parse_pregen_window_size("-1").is_err());
        assert!(parse_pregen_window_size("wide").is_err());
    }

    #[test]
    fn pregen_window_size_must_fit_unload_backpressure_budget() {
        // Pinned budgets rather than the shipped constant: the rule under test is
        // `window^2 * depth <= high`, and asserting it against whatever the default
        // happens to be makes the test fail whenever the default is retuned.
        let budget = UnloadBackpressure::from_high(8192);
        assert!(checked_window_size("64", 2, budget).is_ok());
        assert!(checked_window_size("65", 2, budget).is_err());
        assert!(checked_window_size(&i32::MAX.to_string(), 2, budget).is_err());
        // Depth trades against window area for the same budget.
        assert!(checked_window_size("32", 8, budget).is_ok());
        assert!(checked_window_size("32", 9, budget).is_err());
        // A larger budget admits a deeper pipeline at the same window size.
        let wide = UnloadBackpressure::from_high(32768);
        assert!(checked_window_size("32", 9, wide).is_ok());
        assert_eq!(wide.low, 16384);

        assert!(parse_pregen_active_windows("0").is_err());
        assert_eq!(parse_pregen_active_windows("4"), Ok(4));
    }

    #[test]
    fn shipped_pregen_defaults_fit_their_own_budget() {
        let budget = UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(16));
        assert!(
            check_pregen_window_budget(DEFAULT_PREGEN_WINDOW_SIZE, 16, budget,).is_ok(),
            "default window size and pipeline depth must not trip their own backpressure budget"
        );
    }

    #[test]
    fn default_generation_threads_leave_headroom_for_the_other_pools() {
        use crate::server::{default_chunk_encoding_threads, default_chunk_generation_threads};
        // Small machines keep every thread; there is nothing to spare.
        assert_eq!(default_chunk_generation_threads(1), 1);
        assert_eq!(default_chunk_generation_threads(4), 4);
        assert_eq!(default_chunk_generation_threads(8), 7);
        // Fifteen sixteenths of a large machine: the measured peak, and a broad
        // one (120 threads 9,484 chunks/s against 112 at 9,225 and 127 at 9,429).
        assert_eq!(default_chunk_generation_threads(128), 120);

        assert_eq!(default_chunk_encoding_threads(128), 12);
        // The pools deliberately overcommit: 120 generation, 12 encoding and 16
        // chunk-runtime workers is 148 threads on a 128-thread machine, and that
        // measured faster than any configuration that fits, because generation
        // threads spend real time blocked on storage and step handoffs rather
        // than running. What has to hold is only that generation leaves the
        // other pools something to run on.
        assert!(default_chunk_generation_threads(128) < 128);
    }

    #[test]
    fn any_pipeline_depth_fits_its_own_backpressure_budget() {
        // The watermark is derived from the depth, so every depth the server can
        // choose has to clear the budget check its own watermark implies --
        // otherwise picking a depth is a way to make startup fail.
        for depth in [
            1,
            8,
            DEFAULT_PREGEN_ACTIVE_WINDOWS,
            DEEP_PREGEN_ACTIVE_WINDOWS,
            96,
            127,
        ] {
            let budget =
                UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(depth));
            assert!(
                check_pregen_window_budget(DEFAULT_PREGEN_WINDOW_SIZE, depth, budget).is_ok(),
                "depth {depth} must fit the watermark it derives"
            );
        }
    }

    #[test]
    fn the_shipped_depths_share_one_floor_clamped_watermark() {
        // The watermark is derived from depth but floored, and at the 32-chunk
        // window both shipped depths land on the floor. Asserted because the
        // doc above it explains the deep pipeline's slack in terms of this
        // clamp: if the floor or the multiplier moves, the two stop sharing a
        // watermark and that explanation is no longer the truth.
        let floor_depth = default_pregen_unload_backpressure_high(DEFAULT_PREGEN_ACTIVE_WINDOWS);
        let deep_depth = default_pregen_unload_backpressure_high(DEEP_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(floor_depth, DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH);
        assert_eq!(deep_depth, DEFAULT_PREGEN_UNLOAD_BACKPRESSURE_HIGH);

        // What that costs, stated so it cannot change unnoticed: the deep
        // pipeline runs at half the floor pipeline's backlog slack, and its low
        // watermark is exactly its own in-flight count -- so if backpressure
        // ever does trip at depth 32, activation resumes only after a full
        // pipeline's worth of chunks has drained.
        let window = DEFAULT_PREGEN_WINDOW_SIZE as usize * DEFAULT_PREGEN_WINDOW_SIZE as usize;
        let floor_in_flight = window * DEFAULT_PREGEN_ACTIVE_WINDOWS;
        let deep_in_flight = window * DEEP_PREGEN_ACTIVE_WINDOWS;
        assert_eq!(floor_depth - floor_in_flight, 49152);
        assert_eq!(deep_depth - deep_in_flight, 32768);
        assert_eq!(
            UnloadBackpressure::from_high(deep_depth).low,
            deep_in_flight
        );

        // Above the shipped depths the derivation does take over, which is what
        // keeps a deeper operator-pinned depth from failing its own budget.
        assert_eq!(default_pregen_unload_backpressure_high(48), 98304);
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Memory a window of `window_size` needs before the deep pipeline is taken.
    fn deep_pipeline_requirement(window_size: u64) -> u64 {
        let extra_chunks = window_size
            * window_size
            * (DEEP_PREGEN_ACTIVE_WINDOWS - DEFAULT_PREGEN_ACTIVE_WINDOWS) as u64;
        extra_chunks * PREGEN_IN_FLIGHT_CHUNK_BYTES * PREGEN_DEEP_PIPELINE_MEMORY_DIVISOR
    }

    #[test]
    fn pipeline_depth_deepens_only_with_memory_to_spare() {
        // The shipped window: 16 extra windows of 32x32 is 16,384 in-flight
        // chunks at 256 KiB each, so 4 GiB projected and 16 GiB required.
        assert_eq!(deep_pipeline_requirement(32), 16 * GIB);

        let deep = choose_pregen_active_windows(Some(64 * GIB), DEFAULT_PREGEN_WINDOW_SIZE, None);
        assert_eq!(deep.active_windows, DEEP_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(
            deep.basis,
            PregenDepthBasis::MemoryHeadroom {
                available_bytes: 64 * GIB,
                projected_extra_bytes: 4 * GIB,
            }
        );

        // Exactly enough qualifies; one byte less does not. The boundary is
        // asserted from both sides because a `<` here would silently move the
        // requirement by a whole factor of the divisor on the next edit.
        assert_eq!(
            choose_pregen_active_windows(Some(16 * GIB), DEFAULT_PREGEN_WINDOW_SIZE, None)
                .active_windows,
            DEEP_PREGEN_ACTIVE_WINDOWS
        );
        let tight =
            choose_pregen_active_windows(Some(16 * GIB - 1), DEFAULT_PREGEN_WINDOW_SIZE, None);
        assert_eq!(tight.active_windows, DEFAULT_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(
            tight.basis,
            PregenDepthBasis::MemoryTight {
                available_bytes: 16 * GIB - 1,
                projected_extra_bytes: 4 * GIB,
            }
        );

        // A machine with 4 GiB to spare runs the pregeneration, just not the
        // deep one. That the 4 GiB *container* case reaches this decision with
        // 4 GiB rather than its host's figure is a property of the reader, and
        // is asserted in `a_container_memory_limit_caps_the_hosts_figure`.
        assert_eq!(
            choose_pregen_active_windows(Some(4 * GIB), DEFAULT_PREGEN_WINDOW_SIZE, None)
                .active_windows,
            DEFAULT_PREGEN_ACTIVE_WINDOWS
        );

        // No reading means no evidence of headroom, not permission to assume it.
        let unknown = choose_pregen_active_windows(None, DEFAULT_PREGEN_WINDOW_SIZE, None);
        assert_eq!(unknown.active_windows, DEFAULT_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(unknown.basis, PregenDepthBasis::MemoryUnknown);
    }

    #[test]
    fn pipeline_depth_requirement_scales_with_window_area() {
        // In-flight chunks are `depth * window_size^2`, so halving the window
        // quarters what the deeper pipeline costs and quarters what it demands.
        for (window_size, required) in [(8, GIB), (16, 4 * GIB), (32, 16 * GIB)] {
            assert_eq!(deep_pipeline_requirement(window_size), required);
            let window_size = i32::try_from(window_size).expect("window size fits an i32");
            assert_eq!(
                choose_pregen_active_windows(Some(required), window_size, None).active_windows,
                DEEP_PREGEN_ACTIVE_WINDOWS,
                "window {window_size} should deepen with {required} bytes available"
            );
            assert_eq!(
                choose_pregen_active_windows(Some(required - 1), window_size, None).active_windows,
                DEFAULT_PREGEN_ACTIVE_WINDOWS,
                "window {window_size} should not deepen just below its requirement"
            );
        }
    }

    #[test]
    fn pipeline_depth_never_deepens_past_the_unload_watermark() {
        // A 48-chunk window is inside the budget at depth 16 (36,864 in-flight
        // against a 65,536 watermark) and outside it at depth 32 (73,728), so
        // deepening would turn a working configuration into a startup failure.
        // Memory is deliberately absurd here: the watermark has to veto first.
        let wide = choose_pregen_active_windows(Some(1024 * GIB), 48, None);
        assert_eq!(wide.active_windows, DEFAULT_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(
            wide.basis,
            PregenDepthBasis::BudgetTooTight { watermark: 65536 }
        );
        let budget = UnloadBackpressure::from_high(default_pregen_unload_backpressure_high(
            wide.active_windows,
        ));
        assert!(check_pregen_window_budget(48, wide.active_windows, budget).is_ok());

        // Same veto when the operator pinned a watermark too low for depth 32.
        let pinned =
            choose_pregen_active_windows(Some(1024 * GIB), DEFAULT_PREGEN_WINDOW_SIZE, Some(20000));
        assert_eq!(pinned.active_windows, DEFAULT_PREGEN_ACTIVE_WINDOWS);
        assert_eq!(
            pinned.basis,
            PregenDepthBasis::BudgetTooTight { watermark: 20000 }
        );
        // 32 windows of 32x32 is 32,768 in-flight chunks, so a watermark that
        // clears that leaves the choice to memory again.
        assert_eq!(
            choose_pregen_active_windows(Some(1024 * GIB), DEFAULT_PREGEN_WINDOW_SIZE, Some(32768))
                .active_windows,
            DEEP_PREGEN_ACTIVE_WINDOWS
        );
    }

    #[test]
    fn depth_log_line_carries_the_number_and_the_evidence() {
        // The whole point of the basis is that an operator staring at a resident
        // set can read why it is that size, so the numbers behind the decision
        // have to reach the log, not just the verdict.
        let deep = describe_pregen_depth(choose_pregen_active_windows(
            Some(64 * GIB),
            DEFAULT_PREGEN_WINDOW_SIZE,
            None,
        ));
        assert!(deep.contains("depth 32 windows"), "{deep}");
        assert!(deep.contains("65536 MiB available"), "{deep}");
        assert!(deep.contains("4096 MiB"), "{deep}");

        let tight = describe_pregen_depth(choose_pregen_active_windows(
            Some(GIB),
            DEFAULT_PREGEN_WINDOW_SIZE,
            None,
        ));
        assert!(tight.contains("depth 16 windows"), "{tight}");
        assert!(tight.contains("wants 16384 MiB"), "{tight}");
        assert!(tight.contains("has 1024 MiB"), "{tight}");

        let explicit = describe_pregen_depth(PregenDepth {
            active_windows: 48,
            basis: PregenDepthBasis::Explicit,
        });
        assert!(explicit.contains("depth 48 windows"), "{explicit}");
        assert!(explicit.contains(PREGEN_ACTIVE_WINDOWS_ENV), "{explicit}");
    }

    #[test]
    fn explicit_pipeline_depth_wins_over_the_automatic_choice() {
        let depth = get_pregen_active_windows(DEFAULT_PREGEN_WINDOW_SIZE, None)
            .expect("the ambient environment should hold a valid depth");
        // Asserted against the environment rather than by setting it: mutating
        // the environment is unsound with other tests running in this process.
        match env::var(PREGEN_ACTIVE_WINDOWS_ENV) {
            Ok(value) => {
                assert_eq!(depth.basis, PregenDepthBasis::Explicit);
                assert_eq!(
                    depth.active_windows,
                    value.parse::<usize>().expect("a valid pinned depth")
                );
            }
            Err(_) => assert_ne!(depth.basis, PregenDepthBasis::Explicit),
        }
    }

    #[test]
    fn available_memory_reads_the_kernel_estimate_not_free_pages() {
        // MemFree first and larger, so a parser that matched the wrong field or
        // took the first number would be caught.
        let meminfo = "MemTotal:       32000000 kB\n\
                       MemFree:         9000000 kB\n\
                       MemAvailable:    1048576 kB\n\
                       Buffers:          100000 kB\n";
        assert_eq!(parse_mem_available(meminfo), Some(GIB));
        assert_eq!(parse_mem_available("MemTotal: 32000000 kB\n"), None);
        // A unit this does not understand is not worth guessing at.
        assert_eq!(parse_mem_available("MemAvailable:  16 GB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable:  many kB\n"), None);
    }

    #[test]
    fn a_container_memory_limit_caps_the_hosts_figure() {
        // The deployment the margin exists for: this box's real MemAvailable
        // (1,154,764,308 kB) inside `docker run -m 4g`. Reading the host figure
        // clears the 16 GiB requirement seventy times over and grows the pass
        // toward the 13,007 MiB the deep pipeline was measured at, inside a 4
        // GiB cap. The cgroup's headroom has to win.
        let host = 1_154_764_308 * 1024;
        let container = available_within_cgroup(
            Some(host),
            cgroup_level_headroom(CgroupLimit::Bytes(4 * GIB), Some(GIB / 2)),
        );
        assert_eq!(container, Some(4 * GIB - GIB / 2));
        assert_eq!(
            choose_pregen_active_windows(container, DEFAULT_PREGEN_WINDOW_SIZE, None)
                .active_windows,
            DEFAULT_PREGEN_ACTIVE_WINDOWS,
            "a 4 GiB container must not take the deep pipeline off its host's free memory"
        );

        // The cap is a cap, not a replacement: a container far larger than what
        // the host has left still cannot have more than the host has left.
        assert_eq!(
            available_within_cgroup(
                Some(2 * GIB),
                cgroup_level_headroom(CgroupLimit::Bytes(512 * GIB), Some(0)),
            ),
            Some(2 * GIB)
        );
        // And with no limit anywhere, the host figure is the answer -- otherwise
        // this would have taken the deep pipeline away from every bare-metal
        // machine that was choosing it correctly.
        assert_eq!(
            available_within_cgroup(Some(64 * GIB), CgroupHeadroom::Unlimited),
            Some(64 * GIB)
        );
        assert_eq!(
            choose_pregen_active_windows(
                available_within_cgroup(Some(64 * GIB), CgroupHeadroom::Unlimited),
                DEFAULT_PREGEN_WINDOW_SIZE,
                None
            )
            .active_windows,
            DEEP_PREGEN_ACTIVE_WINDOWS
        );
    }

    #[test]
    fn a_limit_that_cannot_be_read_is_not_headroom() {
        // A limit exists, so `/proc/meminfo` is known to be the wrong number;
        // falling back to it is the failure mode this whole reading exists to
        // avoid, so an unreadable limit has to erase the host figure entirely.
        assert_eq!(
            cgroup_level_headroom(CgroupLimit::Bytes(4 * GIB), None),
            CgroupHeadroom::Unreadable
        );
        assert_eq!(
            cgroup_level_headroom(CgroupLimit::Unparseable, Some(0)),
            CgroupHeadroom::Unreadable
        );
        assert_eq!(
            available_within_cgroup(Some(1024 * GIB), CgroupHeadroom::Unreadable),
            None
        );
        assert_eq!(
            choose_pregen_active_windows(
                available_within_cgroup(Some(1024 * GIB), CgroupHeadroom::Unreadable),
                DEFAULT_PREGEN_WINDOW_SIZE,
                None
            )
            .basis,
            PregenDepthBasis::MemoryUnknown
        );

        // An unreadable level poisons the whole walk for the same reason: it
        // could be the one that binds.
        assert_eq!(
            narrower_headroom(
                CgroupHeadroom::Limited(64 * GIB),
                CgroupHeadroom::Unreadable
            ),
            CgroupHeadroom::Unreadable
        );
        // A charged limit with no host figure at all still answers.
        assert_eq!(
            available_within_cgroup(None, CgroupHeadroom::Limited(4 * GIB)),
            Some(4 * GIB)
        );
    }

    #[test]
    fn an_ancestor_limit_binds_as_hard_as_the_leafs() {
        // Kubernetes puts the pod limit on the parent of the container's own
        // cgroup, so a walk that stopped at the leaf would read a pod capped at
        // 2 GiB as uncapped.
        let leaf = cgroup_level_headroom(CgroupLimit::Unlimited, Some(GIB));
        let parent = cgroup_level_headroom(CgroupLimit::Bytes(2 * GIB), Some(GIB));
        assert_eq!(
            narrower_headroom(leaf, parent),
            CgroupHeadroom::Limited(GIB)
        );
        // Two real limits: the tighter one is what the process is killed at.
        assert_eq!(
            narrower_headroom(
                CgroupHeadroom::Limited(32 * GIB),
                CgroupHeadroom::Limited(3 * GIB)
            ),
            CgroupHeadroom::Limited(3 * GIB)
        );
        // A charge above the limit is a cgroup already in reclaim, not headroom
        // that wrapped around.
        assert_eq!(
            cgroup_level_headroom(CgroupLimit::Bytes(GIB), Some(2 * GIB)),
            CgroupHeadroom::Limited(0)
        );
    }

    #[test]
    fn cgroup_limit_files_are_read_in_both_generations_spellings() {
        // v2 writes `max` for no limit; v1 writes a page-aligned saturation of
        // `i64`, which is not a limit either and must not read as one -- taking
        // it literally would leave 8 EiB of "headroom" and deepen everywhere.
        assert_eq!(parse_cgroup_limit("max\n"), CgroupLimit::Unlimited);
        assert_eq!(
            parse_cgroup_limit("9223372036854771712\n"),
            CgroupLimit::Unlimited
        );
        assert_eq!(
            parse_cgroup_limit("4294967296\n"),
            CgroupLimit::Bytes(4 * GIB)
        );
        assert_eq!(parse_cgroup_limit("unlimited\n"), CgroupLimit::Unparseable);
        assert_eq!(parse_cgroup_current("1073741824\n"), Some(GIB));
        assert_eq!(parse_cgroup_current("nan\n"), None);

        // `/proc/self/cgroup`: the v2 line is the one with no controllers, and
        // the v1 memory line may be co-mounted with other controllers.
        let hybrid = "12:cpu,cpuacct:/user.slice\n\
                      4:memory:/docker/abc\n\
                      0::/user.slice/user-1000.slice/session-18.scope\n";
        assert_eq!(
            cgroup_path(hybrid, None),
            Some("/user.slice/user-1000.slice/session-18.scope")
        );
        assert_eq!(cgroup_path(hybrid, Some("memory")), Some("/docker/abc"));
        // A co-mounted memory controller must not be missed by a prefix match.
        assert_eq!(
            cgroup_path("5:cpu,memory,pids:/kubepods/pod0\n", Some("memory")),
            Some("/kubepods/pod0")
        );
        assert_eq!(cgroup_path("5:cpu:/x\n", Some("memory")), None);
        assert_eq!(cgroup_path("", None), None);
    }

    /// The reading this machine actually performs has to be sane, whatever it
    /// finds: on a bare host that is `MemAvailable`, in a container it is the
    /// cap, and neither may come back as zero or as more than the host has.
    #[test]
    fn the_ambient_memory_reading_does_not_exceed_the_hosts() {
        use std::fs::read_to_string;

        let Some(available) = available_memory_bytes() else {
            // A limit whose numbers could not be read; the depth stays at the
            // floor, which is the safe direction.
            return;
        };
        let host = read_to_string("/proc/meminfo")
            .ok()
            .as_deref()
            .and_then(parse_mem_available);
        if let Some(host) = host {
            // `MemAvailable` is sampled twice here -- once inside
            // `available_memory_bytes` and once above -- and it moves between
            // the two reads on any busy machine, so a bare `<=` fails
            // intermittently (observed roughly one full-suite run in three).
            // The property worth pinning is that this never reports memory of a
            // different order than the host has, not that two samples of a
            // live counter agree, so allow generous drift.
            let tolerance = host / 16;
            assert!(
                available <= host.saturating_add(tolerance),
                "read {available} bytes available, well above the host's {host}"
            );
        }
    }
}
