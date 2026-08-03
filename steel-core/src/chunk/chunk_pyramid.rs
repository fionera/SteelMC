//! This module contains the `ChunkPyramid`, which is used to check chunk dependencies.
//! All structures are const-compatible and computed at compile time.

use std::sync::Arc;

use crate::chunk::{
    chunk_holder::ChunkHolder, chunk_status_tasks::ChunkStatusTasks,
    static_cache_2d::StaticCache2D, status::ChunkStatus,
};
use crate::worldgen::context::WorldGenContext;

/// Number of `ChunkStatus` variants.
const STATUS_COUNT: usize = 12;
/// Maximum dependency radius supported.
const MAX_RADIUS: usize = 16;

/// A collection of chunk dependencies (const-compatible).
#[derive(Debug, Clone, Copy)]
pub struct ChunkDependencies {
    dependency_by_radius: [Option<ChunkStatus>; MAX_RADIUS],
    len: usize,
    radius_by_dependency: [usize; STATUS_COUNT],
}

impl ChunkDependencies {
    /// Empty dependencies constant.
    pub const EMPTY: Self = Self {
        dependency_by_radius: [None; MAX_RADIUS],
        len: 0,
        radius_by_dependency: [0; STATUS_COUNT],
    };

    /// Creates dependencies from requirements and optional parent status.
    #[must_use]
    const fn from_requirements(
        reqs: &[(ChunkStatus, usize)],
        parent_status: Option<ChunkStatus>,
    ) -> Self {
        let mut dependency_by_radius = [None; MAX_RADIUS];
        let mut len = 0;

        // If we have a parent, start with parent at radius 0
        if let Some(parent) = parent_status {
            dependency_by_radius[0] = Some(parent);
            len = 1;
        }

        // Process requirements
        let mut i = 0;
        while i < reqs.len() {
            let (status, radius) = reqs[i];
            let new_len = radius + 1;

            // Extend if needed, filling with this status
            if new_len > len {
                let mut j = len;
                while j < new_len {
                    dependency_by_radius[j] = Some(status);
                    j += 1;
                }
                len = new_len;
            }

            // Update existing entries if this status is higher
            let limit = const_min(len, new_len);
            let mut j = 0;
            while j < limit {
                if let Some(existing) = dependency_by_radius[j]
                    && status.get_index() > existing.get_index()
                {
                    dependency_by_radius[j] = Some(status);
                }
                j += 1;
            }

            i += 1;
        }

        // Build radius_by_dependency
        let radius_by_dependency = Self::build_radius_lookup(&dependency_by_radius, len);

        Self {
            dependency_by_radius,
            len,
            radius_by_dependency,
        }
    }

    /// Builds the radius lookup table from dependency array.
    const fn build_radius_lookup(
        deps: &[Option<ChunkStatus>; MAX_RADIUS],
        len: usize,
    ) -> [usize; STATUS_COUNT] {
        let mut radius_by_dependency = [0usize; STATUS_COUNT];
        let mut radius = 0;
        while radius < len {
            if let Some(dep) = deps[radius] {
                let index = dep.get_index();
                let mut j = 0;
                while j <= index && j < STATUS_COUNT {
                    radius_by_dependency[j] = radius;
                    j += 1;
                }
            }
            radius += 1;
        }
        radius_by_dependency
    }

    /// Computes accumulated dependencies by merging with parent's accumulated dependencies.
    const fn accumulate(&self, parent_accumulated: &Self, parent_status: ChunkStatus) -> Self {
        // Find the last radius where we reference the parent status or higher
        let mut radius_of_parent = 0;
        let mut i = 0;
        while i < self.len {
            if let Some(s) = self.dependency_by_radius[i]
                && s.get_index() >= parent_status.get_index()
            {
                radius_of_parent = i;
            }
            i += 1;
        }

        let parent_len = parent_accumulated.len;
        let new_len = const_max(radius_of_parent + parent_len, self.len);
        let capped_len = const_min(new_len, MAX_RADIUS);

        let mut accumulated = [None; MAX_RADIUS];

        let mut dist = 0;
        while dist < capped_len {
            let dist_in_parent = dist.saturating_sub(radius_of_parent);

            let parent_dep = if dist_in_parent < parent_accumulated.len {
                parent_accumulated.dependency_by_radius[dist_in_parent]
            } else {
                None
            };

            let direct_dep = if dist < self.len {
                self.dependency_by_radius[dist]
            } else {
                None
            };

            accumulated[dist] = const_max_status(direct_dep, parent_dep);
            dist += 1;
        }

        let radius_by_dependency = Self::build_radius_lookup(&accumulated, capped_len);

        Self {
            dependency_by_radius: accumulated,
            len: capped_len,
            radius_by_dependency,
        }
    }

    /// Gets the radius of the dependencies for the given status.
    ///
    /// # Panics
    /// Panics if the status index is out of bounds.
    #[must_use]
    pub const fn get_radius_of(&self, status: ChunkStatus) -> usize {
        self.radius_by_dependency[status.get_index()]
    }

    /// Gets the radius of the dependencies.
    #[must_use]
    pub const fn get_radius(&self) -> usize {
        self.len.saturating_sub(1)
    }

    /// Gets the dependency status at the given distance.
    #[must_use]
    pub const fn get(&self, distance: usize) -> Option<ChunkStatus> {
        if distance < self.len {
            self.dependency_by_radius[distance]
        } else {
            None
        }
    }
}

const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn const_min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}

const fn const_max_status(a: Option<ChunkStatus>, b: Option<ChunkStatus>) -> Option<ChunkStatus> {
    match (a, b) {
        (Some(sa), Some(sb)) => {
            if sa.get_index() > sb.get_index() {
                Some(sa)
            } else {
                Some(sb)
            }
        }
        (Some(s), None) | (None, Some(s)) => Some(s),
        (None, None) => None,
    }
}

/// A task that generates a chunk.
pub type ChunkStatusTask =
    fn(Arc<WorldGenContext>, &ChunkStep, &Arc<StaticCache2D<Arc<ChunkHolder>>>, Arc<ChunkHolder>);

/// A chunk step (const-compatible).
#[derive(Clone, Copy)]
pub struct ChunkStep {
    /// The target status of the step.
    pub target_status: ChunkStatus,
    /// The direct dependencies of the step.
    pub direct_dependencies: ChunkDependencies,
    /// The accumulated dependencies of the step.
    pub accumulated_dependencies: ChunkDependencies,
    /// The block state write radius of the step.
    pub block_state_write_radius: i32,
    /// The task of the step.
    pub task: ChunkStatusTask,
}

impl ChunkStep {
    /// A placeholder step used for array initialization.
    const PLACEHOLDER: Self = Self {
        target_status: ChunkStatus::Empty,
        direct_dependencies: ChunkDependencies::EMPTY,
        accumulated_dependencies: ChunkDependencies::EMPTY,
        block_state_write_radius: -1,
        task: noop_task,
    };

    /// Gets the accumulated radius of the dependencies for the given status.
    #[must_use]
    pub const fn get_accumulated_radius_of(&self, status: ChunkStatus) -> usize {
        if status.get_index() == self.target_status.get_index() {
            0
        } else {
            self.accumulated_dependencies.get_radius_of(status)
        }
    }
}

fn noop_task(
    _context: Arc<WorldGenContext>,
    _step: &ChunkStep,
    _cache: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
    _holder: Arc<ChunkHolder>,
) {
}

/// Whether `next` may run in the same generation job as `prev`.
///
/// Every step is its own rayon job, tokio task and oneshot wake, so a chunk
/// costs twelve of each. Most of that dispatch is avoidable: consecutive steps
/// often need nothing from other chunks that the previous step did not already
/// need, and the halo the job is already holding satisfies them. Fusing such a
/// pair also lets the second step run with the first one's chunk data still in
/// cache, on the same worker.
///
/// Sound exactly when `next` introduces no *new* cross-chunk dependency: for
/// every status, what `next` needs from other chunks, `prev` already required
/// at least as far out.
///
/// Radius 0 means "this chunk only" -- that covers both `next`'s parent, which
/// `prev` has just produced on this chunk, and any status that is not a
/// dependency at all -- so a radius-0 entry never blocks fusion.
///
/// Note that requiring a status at radius *r* implies requiring every earlier
/// status at *r*, and `direct_dependencies` already carries that closure. So
/// `Spawn`, which needs `Biomes` at radius 1, fuses after `Light`, which needs
/// `InitializeLight` at radius 1 and therefore everything below it as well.
///
/// `Light` is excluded because it runs under the light work-window gate, whose
/// reservation is taken outside the job. `Empty` is excluded on both sides: it
/// loads from storage asynchronously instead of running on the generation pool.
///
/// Against the current pyramid this leaves six runs rather than twelve steps:
/// `Empty` | `StructureStarts` | `StructureReferences`+`Biomes` |
/// `Noise`+`Surface`+`Carvers` | `Features`+`InitializeLight` |
/// `Light`+`Spawn`+`Full`.
#[must_use]
pub const fn can_fuse(prev: &ChunkStep, next: &ChunkStep) -> bool {
    let next_index = next.target_status.get_index();
    if next_index == ChunkStatus::Empty.get_index()
        || next_index == ChunkStatus::Light.get_index()
        || prev.target_status.get_index() == ChunkStatus::Empty.get_index()
    {
        return false;
    }

    let mut index = 0;
    while index < STATUS_COUNT {
        let Some(current) = ChunkStatus::from_index(index) else {
            break;
        };
        let needed = next.direct_dependencies.get_radius_of(current);
        if needed > 0 && prev.direct_dependencies.get_radius_of(current) < needed {
            return false;
        }
        index += 1;
    }

    true
}

/// Halo radius the `Light` step reads, which its dependency ring does not say.
///
/// `run_light_stage` sets its workset up with [`LightCacheSetupRadius::Full`],
/// a 5x5 chunk window, while `Light`'s direct dependencies only reach radius 1.
/// The extra ring is opportunistic -- `light.rs` fetches it with `try_get` and
/// `setup_with_scopes` runs relaxed -- so under-sizing the halo here does not
/// fail, it silently treats the outer ring as empty and changes the lighting
/// result. That is why this is a named constant with its own test rather than
/// something derived from the pyramid.
///
/// [`LightCacheSetupRadius::Full`]: crate::chunk::light::LightCacheSetupRadius
pub const LIGHT_HALO_RADIUS: usize = 2;

/// One fused run of generation steps, and what it needs to be able to start.
///
/// Indexed by the status a run *starts* at, so [`RUN_PLANS`] has one entry per
/// status rather than one per run. That matters: a holder can legitimately sit
/// published in the middle of a run -- loaded from disk at, say, `Surface`, or
/// left there when a fused run published `Noise` and then had its remaining
/// claims rolled back -- and must resume from exactly the next status. A table
/// keyed by run would have to either miss that holder or restart it at the
/// run's first status, and `claim_status_work` panics on the latter.
#[derive(Clone, Copy, Debug)]
pub struct RunPlan {
    /// First status of the run.
    pub first: ChunkStatus,
    /// Last status of the run, inclusive.
    pub last: ChunkStatus,
    /// What the run needs of other chunks.
    ///
    /// This is the *first* step's direct dependencies. Property `A2` proves
    /// that dominates every later step in the run, which is what licenses
    /// resolving one halo and taking one dependency wait for the whole run.
    pub ring: ChunkDependencies,
    /// Chunks the halo must cover, as a Chebyshev radius.
    ///
    /// At least the ring's radius, and at least [`LIGHT_HALO_RADIUS`] for a run
    /// containing `Light`.
    pub halo_radius: usize,
}

impl RunPlan {
    const PLACEHOLDER: Self = Self {
        first: ChunkStatus::Empty,
        last: ChunkStatus::Empty,
        ring: ChunkDependencies::EMPTY,
        halo_radius: 0,
    };

    /// Whether `status` falls inside this run.
    #[must_use]
    pub const fn contains(&self, status: ChunkStatus) -> bool {
        let index = status.get_index();
        self.first.get_index() <= index && index <= self.last.get_index()
    }
}

/// The fused run starting at each status.
pub const RUN_PLANS: [RunPlan; STATUS_COUNT] = build_run_plans();

/// The most any run's ring asks of a neighbour at each Chebyshev distance.
///
/// This is what lets the generation drive stop re-reading a halo cell. A drive
/// re-checks its cached halo once per run, and published status is monotone, so
/// a cell seen at or past *every* requirement any plan can put on its distance
/// can never gate any of that drive's remaining runs and is dropped from the
/// walk for good. Against the current pyramid the entry is `StructureStarts`
/// from distance 2 outwards, which is exactly what a radius-8 halo has to
/// satisfy to resolve at all -- so the 289-cell re-check collapses to the nine
/// innermost cells after the first successful resolve.
///
/// Folded over every plan rather than over the ones a drive has left to run. A
/// drive's plan sequence only moves forward, so a per-plan suffix would be
/// tighter, but the only distances where the two differ are 0 and 1 -- nine
/// cells of the square -- and this table cannot go stale against a drive
/// resuming mid-pyramid off a disk load.
const MAX_RING_REQUIREMENT: [Option<ChunkStatus>; MAX_RADIUS] = build_max_ring_requirement();

const fn build_max_ring_requirement() -> [Option<ChunkStatus>; MAX_RADIUS] {
    let mut table = [None; MAX_RADIUS];
    let mut distance = 0;
    while distance < MAX_RADIUS {
        let mut index = 0;
        while index < STATUS_COUNT {
            table[distance] =
                const_max_status(table[distance], RUN_PLANS[index].ring.get(distance));
            index += 1;
        }
        distance += 1;
    }
    table
}

/// The most any run's ring asks of a neighbour at `distance`.
///
/// `None` means no ring names that distance at all: cells out there are
/// halo-only -- present so the step can read them, gating nothing -- so they are
/// settled the moment they are resolved.
#[must_use]
pub const fn max_ring_requirement(distance: usize) -> Option<ChunkStatus> {
    if distance < MAX_RADIUS {
        MAX_RING_REQUIREMENT[distance]
    } else {
        None
    }
}

const fn build_run_plans() -> [RunPlan; STATUS_COUNT] {
    let mut plans = [RunPlan::PLACEHOLDER; STATUS_COUNT];
    let mut index = 0;

    while index < STATUS_COUNT {
        let Some(first) = ChunkStatus::from_index(index) else {
            panic!("status index within STATUS_COUNT must decode")
        };
        let first_step = *GENERATION_PYRAMID.get_step_to(first);

        // Walk forward while the next step needs nothing new from other chunks.
        let mut last_index = index;
        let mut previous = first_step;
        while last_index + 1 < STATUS_COUNT {
            let Some(next) = ChunkStatus::from_index(last_index + 1) else {
                break;
            };
            let next_step = *GENERATION_PYRAMID.get_step_to(next);
            if !can_fuse(&previous, &next_step) {
                break;
            }
            last_index += 1;
            previous = next_step;
        }

        let Some(last) = ChunkStatus::from_index(last_index) else {
            panic!("run end index must decode")
        };
        let ring = first_step.direct_dependencies;
        let mut halo_radius = ring.get_radius();
        let light_index = ChunkStatus::Light.get_index();
        if index <= light_index && light_index <= last_index && halo_radius < LIGHT_HALO_RADIUS {
            halo_radius = LIGHT_HALO_RADIUS;
        }

        plans[index] = RunPlan {
            first,
            last,
            ring,
            halo_radius,
        };
        index += 1;
    }

    plans
}

/// Represents the hierarchy and dependencies for chunk generation or loading.
pub struct ChunkPyramid {
    steps: [ChunkStep; STATUS_COUNT],
}

impl ChunkPyramid {
    /// Gets the step for the given status.
    #[must_use]
    pub const fn get_step_to(&self, status: ChunkStatus) -> &ChunkStep {
        &self.steps[status.get_index()]
    }
}

/// Const-time pyramid builder.
struct ConstPyramidBuilder {
    steps: [ChunkStep; STATUS_COUNT],
    count: usize,
}

impl ConstPyramidBuilder {
    const fn new() -> Self {
        Self {
            steps: [ChunkStep::PLACEHOLDER; STATUS_COUNT],
            count: 0,
        }
    }

    const fn step(
        mut self,
        status: ChunkStatus,
        requirements: &[(ChunkStatus, usize)],
        block_state_write_radius: i32,
        task: ChunkStatusTask,
    ) -> Self {
        // Get parent info if we have previous steps
        let (parent_status, parent_accumulated) = if self.count > 0 {
            let parent = &self.steps[self.count - 1];
            (
                Some(parent.target_status),
                Some(parent.accumulated_dependencies),
            )
        } else {
            (None, None)
        };

        // Compute direct dependencies
        let direct = ChunkDependencies::from_requirements(requirements, parent_status);

        // Compute accumulated dependencies
        let accumulated = match (parent_status, parent_accumulated) {
            (Some(ps), Some(pa)) => direct.accumulate(&pa, ps),
            _ => direct,
        };

        self.steps[self.count] = ChunkStep {
            target_status: status,
            direct_dependencies: direct,
            accumulated_dependencies: accumulated,
            block_state_write_radius,
            task,
        };
        self.count += 1;
        self
    }

    const fn build(self) -> ChunkPyramid {
        ChunkPyramid { steps: self.steps }
    }
}

/// Macro for defining chunk pyramids with nice syntax.
///
/// # Example
/// ```ignore
/// define_pyramid! {
///     pub static MY_PYRAMID = {
///         Empty => { task: my_task },
///         StructureStarts => {
///             requirements: [(StructureStarts, 8)],
///             task: other_task,
///         },
///     };
/// }
/// ```
macro_rules! define_pyramid {
    (
        $vis:vis const $name:ident = {
            $($status:ident => {
                $(requirements: [$( ($req_status:ident, $req_radius:expr) ),* $(,)?] ,)?
                $(block_state_write_radius: $bswr:expr ,)?
                task: $task:expr $(,)?
            }),* $(,)?
        };
    ) => {
        #[expect(missing_docs, reason = "generated pyramid constant; name is self-documenting")]
        $vis const $name: &'static ChunkPyramid = &{
            ConstPyramidBuilder::new()
            $(
                .step(
                    ChunkStatus::$status,
                    &[ $( $( (ChunkStatus::$req_status, $req_radius) ),* )? ],
                    define_pyramid!(@bswr $($bswr)?),
                    $task,
                )
            )*
            .build()
        };
    };

    // Default block_state_write_radius
    (@bswr) => { -1 };
    (@bswr $bswr:expr) => { $bswr };
}

define_pyramid! {
    pub const GENERATION_PYRAMID = {
        Empty => {
            task: ChunkStatusTasks::empty,
        },
        StructureStarts => {
            task: ChunkStatusTasks::generate_structure_starts,
        },
        StructureReferences => {
            requirements: [(StructureStarts, 8)],
            task: ChunkStatusTasks::generate_structure_references,
        },
        Biomes => {
            requirements: [(StructureStarts, 8)],
            task: ChunkStatusTasks::generate_biomes,
        },
        Noise => {
            requirements: [(StructureStarts, 8), (Biomes, 1)],
            block_state_write_radius: 0,
            task: ChunkStatusTasks::generate_noise,
        },
        Surface => {
            requirements: [(StructureStarts, 8), (Biomes, 1)],
            block_state_write_radius: 0,
            task: ChunkStatusTasks::generate_surface,
        },
        Carvers => {
            requirements: [(StructureStarts, 8)],
            block_state_write_radius: 0,
            task: ChunkStatusTasks::generate_carvers,
        },
        Features => {
            requirements: [(StructureStarts, 8), (Carvers, 1)],
            block_state_write_radius: 1,
            task: ChunkStatusTasks::generate_features,
        },
        InitializeLight => {
            task: ChunkStatusTasks::initialize_light,
        },
        Light => {
            requirements: [(InitializeLight, 1)],
            block_state_write_radius: 0,
            task: ChunkStatusTasks::light,
        },
        Spawn => {
            requirements: [(Biomes, 1)],
            task: ChunkStatusTasks::generate_spawn,
        },
        Full => {
            task: ChunkStatusTasks::full,
        },
    };
}


#[cfg(test)]
mod fusion_tests {
    use super::{ChunkStatus, GENERATION_PYRAMID, can_fuse};

    /// The fused runs the current pyramid produces.
    ///
    /// Pinned deliberately: fusion is only sound while a fused step needs
    /// nothing from other chunks that its predecessor did not already need, so
    /// a requirement added to any step here must show up as a change to this
    /// list rather than silently widening what runs inside one job.
    #[test]
    fn fused_runs_match_the_pyramid_dependencies() {
        let mut runs: Vec<Vec<ChunkStatus>> = Vec::new();
        let mut current = Some(ChunkStatus::Empty);
        while let Some(status) = current {
            let step = GENERATION_PYRAMID.get_step_to(status);
            let fuses = runs.last().is_some_and(|_| {
                status.parent().is_some_and(|parent| {
                    can_fuse(GENERATION_PYRAMID.get_step_to(parent), step)
                })
            });
            if fuses {
                runs.last_mut().expect("run exists").push(status);
            } else {
                runs.push(vec![status]);
            }
            current = status.next();
        }

        assert_eq!(
            runs,
            vec![
                vec![ChunkStatus::Empty],
                vec![ChunkStatus::StructureStarts],
                vec![ChunkStatus::StructureReferences, ChunkStatus::Biomes],
                vec![
                    ChunkStatus::Noise,
                    ChunkStatus::Surface,
                    ChunkStatus::Carvers
                ],
                vec![ChunkStatus::Features, ChunkStatus::InitializeLight],
                vec![
                    ChunkStatus::Light,
                    ChunkStatus::Spawn,
                    ChunkStatus::Full
                ],
            ],
        );
    }

    #[test]
    fn a_new_cross_chunk_requirement_blocks_fusion() {
        // Features needs Carvers at radius 1, which Carvers itself did not
        // require of its neighbours, so it must start its own job.
        assert!(!can_fuse(
            GENERATION_PYRAMID.get_step_to(ChunkStatus::Carvers),
            GENERATION_PYRAMID.get_step_to(ChunkStatus::Features),
        ));
        // Surface needs exactly what Noise needed.
        assert!(can_fuse(
            GENERATION_PYRAMID.get_step_to(ChunkStatus::Noise),
            GENERATION_PYRAMID.get_step_to(ChunkStatus::Surface),
        ));
    }

    #[test]
    fn light_is_never_fused() {
        assert!(!can_fuse(
            GENERATION_PYRAMID.get_step_to(ChunkStatus::InitializeLight),
            GENERATION_PYRAMID.get_step_to(ChunkStatus::Light),
        ));
    }
}

#[cfg(test)]
mod run_plan_tests {
    use super::{
        ChunkStatus, GENERATION_PYRAMID, LIGHT_HALO_RADIUS, MAX_RADIUS, RUN_PLANS, STATUS_COUNT,
        max_ring_requirement,
    };
    use crate::chunk::chunk_ticket_manager::{ChunkTicketLevel, generation_status};

    fn statuses() -> impl Iterator<Item = ChunkStatus> {
        (0..STATUS_COUNT).filter_map(ChunkStatus::from_index)
    }

    /// A1 -- waits go strictly down the status chain, so they cannot cycle.
    ///
    /// This is the deadlock-freedom theorem for the whole per-holder model. If
    /// every cross-chunk requirement of a run is for a status strictly below
    /// the one the run produces, then a wait edge always points down a
    /// 12-element chain. `Empty` has no requirements at all, so some holder is
    /// always dispatchable and the system cannot come to rest with work left.
    #[test]
    fn cross_chunk_requirements_are_strictly_below_the_run() {
        for status in statuses() {
            let plan = &RUN_PLANS[status.get_index()];
            for distance in 1..=plan.ring.get_radius() {
                let Some(required) = plan.ring.get(distance) else {
                    continue;
                };
                assert!(
                    required.get_index() < plan.first.get_index(),
                    "{:?} needs {required:?} at distance {distance}, which is not strictly \
                     below it -- a wait edge that does not descend can close a cycle",
                    plan.first,
                );
            }
        }
    }

    /// A2 -- the first step's ring dominates every step in its run.
    ///
    /// This is what licenses resolving one halo and taking one dependency wait
    /// for a whole fused run instead of re-checking between steps.
    #[test]
    fn the_first_steps_ring_dominates_the_whole_run() {
        for status in statuses() {
            let plan = &RUN_PLANS[status.get_index()];
            for index in plan.first.get_index()..=plan.last.get_index() {
                let member = ChunkStatus::from_index(index).expect("run member decodes");
                let step = GENERATION_PYRAMID.get_step_to(member);
                for dependency in statuses() {
                    assert!(
                        step.direct_dependencies.get_radius_of(dependency)
                            <= plan.ring.get_radius_of(dependency),
                        "run starting at {:?}: {member:?} needs {dependency:?} further out \
                         than the run's ring provides",
                        plan.first,
                    );
                }
            }
        }
    }

    /// A3 -- a halo chunk is always allowed to reach what its neighbours need.
    ///
    /// Ticket level `FULL + k` allows exactly `generation_status(FULL + k)`. A
    /// chunk at distance `d` from it sits at level `FULL + k + d`. This asserts
    /// that what that neighbour is allowed to reach is at least what the run at
    /// `FULL + k` requires of distance `d`.
    ///
    /// This was previously only an empirical claim -- instrumenting the working
    /// scheduler over a 301x301 pregeneration counted zero violations. This
    /// makes it a property of the tables instead of a property of one run.
    #[test]
    fn every_ring_cell_is_allowed_to_reach_what_the_run_needs_of_it() {
        let span = usize::from(ChunkTicketLevel::MAX.raw() - ChunkTicketLevel::FULL_CHUNK.raw());
        for offset in 0..=span {
            let raw = ChunkTicketLevel::FULL_CHUNK.raw() + offset as u8;
            let Some(level) = ChunkTicketLevel::new(raw) else {
                continue;
            };
            let Some(target) = generation_status(Some(level)) else {
                continue;
            };
            let plan = &RUN_PLANS[target.get_index()];

            for distance in 0..=plan.ring.get_radius() {
                let Some(required) = plan.ring.get(distance) else {
                    continue;
                };
                let neighbour_raw = raw as usize + distance;
                let allowed = u8::try_from(neighbour_raw)
                    .ok()
                    .and_then(ChunkTicketLevel::new)
                    .and_then(|level| generation_status(Some(level)));
                let Some(allowed) = allowed else {
                    panic!(
                        "a chunk at level {raw} running {target:?} needs distance {distance} at \
                         {required:?}, but level {neighbour_raw} carries no generation status \
                         at all -- that cell has no holder"
                    );
                };
                assert!(
                    allowed.get_index() >= required.get_index(),
                    "a chunk at level {raw} running {target:?} needs distance {distance} at \
                     {required:?}, but level {neighbour_raw} is only allowed {allowed:?}"
                );
            }
        }
    }

    /// A4 -- the ticket span is exactly the pyramid's widest reach.
    ///
    /// The boundary rings have no slack, so this pins the two together: widen
    /// the pyramid without widening the level span and A3 starts failing.
    #[test]
    fn the_ticket_level_span_matches_the_pyramids_reach() {
        let span = usize::from(ChunkTicketLevel::MAX.raw() - ChunkTicketLevel::FULL_CHUNK.raw());
        assert_eq!(
            span,
            GENERATION_PYRAMID
                .get_step_to(ChunkStatus::Full)
                .accumulated_dependencies
                .get_radius_of(ChunkStatus::Empty),
        );
    }

    /// A5 -- the halo covers every chunk a run actually reads.
    ///
    /// The `Light` assertion is separate and separately messaged because it is
    /// the one case that fails *silently*: `run_light_stage` fetches its outer
    /// ring with `try_get` and runs its workset relaxed, so a halo that is one
    /// short does not panic, it lights the chunk differently.
    #[test]
    fn the_halo_covers_every_chunk_a_run_reads() {
        for status in statuses() {
            let plan = &RUN_PLANS[status.get_index()];
            for index in plan.first.get_index()..=plan.last.get_index() {
                let member = ChunkStatus::from_index(index).expect("run member decodes");
                let step = GENERATION_PYRAMID.get_step_to(member);
                assert!(
                    plan.halo_radius >= step.direct_dependencies.get_radius(),
                    "run starting at {:?} resolves a halo of radius {}, but {member:?} reads \
                     out to {}",
                    plan.first,
                    plan.halo_radius,
                    step.direct_dependencies.get_radius(),
                );
            }
        }
    }

    /// A run's ring never asks *more* of a farther neighbour than of a nearer
    /// one.
    ///
    /// `resolve_and_check` walks the halo outwards-in, so this is what makes the
    /// unmet set it returns ordered from the lowest requirement to the highest,
    /// and that ordering is the whole basis of `fanout_selection`: it takes the
    /// tail because the tail is the most constraining end. If a table change
    /// ever made a ring rise with radius, the park would silently start
    /// registering on cells that clear first and every blocked chunk would need
    /// several more admissions to make one run's progress.
    #[test]
    fn a_runs_ring_never_rises_with_radius() {
        for status in statuses() {
            let plan = &RUN_PLANS[status.get_index()];
            for distance in 1..=plan.ring.get_radius() {
                let (Some(nearer), Some(farther)) =
                    (plan.ring.get(distance - 1), plan.ring.get(distance))
                else {
                    continue;
                };
                assert!(
                    farther.get_index() <= nearer.get_index(),
                    "the run starting at {:?} needs {farther:?} at distance {distance} but only \
                     {nearer:?} at {}; the park selects the most constraining dependencies by \
                     taking the outermost-first walk from its tail, which that inverts",
                    plan.first,
                    distance - 1,
                );
            }
        }
    }

    /// A8 -- the retirement table dominates every ring.
    ///
    /// The drive's halo re-check drops a cell as soon as it has been seen at
    /// `max_ring_requirement` for its distance. If the table ever understated a
    /// plan's ring, the cell that plan gates on would be dropped before it was
    /// checked, and the run would dispatch against a neighbour that has not
    /// reached the status it reads -- `claim_status_work`'s compare-exchange
    /// panic, or the `has_parent` assertion in `apply_generated_steps`.
    #[test]
    fn the_retirement_table_dominates_every_runs_ring() {
        for status in statuses() {
            let plan = &RUN_PLANS[status.get_index()];
            for distance in 0..=plan.halo_radius {
                assert!(
                    max_ring_requirement(distance) >= plan.ring.get(distance),
                    "the run starting at {:?} needs {:?} at distance {distance}, but a re-check \
                     retires that distance at {:?}",
                    plan.first,
                    plan.ring.get(distance),
                    max_ring_requirement(distance),
                );
            }
        }
        // Distances the table does not cover must read as "nothing is ever asked
        // here", not index out of bounds.
        assert_eq!(max_ring_requirement(MAX_RADIUS), None);
        assert_eq!(max_ring_requirement(usize::MAX), None);
    }

    /// What makes the re-check cheap rather than merely correct.
    ///
    /// A radius-8 halo only resolves once every cell is at `StructureStarts`,
    /// and from distance 2 outwards that is also the most any ring asks -- so
    /// the 280 outer cells retire on the resolve that produced them and the
    /// three remaining re-checks of that halo walk nine cells. A pyramid change
    /// that raised any outer entry would put those 280 cells back into every
    /// re-check without breaking a thing, so it is pinned here.
    #[test]
    fn the_outer_rings_retire_at_the_status_a_wide_halo_already_needs() {
        for distance in 2..=RUN_PLANS[ChunkStatus::Noise.get_index()].halo_radius {
            assert_eq!(
                max_ring_requirement(distance),
                Some(ChunkStatus::StructureStarts),
                "distance {distance} no longer retires at the status a radius-8 resolve proves",
            );
        }
    }

    #[test]
    fn the_light_run_keeps_its_five_by_five_window() {
        let plan = &RUN_PLANS[ChunkStatus::Light.get_index()];
        assert_eq!(
            plan.halo_radius, LIGHT_HALO_RADIUS,
            "the light stage reads a 5x5 chunk window opportunistically with try_get; a smaller \
             halo does not fail, it silently treats the outer ring as empty and changes the \
             lighting output",
        );
        assert!(plan.contains(ChunkStatus::Light));
    }

    /// A6 -- every status a ticket level can allow is the END of a run.
    ///
    /// This is what makes truncating a run at the allowed status dead code in
    /// production: a holder is never allowed to stop half way through a run.
    #[test]
    fn every_allowed_status_is_a_run_terminal() {
        let span = usize::from(ChunkTicketLevel::MAX.raw() - ChunkTicketLevel::FULL_CHUNK.raw());
        for offset in 0..=span {
            let raw = ChunkTicketLevel::FULL_CHUNK.raw() + offset as u8;
            let Some(level) = ChunkTicketLevel::new(raw) else {
                continue;
            };
            let Some(allowed) = generation_status(Some(level)) else {
                continue;
            };
            let plan = &RUN_PLANS[allowed.get_index()];
            assert_eq!(
                plan.last, allowed,
                "level {raw} allows {allowed:?}, which is not the last status of its run \
                 (run is {:?}..={:?}) -- a holder would have to stop mid-run",
                plan.first, plan.last,
            );
        }
    }

    /// A7 -- `RUN_PLANS` reproduces the fused run list.
    #[test]
    fn run_plans_agree_with_the_pinned_fusion_list() {
        let mut runs: Vec<Vec<ChunkStatus>> = Vec::new();
        let mut index = 0;
        while index < STATUS_COUNT {
            let status = ChunkStatus::from_index(index).expect("status decodes");
            let plan = &RUN_PLANS[index];
            assert_eq!(plan.first, status);
            let members: Vec<ChunkStatus> = (plan.first.get_index()..=plan.last.get_index())
                .filter_map(ChunkStatus::from_index)
                .collect();
            runs.push(members);
            index = plan.last.get_index() + 1;
        }

        assert_eq!(
            runs,
            vec![
                vec![ChunkStatus::Empty],
                vec![ChunkStatus::StructureStarts],
                vec![ChunkStatus::StructureReferences, ChunkStatus::Biomes],
                vec![
                    ChunkStatus::Noise,
                    ChunkStatus::Surface,
                    ChunkStatus::Carvers
                ],
                vec![ChunkStatus::Features, ChunkStatus::InitializeLight],
                vec![
                    ChunkStatus::Light,
                    ChunkStatus::Spawn,
                    ChunkStatus::Full
                ],
            ],
        );
    }
}
