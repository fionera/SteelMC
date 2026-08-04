//! Optional L3-domain pinning for the chunk generation pool, and an optional
//! reservation of domains that only generation runs on.
//!
//! Generation is bound by cache, not by dispatch: over a 601x601 pregeneration
//! 89.2% of fills came from the local L2 and 7.0% from the local CCX's L3, with
//! only 3.8% from DRAM and none from another CCX. The same run showed ~40,000
//! CPU migrations a second, and a migration across an L3 boundary discards the
//! warm L2 *and* the 32 MiB L3 behind it, including that CCX's copy of the
//! ~2.9 MB climate R-tree biome lookup streams 1,536 times per chunk.
//!
//! Confining each worker to one L3 domain is the cheapest thing that can be
//! done about that. On its own it measured **flat**: 10,613 chunks/s pinned
//! against 10,577 unpinned, with migrations down only 2%. The reason is that
//! pinning reserves nothing. About 172 threads run on this box's 128 CPUs --
//! generation 120, the chunk runtime 24, the main runtime 16, the encoding pool
//! 12 -- and only generation was pinned, so the other 52 kept landing on
//! generation's CPUs and preempting it at ~385,000 context switches a second. A
//! worker that keeps its mask but loses its core still loses its cache.
//!
//! [`RESERVE_ENV`] is the experiment that follows from that reading: give
//! generation a set of L3 domains nothing else runs on, and confine the two
//! tokio runtimes and the encoding pool to the rest. Whether it buys anything is
//! not known -- nothing has been measured with it yet.
//!
//! Both switches are runtime rather than cargo features because the intended use
//! is A/B-ing two runs of one binary, where a rebuild between arms would change
//! more than the pinning.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::env;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use steel_utils::cpu::affinity::{current_thread_cpus, pin_current_thread};
use steel_utils::cpu::topology::{CacheDomain, L3Domains, format_cpu_list};
use tokio::runtime::Builder as RuntimeBuilder;

/// Runtime switch for generation-pool pinning, default off.
const PIN_ENV: &str = "STEEL_GENERATION_PIN_L3";

/// How many L3 domains to keep clear of generation, default 0.
///
/// Only read when [`PIN_ENV`] is on: with generation unpinned there is nothing
/// to reserve *from*, and confining the runtimes alone would be a third arm
/// nobody asked for.
const RESERVE_ENV: &str = "STEEL_GENERATION_RESERVE_DOMAINS";

/// Values of [`PIN_ENV`] that turn pinning on.
const TRUTHY: [&str; 4] = ["1", "true", "yes", "on"];

/// What runs on the reserved domains, named the same way in the summary and in
/// every warning so a log can be grepped for one of them.
const RESERVED_TENANTS: &str = "the chunk runtime, the main runtime and the encoding pool";

/// The one resolution for the process.
///
/// [`GenerationAffinity::resolve`] reads the calling thread's affinity mask,
/// and that read only describes the machine until something has been pinned.
/// The server resolves from `main`, before either tokio runtime is built, and
/// `Server::new_with_commands` resolves again to get at the same decision. That
/// second call is on the thread that entered the runtime through `block_on`,
/// which `run_server` pins to the reserved set once the pools exist -- so a
/// recomputation there could read the reserved set as if it were the whole
/// machine and hand generation the domains it was supposed to stay off. The
/// decision is therefore made once, on whichever thread asks first, and reused.
///
/// This protects the *plan*. The thread counts the plan is sized against are a
/// separate hazard with a separate fix: see
/// [`steel_utils::cpu::parallelism`], since `available_parallelism` reports the
/// caller's mask too.
static RESOLVED: OnceLock<GenerationAffinity> = OnceLock::new();

/// What to do with the generation pool's worker threads and with everything
/// else's, decided once at startup.
pub struct GenerationAffinity {
    /// `None` when pinning is off or was skipped.
    plan: Option<Arc<PinPlan>>,
    /// `None` unless [`RESERVE_ENV`] asked for domains and they could be spared.
    reserved: Option<Arc<ReservedPlan>>,
    summary: String,
}

/// A reading of the pin counters, to be handed back to [`GenerationAffinity::verify`]
/// and [`GenerationAffinity::verify_reserved`].
///
/// The counters live in the one process-wide decision (see [`RESOLVED`]), but
/// the pregen bench builds a fresh generation pool, two runtimes and an
/// encoding pool for every repetition. Reported as running totals, repetition 2
/// re-reports repetition 1's failures and the totals outgrow their own
/// denominator -- "did not take effect for 336 of 112 workers" -- and a
/// repetition that pinned cleanly after a dirty one still reads as dirty.
/// Reporting against a checkpoint keeps each line about the pools its caller
/// just built.
#[derive(Clone, Copy)]
pub struct PinCheckpoint {
    generation_failures: usize,
    reserved_threads: usize,
    reserved_failures: usize,
}

impl PinCheckpoint {
    /// The reading before this process pinned anything.
    ///
    /// For a process that builds one set of pools and wants the totals -- the
    /// server, where the runtimes are pinned in `main` long before anything is
    /// in a position to take a checkpoint.
    pub const PROCESS_START: Self = Self {
        generation_failures: 0,
        reserved_threads: 0,
        reserved_failures: 0,
    };
}

/// Which L3 domain each generation worker index belongs to.
struct PinPlan {
    /// Domains already narrowed to the CPUs this process may use, and already
    /// stripped of any reserved ones.
    domains: Box<[CacheDomain]>,
    workers: usize,
    /// Workers whose `sched_setaffinity` call failed, counted by the start
    /// handler so the outcome can be reported rather than only logged.
    failures: AtomicUsize,
}

/// The domains generation is kept off, and their CPUs as one mask.
struct ReservedPlan {
    /// Union of the reserved domains' CPUs, ascending.
    cpus: Box<[usize]>,
    /// Threads that asked to be pinned here, and the ones that could not be.
    /// Counted for the same reason [`PinPlan::failures`] is: `log` is a no-op
    /// until a logger exists and the pregen bench installs none.
    threads: AtomicUsize,
    failures: AtomicUsize,
}

impl ReservedPlan {
    /// Pins the calling thread to the reserved set, warning rather than failing.
    fn pin_calling_thread(&self, tenant: &str) {
        self.threads.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = pin_current_thread(&self.cpus) {
            self.failures.fetch_add(1, Ordering::Relaxed);
            // A failure leaves this thread's mask untouched, so it keeps the
            // process-wide one: the thread is unconfined, never confined to
            // something wrong. That is a worse arm, not a broken server.
            log::warn!(
                "{tenant} thread could not be pinned to the reserved L3 domains (cpus {}): {error}; it keeps the default affinity mask and may run on generation's CPUs",
                format_cpu_list(&self.cpus)
            );
        }
    }
}

impl PinPlan {
    fn domain_for_worker(&self, worker: usize) -> &CacheDomain {
        let index = domain_index_for_worker(worker, self.workers, self.domains.len());
        &self.domains[index]
    }

    fn cpus(&self) -> usize {
        self.domains.iter().map(|domain| domain.cpus().len()).sum()
    }

    /// Renders the worker-block assignment, e.g.
    /// `workers 0-14 -> cpus 0-7,64-71;`, one clause per block.
    fn describe_blocks(&self) -> String {
        let mut blocks = String::new();
        let mut block_start = 0;
        for worker in 0..self.workers {
            let last = worker + 1 == self.workers;
            let ends_block = last
                || domain_index_for_worker(worker, self.workers, self.domains.len())
                    != domain_index_for_worker(worker + 1, self.workers, self.domains.len());
            if !ends_block {
                continue;
            }
            let domain = self.domain_for_worker(block_start);
            let workers = if block_start == worker {
                format!("worker {worker}")
            } else {
                format!("workers {block_start}-{worker}")
            };
            // A write to a String cannot fail.
            let _ = write!(blocks, " {workers} -> cpus {domain};");
            block_start = worker + 1;
        }
        blocks
    }
}

/// Maps a worker index onto a domain index in contiguous blocks.
///
/// Contiguous, not round-robin: neighbouring worker indices have to end up on
/// the same domain, because the follow-up step routes spatially adjacent chunks
/// to neighbouring workers and round-robin would scatter exactly the work that
/// should share a cache.
///
/// The multiply-then-divide also spreads the remainder instead of overloading
/// the last domain: 120 workers over 8 domains gives 15 each, 121 gives one
/// domain 16, and 110 over the 7 domains a one-domain reservation leaves gives
/// five domains 16 and two 15 -- spread through the range, not stacked at the
/// end.
fn domain_index_for_worker(worker: usize, workers: usize, domains: usize) -> usize {
    debug_assert!(workers > 0 && domains > 0, "resolve rejects empty inputs");
    (worker * domains / workers.max(1)).min(domains.saturating_sub(1))
}

/// Narrows a machine's L3 domains to the ones this process can usefully pin
/// `workers` threads across, given the CPUs it is `allowed` to run on.
///
/// The intersection is the whole point. The sysfs cache tree describes the
/// *machine*: it is namespaced by neither cgroup nor affinity, so `cpu100/cache`
/// is present even inside a container restricted to cpus 0-7. A plan taken
/// straight from it goes wrong in two directions. Under `taskset` or
/// `numactl --physcpubind` the kernel lets a thread widen its own mask, so
/// pinning would scatter workers over CPUs the operator excluded -- and against
/// an unpinned control arm that honours the restriction, the A/B would be
/// comparing two different machines. Under a cpuset cgroup the kernel refuses
/// instead, so only the workers whose domain happened to overlap the cpuset get
/// pinned and the rest fail with `EINVAL`: the half-pinned pool, with a summary
/// line claiming a full pin.
///
/// Returns the usable domains and a note for the summary, or the reason to skip.
fn usable_domains(
    topology: &L3Domains,
    allowed: &[usize],
    workers: usize,
) -> Result<(Vec<CacheDomain>, String), String> {
    let machine_domains = topology.domains().len();
    let machine_cpus = topology.total_cpus();
    if let [single] = topology.domains() {
        return Err(format!(
            "the machine reports a single L3 domain of {single}; pinning could only take CPUs away"
        ));
    }

    let domains: Vec<CacheDomain> = topology
        .domains()
        .iter()
        .filter_map(|domain| domain.intersect(allowed))
        .collect();
    if domains.len() < 2 {
        return Err(format!(
            "the {} CPUs this process may use reach only {} of the machine's {machine_domains} L3 domains; with fewer than two domains to spread over, pinning could only take CPUs away",
            allowed.len(),
            domains.len()
        ));
    }
    let total_cpus: usize = domains.iter().map(|domain| domain.cpus().len()).sum();
    if total_cpus < workers {
        // Fewer usable CPUs than workers means every block is oversubscribed
        // before the run starts, so the migrations pinning exists to stop would
        // simply happen inside a domain instead.
        return Err(format!(
            "only {total_cpus} CPUs across {} L3 domains are usable by this process, fewer than the pool's {workers} workers",
            domains.len()
        ));
    }

    let mask_note = if total_cpus == machine_cpus {
        String::new()
    } else {
        format!(
            " (narrowed by this process's affinity mask from the machine's {machine_cpus} CPUs in {machine_domains} domains)"
        )
    };
    Ok((domains, mask_note))
}

/// How the usable domains were carved up.
struct Split {
    /// The domains generation workers are spread over. Never empty.
    generation: Vec<CacheDomain>,
    /// The mask nothing but [`RESERVED_TENANTS`] runs on. `None` when no
    /// reservation was asked for or when one was refused.
    reserved: Option<Arc<ReservedPlan>>,
    /// The clause the summary ends with, stating what was reserved or why
    /// nothing was.
    note: String,
}

/// [`RESERVE_ENV`] as read.
struct Requested {
    /// How many domains to reserve.
    domains: usize,
    /// How to name the setting in the summary. Not derivable from `domains`:
    /// unset and unparseable both mean zero domains, and printing them as `=0`
    /// would tell an operator sweeping this variable that they typed a 0 when
    /// they typed a typo.
    setting: String,
}

/// Splits `domains` into the block generation may use and the last
/// `requested.domains`, which are reserved for everything else.
///
/// The *last* rather than the first only so that an operator sweeping
/// `RESERVE_ENV` upwards sees generation's lowest domain stay put across the
/// sweep; nothing about the hardware prefers either end.
///
/// Never fails. A reservation that cannot be honoured leaves the existing
/// `RESERVE_ENV`-unset behaviour in place and says so, because the alternatives
/// are refusing to start over a tuning knob or silently shrinking the pool the
/// operator is sweeping against.
fn split_reserved(mut domains: Vec<CacheDomain>, requested: &Requested, workers: usize) -> Split {
    let usable = domains.len();
    let setting = &requested.setting;
    let refused = |reason: String| {
        format!(
            "no L3 domains reserved: {reason}, so {RESERVED_TENANTS} keep the process-wide mask and stay free to run on generation's CPUs"
        )
    };

    if requested.domains == 0 {
        return Split {
            generation: domains,
            reserved: None,
            note: refused(setting.clone()),
        };
    }
    if requested.domains >= usable {
        return Split {
            generation: domains,
            reserved: None,
            note: refused(format!(
                "{setting} is not fewer than the {usable} usable L3 domains, and generation needs at least one"
            )),
        };
    }

    let kept = usable - requested.domains;
    let generation_cpus: usize = domains[..kept]
        .iter()
        .map(|domain| domain.cpus().len())
        .sum();
    if generation_cpus < workers {
        // Reserving here would oversubscribe generation's own domains, which is
        // the thing the reservation exists to stop happening to them. Refusing
        // is not a judgement about the right pool size -- the operator sweeps
        // that -- only about this pairing of the two.
        return Split {
            generation: domains,
            reserved: None,
            note: refused(format!(
                "{setting} would leave {generation_cpus} CPUs across {kept} L3 domains for {workers} generation workers"
            )),
        };
    }

    // The tenants are pinned to the *union*, not domain by domain: they are not
    // the workload this is protecting, and giving them the whole reserved block
    // leaves the kernel free to balance them across it.
    let mut cpus: Vec<usize> = domains
        .split_off(kept)
        .iter()
        .flat_map(|domain| domain.cpus().iter().copied())
        .collect();
    cpus.sort_unstable();
    let note = format!(
        "the last {} of {usable} usable L3 domains reserved ({setting}) for {RESERVED_TENANTS}: cpus {}",
        requested.domains,
        format_cpu_list(&cpus)
    );
    Split {
        generation: domains,
        reserved: Some(Arc::new(ReservedPlan {
            cpus: cpus.into_boxed_slice(),
            threads: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
        })),
        note,
    }
}

/// The one line the startup log gets, whatever happened.
fn summary_line(
    plan: &PinPlan,
    usable_domains: usize,
    usable_cpus: usize,
    notes: &Notes,
) -> String {
    format!(
        "generation pool L3 pinning enabled ({PIN_ENV}): {} workers over {} of {usable_domains} usable L3 domains, {} of {usable_cpus} CPUs{};{} {}",
        plan.workers,
        plan.domains.len(),
        plan.cpus(),
        notes.mask,
        plan.describe_blocks(),
        notes.reservation,
    )
}

/// The two free-text clauses [`summary_line`] cannot derive from the plan.
struct Notes {
    /// Whether an operator confinement narrowed what the machine offered.
    mask: String,
    /// What was reserved, or why nothing was.
    reservation: String,
}

impl GenerationAffinity {
    /// The process's affinity decision, made on first call and reused after.
    ///
    /// `workers` is the generation pool's thread count; it is only consulted on
    /// the first call, since by the second the pool it describes already exists.
    ///
    /// Reads nothing and calls nothing when the switch is off: the whole
    /// feature has to be inert in the control arm of an A/B.
    pub fn resolve(workers: usize) -> &'static Self {
        RESOLVED.get_or_init(|| Self::compute(workers))
    }

    fn compute(workers: usize) -> Self {
        let Ok(requested) = env::var(PIN_ENV) else {
            return Self::skipped(format!("disabled ({PIN_ENV} unset)"));
        };
        if !TRUTHY.contains(&requested.trim().to_ascii_lowercase().as_str()) {
            return Self::skipped(format!(
                "disabled ({PIN_ENV}={requested:?} is not one of {TRUTHY:?})"
            ));
        }

        if workers == 0 {
            return Self::skipped(format!(
                "skipped ({PIN_ENV} set, but the pool has no workers)"
            ));
        }
        let Some(topology) = L3Domains::read() else {
            return Self::skipped(format!(
                "skipped ({PIN_ENV} set, but no level-3 unified cache domains could be read from /sys/devices/system/cpu)"
            ));
        };
        // Read on a thread nothing has pinned yet -- see `RESOLVED`. Rayon's
        // and tokio's workers inherit their creator's mask, so this is also the
        // mask they all start with.
        let allowed = match current_thread_cpus() {
            Ok(allowed) => allowed,
            Err(error) => {
                return Self::skipped(format!(
                    "skipped ({PIN_ENV} set, but this process's own affinity mask could not be read: {error})"
                ));
            }
        };

        match usable_domains(&topology, &allowed, workers) {
            Err(reason) => Self::skipped(format!("skipped ({PIN_ENV} set, but {reason})")),
            Ok((domains, mask)) => {
                let usable_domains = domains.len();
                let usable_cpus = domains.iter().map(|domain| domain.cpus().len()).sum();
                let split = split_reserved(domains, &requested_reserve(), workers);
                let plan = PinPlan {
                    domains: split.generation.into_boxed_slice(),
                    workers,
                    failures: AtomicUsize::new(0),
                };
                let notes = Notes {
                    mask,
                    reservation: split.note,
                };
                let summary = summary_line(&plan, usable_domains, usable_cpus, &notes);
                Self {
                    plan: Some(Arc::new(plan)),
                    reserved: split.reserved,
                    summary,
                }
            }
        }
    }

    fn skipped(summary: String) -> Self {
        Self {
            plan: None,
            reserved: None,
            summary: format!("generation pool L3 pinning {summary}"),
        }
    }

    /// One line describing what was found and what will happen, for the startup
    /// log. Worth logging in both arms: it is how a run is identified after the
    /// fact as pinned or not, and how the machine was carved up.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Installs the start handler that pins each generation worker as it comes
    /// up.
    ///
    /// A no-op builder passthrough when pinning is off, so the pool is built
    /// exactly as it was before.
    #[must_use]
    pub fn apply(&self, builder: ThreadPoolBuilder) -> ThreadPoolBuilder {
        let Some(plan) = self.plan.clone() else {
            return builder;
        };
        builder.start_handler(move |worker| {
            let domain = plan.domain_for_worker(worker);
            if let Err(error) = pin_current_thread(domain.cpus()) {
                // Failure leaves this thread's existing mask untouched, so it
                // simply keeps the process-wide one and behaves like the
                // unpinned build. The rest of the pool stays pinned: an
                // all-or-nothing rollback would mean unpinning workers that
                // already started, which is the same outcome as this for them
                // and worse for everyone else.
                //
                // Counted as well as logged. `log` is a no-op until a logger is
                // installed, and the pregen bench -- the harness this switch is
                // measured with -- installs none, so a warning alone would let
                // a run print the plan and then measure a pool where nothing
                // was pinned.
                plan.failures.fetch_add(1, Ordering::Relaxed);
                log::warn!(
                    "generation worker {worker} could not be pinned to L3 domain cpus {domain}: {error}; it keeps the default affinity mask"
                );
            }
        })
    }

    /// Confines a rayon pool -- the encoding pool -- to the reserved domains.
    ///
    /// A no-op passthrough when nothing was reserved.
    #[must_use]
    pub fn apply_reserved(
        &self,
        builder: ThreadPoolBuilder,
        tenant: &'static str,
    ) -> ThreadPoolBuilder {
        let Some(reserved) = self.reserved.clone() else {
            return builder;
        };
        builder.start_handler(move |_| reserved.pin_calling_thread(tenant))
    }

    /// Confines a tokio runtime to the reserved domains.
    ///
    /// Takes the builder by `&mut` because that is tokio's own shape, and
    /// covers the blocking pool as well as the workers: `spawn_blocking`
    /// threads are exactly the kind of thread the reservation exists to keep
    /// off generation's CPUs, and they are started lazily, long after the
    /// summary line was printed.
    ///
    /// A no-op when nothing was reserved -- in particular no handler is
    /// installed, so the control arm's runtimes are built exactly as before.
    pub fn apply_reserved_runtime(&self, builder: &mut RuntimeBuilder, tenant: &'static str) {
        let Some(reserved) = self.reserved.clone() else {
            return;
        };
        builder.on_thread_start(move || reserved.pin_calling_thread(tenant));
    }

    /// Confines the calling thread to the reserved domains, for the one thread
    /// [`Self::apply_reserved_runtime`] cannot reach.
    ///
    /// `on_thread_start` fires only for threads the runtime *spawns*. The
    /// thread that enters a runtime through `block_on` keeps the process-wide
    /// mask, and in the pregen bench that thread drives the whole
    /// pregeneration: measured at 270 ms of CPU over a 1.44 s repetition, about
    /// a fifth of a core, free to land on any of generation's CPUs while the
    /// summary line claimed the main runtime was confined.
    ///
    /// Call it only once every pool the thread creates has been built: rayon
    /// and tokio workers inherit their creator's mask, and a worker whose own
    /// pin fails must fall back to the whole machine, never to the domains it
    /// was meant to stay off.
    ///
    /// A no-op when nothing was reserved.
    pub fn pin_calling_thread_to_reserved(&self, tenant: &'static str) {
        if let Some(reserved) = self.reserved.as_ref() {
            reserved.pin_calling_thread(tenant);
        }
    }

    /// The counters as they stand, for a caller about to build a set of pools.
    #[must_use]
    pub fn checkpoint(&self) -> PinCheckpoint {
        PinCheckpoint {
            generation_failures: self
                .plan
                .as_ref()
                .map_or(0, |plan| plan.failures.load(Ordering::Relaxed)),
            reserved_threads: self
                .reserved
                .as_ref()
                .map_or(0, |reserved| reserved.threads.load(Ordering::Relaxed)),
            reserved_failures: self
                .reserved
                .as_ref()
                .map_or(0, |reserved| reserved.failures.load(Ordering::Relaxed)),
        }
    }

    /// Checks, once the pool exists, that the plan in [`Self::summary`] is what
    /// actually happened; returns a line to surface if it is not.
    ///
    /// `build()` returns as soon as the threads are spawned, which can be
    /// before their start handlers have run, so the count is only trustworthy
    /// behind a barrier. A broadcast is the cheapest one available: a worker
    /// cannot take a broadcast job until it has entered its work loop, which is
    /// after its start handler returned, and it also gives the happens-before
    /// edge that makes the relaxed increments visible here.
    #[must_use]
    pub fn verify(&self, pool: &ThreadPool, since: PinCheckpoint) -> Option<String> {
        let plan = self.plan.as_ref()?;
        let _ = pool.broadcast(|_| ());
        let failures = plan
            .failures
            .load(Ordering::Relaxed)
            .saturating_sub(since.generation_failures);
        (failures > 0).then(|| {
            format!(
                "generation pool L3 pinning did not take effect for {failures} of {} workers, which keep the default affinity mask; this run is not a clean pinned arm",
                plan.workers
            )
        })
    }

    /// The same check for the reserved side, or `None` if it is clean so far.
    ///
    /// `encoding_pool` is the reserved rayon pool, taken to barrier on: rayon
    /// sets a worker's primed latch *before* running its start handler and
    /// `ThreadPoolBuilder::build` does not even wait for that, so reading the
    /// counters straight after `build()` reads zeros -- checked by deleting
    /// this line, which fails the two tests below on every run. A run where
    /// every reserved pin failed would then print the full carve-up and nothing
    /// else, since `log::warn!` is invisible in the bench.
    ///
    /// Still only a snapshot for the two runtimes: their `on_thread_start`
    /// handlers run asynchronously on the new threads, and tokio starts
    /// blocking-pool threads on demand long afterwards. Read a clean result as
    /// "the encoding pool is clean and nothing else has failed yet". The
    /// per-thread warning is the complete record; this is the summary an
    /// operator will actually notice.
    #[must_use]
    pub fn verify_reserved(
        &self,
        encoding_pool: &ThreadPool,
        since: PinCheckpoint,
    ) -> Option<String> {
        let reserved = self.reserved.as_ref()?;
        let _ = encoding_pool.broadcast(|_| ());
        let failures = reserved
            .failures
            .load(Ordering::Relaxed)
            .saturating_sub(since.reserved_failures);
        let threads = reserved
            .threads
            .load(Ordering::Relaxed)
            .saturating_sub(since.reserved_threads);
        (failures > 0).then(|| {
            format!(
                "{failures} of the {threads} threads so far asked to run on the reserved L3 domains (cpus {}) could not be pinned there and keep the default affinity mask; this run is not a clean reserved arm",
                format_cpu_list(&reserved.cpus)
            )
        })
    }
}

/// [`RESERVE_ENV`] as a count, defaulting to 0.
///
/// Garbage is 0 rather than a startup failure, matching every other decision
/// here: this is a tuning knob for an experiment, and refusing to boot over a
/// typo in it would be out of proportion. The summary quotes back what was
/// actually set, so a mistyped sweep step shows up as an arm that did nothing
/// and says why, rather than as an arm that silently did something else.
fn requested_reserve() -> Requested {
    interpret_reserve(env::var(RESERVE_ENV).ok().as_deref())
}

/// [`requested_reserve`] with the environment read out, so the three readings
/// can be tested without mutating this process's environment.
fn interpret_reserve(raw: Option<&str>) -> Requested {
    let Some(raw) = raw else {
        return Requested {
            domains: 0,
            setting: format!("{RESERVE_ENV} is unset"),
        };
    };
    match raw.trim().parse::<usize>() {
        Ok(domains) => Requested {
            domains,
            setting: format!("{RESERVE_ENV}={domains}"),
        },
        Err(_) => Requested {
            domains: 0,
            setting: format!("{RESERVE_ENV}={raw:?} is not a domain count"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arc, AtomicUsize, GenerationAffinity, Notes, PinCheckpoint, PinPlan, Requested,
        ReservedPlan, ThreadPoolBuilder, domain_index_for_worker, interpret_reserve, split_reserved,
        summary_line, usable_domains,
    };
    use std::thread;
    use steel_utils::cpu::affinity::current_thread_cpus;
    use steel_utils::cpu::topology::{CacheDomain, CacheIndex, L3Domains, format_cpu_list};

    /// The topology described by `lists`, parsed rather than read from the
    /// machine running the test.
    fn topology(lists: &[&str]) -> L3Domains {
        L3Domains::from_cache_entries(lists.iter().map(|list| CacheIndex {
            level: "3",
            kind: "Unified",
            shared_cpu_list: list,
        }))
        .expect("the test's cpu lists should parse")
    }

    /// The domains described by `lists`, as `usable_domains` would hand them on.
    fn domains(lists: &[&str]) -> Vec<CacheDomain> {
        topology(lists).domains().to_vec()
    }

    /// This box's eight L3 domains, in sysfs's own syntax.
    const ZEN5: [&str; 8] = [
        "0-7,64-71",
        "8-15,72-79",
        "16-23,80-87",
        "24-31,88-95",
        "32-39,96-103",
        "40-47,104-111",
        "48-55,112-119",
        "56-63,120-127",
    ];

    /// A plan over the domains described by `lists`.
    fn plan(workers: usize, lists: &[&str]) -> PinPlan {
        PinPlan {
            domains: domains(lists).into_boxed_slice(),
            workers,
            failures: AtomicUsize::new(0),
        }
    }

    /// A reserved mask over `cpus`, standing in for one `split_reserved` built.
    fn reserved(cpus: &[usize]) -> Arc<ReservedPlan> {
        Arc::new(ReservedPlan {
            cpus: cpus.to_vec().into_boxed_slice(),
            threads: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
        })
    }

    /// `RESERVE_ENV` set to a plain count, as an operator's sweep would.
    fn asked(domains: usize) -> Requested {
        Requested {
            domains,
            setting: format!("STEEL_GENERATION_RESERVE_DOMAINS={domains}"),
        }
    }

    fn notes(reservation: &str) -> Notes {
        Notes {
            mask: String::new(),
            reservation: reservation.to_owned(),
        }
    }

    /// The CPU ids a domain was narrowed to, per surviving domain.
    fn narrowed(domains: &[CacheDomain]) -> Vec<String> {
        domains.iter().map(CacheDomain::to_string).collect()
    }

    fn blocks(workers: usize, domains: usize) -> Vec<usize> {
        (0..workers)
            .map(|worker| domain_index_for_worker(worker, workers, domains))
            .collect()
    }

    #[test]
    fn assigns_contiguous_blocks() {
        let assignment = blocks(120, 8);
        assert_eq!(assignment[0..15], [0; 15]);
        assert_eq!(assignment[15..30], [1; 15]);
        assert_eq!(assignment[105..120], [7; 15]);
    }

    #[test]
    fn every_domain_is_used_and_never_exceeded() {
        for (workers, domains) in [(120, 8), (8, 8), (9, 8), (121, 8), (127, 8), (16, 2)] {
            let assignment = blocks(workers, domains);
            assert_eq!(*assignment.iter().max().unwrap_or(&0), domains - 1);
            assert_eq!(assignment.iter().min().copied(), Some(0));
            // Contiguity: the domain index never decreases and never skips.
            for pair in assignment.windows(2) {
                assert!(pair[1] == pair[0] || pair[1] == pair[0] + 1);
            }
        }
    }

    #[test]
    fn block_sizes_stay_balanced() {
        let assignment = blocks(121, 8);
        let mut counts = vec![0usize; 8];
        for domain in assignment {
            counts[domain] += 1;
        }
        assert_eq!(counts.iter().sum::<usize>(), 121);
        let (min, max) = (
            counts.iter().min().copied().unwrap_or(0),
            counts.iter().max().copied().unwrap_or(0),
        );
        assert!(max - min <= 1, "unbalanced blocks: {counts:?}");
    }

    #[test]
    fn an_unrestricted_process_gets_the_whole_machine_and_no_note() {
        let all: Vec<usize> = (0..128).collect();
        let (domains, note) =
            usable_domains(&topology(&ZEN5), &all, 120).expect("120 workers fit on 128 CPUs");

        assert_eq!(narrowed(&domains), ZEN5);
        assert_eq!(note, "");
    }

    #[test]
    fn a_taskset_confined_process_is_pinned_only_inside_its_mask() {
        // `taskset -c 0-15` on this box. sysfs still reports all eight domains
        // and 128 CPUs; without the intersection the plan would hand workers
        // cpus 16-127, which the kernel would grant -- silently escaping the
        // operator's restriction and making the pinned arm run on eight times
        // the hardware of the control arm.
        let allowed: Vec<usize> = (0..16).collect();
        let (domains, note) =
            usable_domains(&topology(&ZEN5), &allowed, 8).expect("two domains and 16 CPUs remain");

        assert_eq!(narrowed(&domains), ["0-7", "8-15"]);
        for domain in &domains {
            assert!(
                domain.cpus().iter().all(|cpu| allowed.contains(cpu)),
                "worker mask escaped the process's own affinity mask: {domain}"
            );
        }
        assert!(note.contains("128 CPUs in 8 domains"), "{note}");
    }

    #[test]
    fn a_cpuset_that_leaves_one_domain_is_skipped_not_half_pinned() {
        // `--cpuset-cpus=0-7`: sysfs claims 8 domains and 128 CPUs, so a guard
        // that only compared the machine's CPU count against the worker count
        // would pass and then pin the one overlapping block while every other
        // worker failed with EINVAL.
        let allowed: Vec<usize> = (0..8).collect();
        let reason = usable_domains(&topology(&ZEN5), &allowed, 7)
            .expect_err("a single reachable domain is not worth pinning");

        assert!(
            reason.contains("reach only 1 of the machine's 8"),
            "{reason}"
        );
    }

    #[test]
    fn more_workers_than_usable_cpus_is_skipped() {
        // The container case the machine-wide CPU count cannot see: 32 allowed
        // CPUs, but sysfs says 128 and the pool was sized elsewhere.
        let allowed: Vec<usize> = (0..32).collect();
        let reason = usable_domains(&topology(&ZEN5), &allowed, 64)
            .expect_err("64 workers do not fit on 32 usable CPUs");

        assert!(reason.contains("only 32 CPUs"), "{reason}");
        assert!(reason.contains("64 workers"), "{reason}");
    }

    #[test]
    fn a_single_domain_machine_is_skipped_before_anything_else() {
        let reason = usable_domains(&topology(&["0-15"]), &(0..16).collect::<Vec<_>>(), 8)
            .expect_err("one domain is not worth pinning");

        assert!(reason.contains("single L3 domain of 0-15"), "{reason}");
    }

    #[test]
    fn an_unset_or_mistyped_setting_reserves_nothing_and_says_which() {
        // Both mean zero domains, but an operator sweeping this variable has to
        // be able to tell "I did not set it" from "I set it to something the
        // server could not read" -- reporting either as `=0` would claim they
        // typed a 0.
        let unset = interpret_reserve(None);
        assert_eq!(unset.domains, 0);
        assert_eq!(unset.setting, "STEEL_GENERATION_RESERVE_DOMAINS is unset");

        let typo = interpret_reserve(Some("one"));
        assert_eq!(typo.domains, 0);
        assert_eq!(
            typo.setting,
            "STEEL_GENERATION_RESERVE_DOMAINS=\"one\" is not a domain count"
        );

        let zero = interpret_reserve(Some("0"));
        assert_eq!(zero.domains, 0);
        assert_eq!(zero.setting, "STEEL_GENERATION_RESERVE_DOMAINS=0");

        let set = interpret_reserve(Some(" 2\n"));
        assert_eq!(set.domains, 2);
        assert_eq!(set.setting, "STEEL_GENERATION_RESERVE_DOMAINS=2");
    }

    #[test]
    fn reserving_nothing_leaves_every_domain_to_generation() {
        let split = split_reserved(domains(&ZEN5), &asked(0), 120);

        assert_eq!(narrowed(&split.generation), ZEN5);
        assert!(split.reserved.is_none());
        assert!(
            split.note.contains("no L3 domains reserved"),
            "{}",
            split.note
        );
    }

    #[test]
    fn reserving_one_domain_takes_the_last_one() {
        let split = split_reserved(domains(&ZEN5), &asked(1), 112);

        assert_eq!(narrowed(&split.generation), ZEN5[..7]);
        // The mask the runtimes and the encoding pool get: the union of the
        // reserved domains, merged across the join and sorted, because that is
        // what `sched_setaffinity` is handed.
        let reserved = split.reserved.expect("one domain was reserved");
        assert_eq!(format_cpu_list(&reserved.cpus), "56-63,120-127");
        assert!(split.note.contains("the last 1 of 8"), "{}", split.note);
    }

    #[test]
    fn reserving_all_but_one_domain_leaves_generation_the_first() {
        // N = D - 1: the extreme the sweep ends at, where generation has one
        // domain and everything else has seven.
        let split = split_reserved(domains(&ZEN5), &asked(7), 16);

        assert_eq!(narrowed(&split.generation), ["0-7,64-71"]);
        let reserved = split.reserved.expect("seven domains were reserved");
        assert_eq!(format_cpu_list(&reserved.cpus), "8-63,72-127");
    }

    #[test]
    fn reserving_every_domain_is_refused_not_obeyed() {
        // N >= D would leave generation nowhere to run. Falling back to the
        // unreserved plan keeps the server up and says so.
        for requested in [8, 9, 100] {
            let split = split_reserved(domains(&ZEN5), &asked(requested), 120);

            assert_eq!(narrowed(&split.generation), ZEN5);
            assert!(split.reserved.is_none());
            assert!(
                split.note.contains(&format!(
                    "STEEL_GENERATION_RESERVE_DOMAINS={requested} is not fewer than the 8"
                )),
                "{}",
                split.note
            );
        }
    }

    #[test]
    fn a_single_domain_is_never_reserved_away_from_generation() {
        // D == 1 cannot reach `split_reserved` through `usable_domains`, which
        // rejects it first, but the arithmetic must not depend on that.
        let split = split_reserved(domains(&["0-15"]), &asked(1), 8);

        assert_eq!(narrowed(&split.generation), ["0-15"]);
        assert!(split.reserved.is_none());
    }

    #[test]
    fn a_reservation_that_would_oversubscribe_generation_is_refused() {
        // This box's *default* pool is 120 workers, and reserving one of eight
        // domains leaves 112 CPUs -- so the default configuration refuses the
        // reservation rather than squeezing into it, because squeezing in would
        // put back exactly the preemption the reservation exists to remove. A
        // sweep has to bring `--gen-threads` down with the reservation, which
        // is why this refuses instead of shrinking the pool itself.
        let split = split_reserved(domains(&ZEN5), &asked(1), 120);
        assert_eq!(narrowed(&split.generation), ZEN5);
        assert!(split.reserved.is_none());
        assert!(
            split.note.contains("would leave 112 CPUs across 7"),
            "{}",
            split.note
        );
        assert!(
            split.note.contains("120 generation workers"),
            "{}",
            split.note
        );

        // Exactly one worker per CPU is the tightest fit accepted.
        let fits = split_reserved(domains(&ZEN5), &asked(1), 112);
        assert!(fits.reserved.is_some());
    }

    #[test]
    fn workers_not_divisible_by_the_remaining_domains_still_spread_evenly() {
        // A reservation leaves 7 domains, which divides no round worker count.
        // 110 workers must come out 15 or 16 per domain with the two short
        // blocks spread through the range, not as six full domains and a
        // starved one at the end.
        let split = split_reserved(domains(&ZEN5), &asked(1), 110);
        assert_eq!(split.generation.len(), 7);
        let mut counts = vec![0usize; split.generation.len()];
        for domain in blocks(110, split.generation.len()) {
            counts[domain] += 1;
        }

        assert_eq!(counts.iter().sum::<usize>(), 110);
        assert_eq!(counts, [16, 16, 16, 15, 16, 16, 15]);
    }

    #[test]
    fn summary_names_every_block_and_its_cpus() {
        let plan = plan(120, &ZEN5);
        let summary = summary_line(&plan, 8, 128, &notes("no L3 domains reserved: ..."));

        assert!(
            summary.contains("120 workers over 8 of 8 usable L3 domains, 128 of 128 CPUs"),
            "{summary}"
        );
        assert!(
            summary.contains("workers 0-14 -> cpus 0-7,64-71"),
            "{summary}"
        );
        assert!(
            summary.contains("workers 15-29 -> cpus 8-15,72-79"),
            "{summary}"
        );
        assert!(
            summary.contains("workers 105-119 -> cpus 56-63,120-127"),
            "{summary}"
        );
        assert_eq!(summary.matches("-> cpus").count(), 8, "{summary}");
    }

    #[test]
    fn summary_states_the_whole_carve_up_in_one_line() {
        // What an operator has to be able to read off a single line: how many
        // domains generation got out of how many, where its workers went, which
        // domains were reserved, and what runs on them.
        let split = split_reserved(domains(&ZEN5), &asked(1), 112);
        let plan = PinPlan {
            domains: split.generation.into_boxed_slice(),
            workers: 112,
            failures: AtomicUsize::new(0),
        };
        let summary = summary_line(&plan, 8, 128, &notes(&split.note));

        assert_eq!(summary.lines().count(), 1, "{summary}");
        assert!(
            summary.contains("112 workers over 7 of 8 usable L3 domains, 112 of 128 CPUs"),
            "{summary}"
        );
        assert!(
            summary.contains("workers 0-15 -> cpus 0-7,64-71"),
            "{summary}"
        );
        assert!(
            summary.contains("workers 96-111 -> cpus 48-55,112-119"),
            "{summary}"
        );
        assert_eq!(summary.matches("-> cpus").count(), 7, "{summary}");
        assert!(
            summary.contains(
                "the last 1 of 8 usable L3 domains reserved (STEEL_GENERATION_RESERVE_DOMAINS=1) for the chunk runtime, the main runtime and the encoding pool: cpus 56-63,120-127"
            ),
            "{summary}"
        );
    }

    #[test]
    fn summary_names_a_lone_worker_in_the_singular() {
        let plan = plan(2, &["0-3", "4-7"]);
        let summary = summary_line(&plan, 2, 8, &notes(""));
        assert!(summary.contains("worker 0 -> cpus 0-3"), "{summary}");
        assert!(summary.contains("worker 1 -> cpus 4-7"), "{summary}");
    }

    #[test]
    fn a_pool_whose_workers_all_failed_to_pin_is_reported_not_just_logged() {
        // CPU ids above a cpu_set_t's 1024-bit capacity, so every worker's pin
        // is refused without any thread's affinity actually changing.
        // `log::warn!` is invisible in the pregen bench, which installs no
        // logger, so a run there could otherwise print the plan and then
        // measure a completely unpinned pool.
        let affinity = GenerationAffinity {
            plan: Some(Arc::new(plan(2, &["2000-2001", "3000-3001"]))),
            reserved: None,
            summary: String::new(),
        };
        let pool = affinity
            .apply(ThreadPoolBuilder::new().num_threads(2))
            .build()
            .expect("a two-thread pool should build even when no worker can pin");

        let warning = affinity
            .verify(&pool, PinCheckpoint::PROCESS_START)
            .expect("two failed pins should be reported");
        assert!(warning.contains("2 of 2 workers"), "{warning}");
        assert_eq!(
            affinity.verify_reserved(&pool, PinCheckpoint::PROCESS_START),
            None
        );
    }

    #[test]
    fn a_reserved_pool_that_could_not_pin_is_reported_not_just_logged() {
        let affinity = GenerationAffinity {
            plan: None,
            reserved: Some(reserved(&[2000, 2001])),
            summary: String::new(),
        };
        let pool = affinity
            .apply_reserved(ThreadPoolBuilder::new().num_threads(2), "encoding pool")
            .build()
            .expect("a two-thread pool should build even when no worker can pin");

        // No manual barrier: `build` only guarantees the threads were spawned,
        // and rayon sets a worker's primed latch before running its start
        // handler, so the counters are zero here until something waits for the
        // handlers. `verify_reserved` owns that wait -- when it did not, this
        // test had to broadcast by hand and the two production call sites,
        // which could not, reported every reserved arm as clean.
        let warning = affinity
            .verify_reserved(&pool, PinCheckpoint::PROCESS_START)
            .expect("two failed pins should be reported");
        assert!(warning.contains("2 of the 2 threads"), "{warning}");
    }

    #[test]
    fn a_pool_that_pinned_cleanly_reports_nothing() {
        // Every thread asks for exactly the mask it already has, so the pins
        // succeed and nothing moves.
        let allowed = current_thread_cpus().expect("this thread has an affinity mask");
        let list = format_cpu_list(&allowed);
        let affinity = GenerationAffinity {
            plan: Some(Arc::new(plan(2, &[&list]))),
            reserved: Some(reserved(&allowed)),
            summary: String::new(),
        };
        let pool = affinity
            .apply(affinity.apply_reserved(ThreadPoolBuilder::new().num_threads(2), "test"))
            .build()
            .expect("a two-thread pool should build");

        assert_eq!(affinity.verify(&pool, PinCheckpoint::PROCESS_START), None);
        assert_eq!(
            affinity.verify_reserved(&pool, PinCheckpoint::PROCESS_START),
            None
        );
    }

    #[test]
    fn nothing_is_installed_when_there_is_no_plan() {
        // The control arm: both builders must come back untouched, so an
        // unpinned run is built exactly as it was before this module existed.
        let affinity = GenerationAffinity {
            plan: None,
            reserved: None,
            summary: String::new(),
        };
        let pool = affinity
            .apply(affinity.apply_reserved(ThreadPoolBuilder::new().num_threads(1), "test"))
            .build()
            .expect("a one-thread pool should build");
        let before = current_thread_cpus().expect("this thread has an affinity mask");
        let inside = pool.install(|| current_thread_cpus().expect("workers have a mask"));

        assert_eq!(inside, before);
        assert_eq!(affinity.verify(&pool, PinCheckpoint::PROCESS_START), None);
        assert_eq!(
            affinity.verify_reserved(&pool, PinCheckpoint::PROCESS_START),
            None
        );
    }

    /// A pool of `threads` generation workers, built the way the server builds
    /// the generation pool.
    fn generation_pool(affinity: &GenerationAffinity, threads: usize) -> rayon::ThreadPool {
        affinity
            .apply(ThreadPoolBuilder::new().num_threads(threads))
            .build()
            .expect("a generation pool should build even when no worker can pin")
    }

    /// The reserved-side pool, built the way the server builds the encoding
    /// pool. Separate from [`generation_pool`] because the two handlers are:
    /// `start_handler` replaces rather than chains, so one builder cannot carry
    /// both.
    fn encoding_pool(affinity: &GenerationAffinity, threads: usize) -> rayon::ThreadPool {
        affinity
            .apply_reserved(ThreadPoolBuilder::new().num_threads(threads), "encoding pool")
            .build()
            .expect("an encoding pool should build even when no worker can pin")
    }

    /// An affinity whose every pin fails: CPU ids above a `cpu_set_t`'s
    /// 1024-bit capacity, refused before any thread's mask changes.
    fn unpinnable() -> GenerationAffinity {
        GenerationAffinity {
            plan: Some(Arc::new(plan(2, &["2000-2001", "3000-3001"]))),
            reserved: Some(reserved(&[2000, 2001])),
            summary: String::new(),
        }
    }

    #[test]
    fn a_second_set_of_pools_is_reported_on_its_own() {
        // The bench's repetitions: one process-wide decision, a fresh
        // generation pool and encoding pool per repetition. Reported as running
        // totals, the second repetition would claim "4 of 2 workers" -- more
        // failures than the pool has workers.
        let affinity = unpinnable();

        let first_generation = affinity
            .verify(&generation_pool(&affinity, 2), PinCheckpoint::PROCESS_START)
            .expect("the first pool's two failures should be reported");
        assert!(
            first_generation.contains("2 of 2 workers"),
            "{first_generation}"
        );
        let first_reserved = affinity
            .verify_reserved(&encoding_pool(&affinity, 2), PinCheckpoint::PROCESS_START)
            .expect("the first pool's two reserved failures should be reported");
        assert!(
            first_reserved.contains("2 of the 2 threads"),
            "{first_reserved}"
        );

        let checkpoint = affinity.checkpoint();
        let second_generation = affinity
            .verify(&generation_pool(&affinity, 2), checkpoint)
            .expect("the second pool's two failures should be reported");
        assert!(
            second_generation.contains("2 of 2 workers"),
            "{second_generation}"
        );
        let second_reserved = affinity
            .verify_reserved(&encoding_pool(&affinity, 2), checkpoint)
            .expect("the second pool's two reserved failures should be reported");
        assert!(
            second_reserved.contains("2 of the 2 threads"),
            "{second_reserved}"
        );
    }

    #[test]
    fn a_clean_set_after_a_dirty_one_reports_nothing() {
        // The other half of the checkpoint: cumulative counters would report a
        // clean repetition as dirty forever after the first failure.
        let dirty = unpinnable();
        assert!(
            dirty
                .verify(&generation_pool(&dirty, 2), PinCheckpoint::PROCESS_START)
                .is_some()
        );
        assert!(
            dirty
                .verify_reserved(&encoding_pool(&dirty, 2), PinCheckpoint::PROCESS_START)
                .is_some()
        );
        let checkpoint = dirty.checkpoint();

        // A plan every thread can satisfy: each asks for exactly the mask it
        // already has, so nothing moves and nothing fails.
        let allowed = current_thread_cpus().expect("this thread has an affinity mask");
        let list = format_cpu_list(&allowed);
        let workable = GenerationAffinity {
            plan: Some(Arc::new(plan(2, &[&list]))),
            reserved: Some(reserved(&allowed)),
            summary: String::new(),
        };
        let generation = generation_pool(&workable, 2);
        let encoding = encoding_pool(&workable, 2);
        // Barriers, since the clean pools' counters belong to `workable` while
        // the reading below is taken through `dirty`.
        let _ = generation.broadcast(|_| ());
        let _ = encoding.broadcast(|_| ());

        assert_eq!(dirty.verify(&generation, checkpoint), None);
        assert_eq!(dirty.verify_reserved(&encoding, checkpoint), None);
    }

    #[test]
    fn the_calling_thread_can_be_confined_to_the_reserved_set() {
        // The thread that calls `block_on` is the one tokio's
        // `on_thread_start` never sees, and in the bench it drives the whole
        // pregeneration. Run in a thread of its own, which then exits: this
        // really does change a mask, and the test harness's threads must not
        // keep it.
        let allowed = current_thread_cpus().expect("this thread has an affinity mask");
        if allowed.len() < 2 {
            // Nothing to narrow to, so the assertion could not discriminate.
            return;
        }
        let target = allowed[..1].to_vec();
        let affinity = GenerationAffinity {
            plan: None,
            reserved: Some(reserved(&target)),
            summary: String::new(),
        };

        let seen = thread::scope(|scope| {
            scope
                .spawn(|| {
                    affinity.pin_calling_thread_to_reserved("main runtime driver");
                    current_thread_cpus().expect("the mask should still be readable")
                })
                .join()
                .expect("the pinned thread should not panic")
        });

        assert_eq!(seen, target);
    }

    #[test]
    fn the_calling_thread_is_untouched_when_nothing_was_reserved() {
        // The control arm: no reservation, no syscall, no mask change.
        let affinity = GenerationAffinity {
            plan: None,
            reserved: None,
            summary: String::new(),
        };
        let before = current_thread_cpus().expect("this thread has an affinity mask");
        affinity.pin_calling_thread_to_reserved("main runtime driver");

        assert_eq!(
            current_thread_cpus().expect("the mask should still be readable"),
            before
        );
    }

    #[test]
    fn more_domains_than_workers_still_stays_in_range() {
        let assignment = blocks(3, 8);
        assert!(assignment.iter().all(|&domain| domain < 8));
        assert_eq!(assignment[0], 0);
    }
}
