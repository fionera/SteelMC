//! Optional L3-domain pinning for the chunk generation pool.
//!
//! Generation is bound by cache, not by dispatch: over a 601x601 pregeneration
//! 89.2% of fills came from the local L2 and 7.0% from the local CCX's L3, with
//! only 3.8% from DRAM and none from another CCX. The same run showed ~40,000
//! CPU migrations a second, and a migration across an L3 boundary discards the
//! warm L2 *and* the 32 MiB L3 behind it, including that CCX's copy of the
//! ~2.9 MB climate R-tree biome lookup streams 1,536 times per chunk.
//!
//! Confining each worker to one L3 domain is the cheapest thing that can be
//! done about that, and it is only step one: once workers are stable within a
//! domain, generation work can be routed so that spatially adjacent chunks land
//! on the same domain.
//!
//! This pool only. The tokio runtimes and the encoding pool are left alone so
//! that an A/B run measures one variable.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::env;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use steel_utils::cpu::affinity::{current_thread_cpus, pin_current_thread};
use steel_utils::cpu::topology::{CacheDomain, L3Domains};

/// Runtime switch for generation-pool pinning, default off.
///
/// Runtime rather than a cargo feature because the intended use is A/B-ing two
/// runs of one binary, where a rebuild between arms would change more than the
/// pinning.
const PIN_ENV: &str = "STEEL_GENERATION_PIN_L3";

/// Values of [`PIN_ENV`] that turn pinning on.
const TRUTHY: [&str; 4] = ["1", "true", "yes", "on"];

/// What to do with the generation pool's worker threads, decided once at
/// startup.
pub struct GenerationAffinity {
    /// `None` when pinning is off or was skipped.
    plan: Option<Arc<PinPlan>>,
    summary: String,
}

/// Which L3 domain each worker index belongs to.
struct PinPlan {
    /// Domains already narrowed to the CPUs this process may use.
    domains: Box<[CacheDomain]>,
    workers: usize,
    /// Workers whose `sched_setaffinity` call failed, counted by the start
    /// handler so the outcome can be reported rather than only logged.
    failures: AtomicUsize,
}

impl PinPlan {
    fn domain_for_worker(&self, worker: usize) -> &CacheDomain {
        let index = domain_index_for_worker(worker, self.workers, self.domains.len());
        &self.domains[index]
    }

    /// Renders the worker-block assignment, e.g.
    /// `workers 0-14 -> cpus 0-7,64-71`.
    fn describe(&self, total_cpus: usize, mask_note: &str) -> String {
        let mut summary = format!(
            "generation pool L3 pinning enabled ({PIN_ENV}): {} workers over {} L3 domains, {total_cpus} CPUs{mask_note};",
            self.workers,
            self.domains.len()
        );
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
            let _ = write!(summary, " {workers} -> cpus {domain};");
            block_start = worker + 1;
        }
        summary.pop();
        summary
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
/// domain 16.
fn domain_index_for_worker(worker: usize, workers: usize, domains: usize) -> usize {
    debug_assert!(workers > 0 && domains > 0, "resolve rejects empty inputs");
    (worker * domains / workers.max(1)).min(domains.saturating_sub(1))
}

/// Narrows a machine's L3 domains to the ones `workers` threads can usefully be
/// pinned to, given the CPUs this process is `allowed` to run on.
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

impl GenerationAffinity {
    /// Decides how the generation pool's `workers` threads should be pinned.
    ///
    /// Reads nothing and calls nothing when the switch is off: the whole
    /// feature has to be inert in the control arm of an A/B.
    #[must_use]
    pub fn resolve(workers: usize) -> Self {
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
        // Read on the thread that will build the pool: rayon's workers inherit
        // their creator's mask, so this is the mask they start with.
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
            Ok((domains, mask_note)) => {
                let total_cpus = domains.iter().map(|domain| domain.cpus().len()).sum();
                let plan = PinPlan {
                    domains: domains.into_boxed_slice(),
                    workers,
                    failures: AtomicUsize::new(0),
                };
                let summary = plan.describe(total_cpus, &mask_note);
                Self {
                    plan: Some(Arc::new(plan)),
                    summary,
                }
            }
        }
    }

    fn skipped(summary: String) -> Self {
        Self {
            plan: None,
            summary: format!("generation pool L3 pinning {summary}"),
        }
    }

    /// One line describing what was found and what will happen, for the startup
    /// log. Worth logging in both arms: it is how a run is identified after the
    /// fact as pinned or not.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Installs the start handler that pins each worker as it comes up.
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
    pub fn verify(&self, pool: &ThreadPool) -> Option<String> {
        let plan = self.plan.as_ref()?;
        let _ = pool.broadcast(|_| ());
        let failures = plan.failures.load(Ordering::Relaxed);
        (failures > 0).then(|| {
            format!(
                "generation pool L3 pinning did not take effect for {failures} of {} workers, which keep the default affinity mask; this run is not a clean pinned arm",
                plan.workers
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arc, GenerationAffinity, PinPlan, ThreadPoolBuilder, domain_index_for_worker,
        usable_domains,
    };
    use std::sync::atomic::AtomicUsize;
    use steel_utils::cpu::affinity::current_thread_cpus;
    use steel_utils::cpu::topology::{CacheDomain, CacheIndex, L3Domains};

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
            domains: topology(lists).domains().to_vec().into_boxed_slice(),
            workers,
            failures: AtomicUsize::new(0),
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

        assert!(reason.contains("reach only 1 of the machine's 8"), "{reason}");
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
    fn summary_names_every_block_and_its_cpus() {
        let plan = plan(120, &ZEN5);
        let summary = plan.describe(128, "");

        assert!(
            summary.contains("120 workers over 8 L3 domains, 128 CPUs"),
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
    fn summary_names_a_lone_worker_in_the_singular() {
        let plan = plan(2, &["0-3", "4-7"]);
        let summary = plan.describe(8, "");
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
            summary: String::new(),
        };
        let pool = affinity
            .apply(ThreadPoolBuilder::new().num_threads(2))
            .build()
            .expect("a two-thread pool should build even when no worker can pin");

        let warning = affinity
            .verify(&pool)
            .expect("two failed pins should be reported");
        assert!(warning.contains("2 of 2 workers"), "{warning}");
    }

    #[test]
    fn a_pool_that_pinned_cleanly_reports_nothing() {
        // Every thread asks for exactly the mask it already has, so the pins
        // succeed and nothing moves.
        let allowed = current_thread_cpus().expect("this thread has an affinity mask");
        let list = allowed
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let affinity = GenerationAffinity {
            plan: Some(Arc::new(plan(2, &[&list]))),
            summary: String::new(),
        };
        let pool = affinity
            .apply(ThreadPoolBuilder::new().num_threads(2))
            .build()
            .expect("a two-thread pool should build");

        assert_eq!(affinity.verify(&pool), None);
    }

    #[test]
    fn more_domains_than_workers_still_stays_in_range() {
        let assignment = blocks(3, 8);
        assert!(assignment.iter().all(|&domain| domain < 8));
        assert_eq!(assignment[0], 0);
    }
}
