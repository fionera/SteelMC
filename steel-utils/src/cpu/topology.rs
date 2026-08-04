//! Level-3 cache sharing domains, read from `/sys/devices/system/cpu`.
//!
//! Everything here is best-effort by construction. Containers mount a partial
//! `/sys`, virtual machines may expose no cache topology at all, and non-Linux
//! targets have none of these files, so every entry point returns `Option` and
//! callers are expected to carry on unpinned rather than treat absence as an
//! error.

use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::Path;

/// Root of the per-CPU sysfs tree.
const CPU_SYSFS_ROOT: &str = "/sys/devices/system/cpu";

/// Largest CPU id accepted from a `shared_cpu_list`.
///
/// A range is expanded eagerly, so a corrupt or hostile `0-99999999999` would
/// otherwise turn into an allocation the size of the number. Linux's own
/// `CONFIG_NR_CPUS` ceiling is 8192, so nothing real is lost by refusing above
/// it.
const MAX_CPU_ID: usize = 8192;

/// One `index*` directory's three interesting files, already read to strings.
///
/// The reader hands these to the parser instead of paths so the parser -- where
/// every format decision lives -- can be tested against literals rather than
/// against whatever machine the test happens to run on.
#[derive(Debug, Clone, Copy)]
pub struct CacheIndex<'a> {
    /// Contents of `level`, e.g. `"3\n"`.
    pub level: &'a str,
    /// Contents of `type`, e.g. `"Unified\n"`.
    pub kind: &'a str,
    /// Contents of `shared_cpu_list`, e.g. `"0-7,64-71\n"`.
    pub shared_cpu_list: &'a str,
}

/// The set of CPUs that share one cache.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheDomain {
    cpus: Box<[usize]>,
}

impl CacheDomain {
    /// The domain's CPU ids, ascending and deduplicated.
    #[must_use]
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }

    /// The part of this domain that is also in `allowed`, or `None` if none of
    /// it is.
    ///
    /// This tree is not namespaced by cgroup and not filtered by affinity:
    /// `cpu100/cache/` exists here whatever the process is allowed to run on.
    /// A caller that turns a domain into an affinity mask therefore has to
    /// narrow it to the CPUs it may actually use, or it either escapes an
    /// operator's `taskset` confinement or hands the kernel CPUs a cpuset
    /// forbids and gets `EINVAL`.
    #[must_use]
    pub fn intersect(&self, allowed: &[usize]) -> Option<Self> {
        let cpus: Box<[usize]> = self
            .cpus
            .iter()
            .copied()
            .filter(|cpu| allowed.contains(cpu))
            .collect();
        if cpus.is_empty() {
            return None;
        }
        Some(Self { cpus })
    }
}

impl Display for CacheDomain {
    /// Renders back into sysfs's own range syntax (`0-7,64-71`) so a log line
    /// can be compared against `shared_cpu_list` by eye.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let mut index = 0;
        let mut first = true;
        while index < self.cpus.len() {
            let start = self.cpus[index];
            let mut end = start;
            while index + 1 < self.cpus.len() && self.cpus[index + 1] == end + 1 {
                index += 1;
                end = self.cpus[index];
            }
            if !first {
                formatter.write_str(",")?;
            }
            first = false;
            if start == end {
                write!(formatter, "{start}")?;
            } else {
                write!(formatter, "{start}-{end}")?;
            }
            index += 1;
        }
        Ok(())
    }
}

/// The distinct level-3 unified cache domains of a machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Domains {
    domains: Box<[CacheDomain]>,
}

impl L3Domains {
    /// Reads the machine's level-3 domains, or `None` if the topology cannot be
    /// established.
    #[must_use]
    pub fn read() -> Option<Self> {
        if cfg!(target_os = "linux") {
            Self::read_from(Path::new(CPU_SYSFS_ROOT))
        } else {
            None
        }
    }

    /// Reads a CPU sysfs tree rooted anywhere.
    fn read_from(root: &Path) -> Option<Self> {
        let mut files: Vec<(String, String, String)> = Vec::new();
        for cpu_entry in fs::read_dir(root).ok()? {
            let Ok(cpu_entry) = cpu_entry else { continue };
            let path = cpu_entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            // `cpufreq`, `cpuidle` and friends live next to the `cpuN` dirs.
            let Some(digits) = name.strip_prefix("cpu") else {
                continue;
            };
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            // Offline CPUs and restricted containers have no `cache` dir; that
            // is a missing CPU, not a broken machine, so skip rather than bail.
            let Ok(indices) = fs::read_dir(path.join("cache")) else {
                continue;
            };
            for index_entry in indices.flatten() {
                let index = index_entry.path();
                let is_index_dir = index
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("index"));
                if !is_index_dir {
                    continue;
                }
                // The level of an `index*` dir is whatever its `level` file
                // says; the numbering is not guaranteed to put L3 at `index3`.
                let (Ok(level), Ok(kind), Ok(list)) = (
                    fs::read_to_string(index.join("level")),
                    fs::read_to_string(index.join("type")),
                    fs::read_to_string(index.join("shared_cpu_list")),
                ) else {
                    continue;
                };
                files.push((level, kind, list));
            }
        }

        Self::from_cache_entries(files.iter().map(|(level, kind, list)| CacheIndex {
            level,
            kind,
            shared_cpu_list: list,
        }))
    }

    /// Collects the distinct level-3 unified domains described by `entries`.
    ///
    /// Every CPU in a domain reports that domain, so the same set arrives once
    /// per hardware thread and is deduplicated here.
    #[must_use]
    pub fn from_cache_entries<'a>(
        entries: impl IntoIterator<Item = CacheIndex<'a>>,
    ) -> Option<Self> {
        let mut domains: Vec<CacheDomain> = Vec::new();
        for entry in entries {
            // A cache level that is not an integer is not level 3; ignoring it
            // is different from the garbage case below, where a domain we do
            // want cannot be read and guessing at it would be worse than not
            // pinning.
            if entry.level.trim() != "3" || entry.kind.trim() != "Unified" {
                continue;
            }
            let cpus = parse_cpu_list(entry.shared_cpu_list)?;
            let domain = CacheDomain {
                cpus: cpus.into_boxed_slice(),
            };
            if !domains.contains(&domain) {
                domains.push(domain);
            }
        }
        if domains.is_empty() {
            return None;
        }
        // Ordered by lowest CPU id. Callers hand out contiguous worker blocks by
        // position, and sysfs directory order is not stable, so the mapping has
        // to come from the ids themselves to be reproducible across runs.
        domains.sort_unstable();
        Some(Self {
            domains: domains.into_boxed_slice(),
        })
    }

    /// The domains, ordered by their lowest CPU id.
    #[must_use]
    pub fn domains(&self) -> &[CacheDomain] {
        &self.domains
    }

    /// Total CPUs across all domains.
    #[must_use]
    pub fn total_cpus(&self) -> usize {
        self.domains.iter().map(|domain| domain.cpus.len()).sum()
    }
}

/// Parses a sysfs CPU list: comma-separated ids and `start-end` ranges, such as
/// `0-7,64-71` or `3`.
///
/// Returns `None` on anything that is not exactly that, including an empty
/// list, so a malformed file is never silently read as a smaller set of CPUs.
#[must_use]
pub fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',') {
        let part = part.trim();
        let Some((start, end)) = part.split_once('-') else {
            let cpu: usize = part.parse().ok()?;
            if cpu > MAX_CPU_ID {
                return None;
            }
            cpus.push(cpu);
            continue;
        };
        let start: usize = start.trim().parse().ok()?;
        let end: usize = end.trim().parse().ok()?;
        if end < start || end > MAX_CPU_ID {
            return None;
        }
        cpus.extend(start..=end);
    }
    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();
    cpus.dedup();
    Some(cpus)
}

#[cfg(test)]
mod tests {
    use super::{CacheIndex, L3Domains, parse_cpu_list};
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entry<'a>(level: &'a str, kind: &'a str, shared_cpu_list: &'a str) -> CacheIndex<'a> {
        CacheIndex {
            level,
            kind,
            shared_cpu_list,
        }
    }

    /// The eight domains of the Zen 5 box this was written for, in the exact
    /// form sysfs prints them (trailing newline included).
    fn zen5_entries() -> Vec<(String, String, String)> {
        let mut entries = Vec::new();
        for domain in 0..8 {
            let low = domain * 8;
            let high = 64 + domain * 8;
            let list = format!("{low}-{},{high}-{}\n", low + 7, high + 7);
            // Every one of the 16 threads in the domain reports it.
            for _ in 0..16 {
                entries.push(("3\n".to_owned(), "Unified\n".to_owned(), list.clone()));
            }
        }
        entries
    }

    /// Builds a throwaway CPU sysfs tree: `caches[cpu]` is that CPU's list of
    /// `(level, type, shared_cpu_list)` index dirs, written in the given order.
    fn fake_sysfs(caches: &[&[(&str, &str, &str)]]) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = env::temp_dir().join(format!(
            "steel-l3-topology-{}-{}",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        // Sysfs has non-CPU entries in this directory; the reader has to ignore
        // them rather than trip over them.
        fs::create_dir_all(root.join("cpufreq")).expect("temp dir should be creatable");
        fs::write(root.join("online"), "0-1\n").expect("temp file should be writable");
        for (cpu, indices) in caches.iter().enumerate() {
            for (index, (level, kind, list)) in indices.iter().enumerate() {
                let dir = root
                    .join(format!("cpu{cpu}"))
                    .join("cache")
                    .join(format!("index{index}"));
                fs::create_dir_all(&dir).expect("temp dir should be creatable");
                fs::write(dir.join("level"), level).expect("temp file should be writable");
                fs::write(dir.join("type"), kind).expect("temp file should be writable");
                fs::write(dir.join("shared_cpu_list"), list).expect("temp file should be writable");
            }
        }
        root
    }

    #[test]
    fn reads_a_sysfs_tree_where_l3_is_not_index3() {
        // Deliberately at index0 and index1: the level comes from the `level`
        // file, never from the directory number.
        let root = fake_sysfs(&[
            &[("3\n", "Unified\n", "0-1\n"), ("1\n", "Data\n", "0\n")],
            &[("1\n", "Data\n", "1\n"), ("3\n", "Unified\n", "0-1\n")],
        ]);
        let topology = L3Domains::read_from(&root);
        let _ = fs::remove_dir_all(&root);

        let topology = topology.expect("the fake tree describes one L3 domain");
        assert_eq!(topology.domains().len(), 1);
        assert_eq!(topology.domains()[0].to_string(), "0-1");
    }

    #[test]
    fn missing_sysfs_tree_is_none() {
        assert_eq!(
            L3Domains::read_from(Path::new("/nonexistent/steel/cpu/topology")),
            None
        );
    }

    #[test]
    fn sysfs_tree_without_caches_is_none() {
        let root = fake_sysfs(&[&[], &[]]);
        let topology = L3Domains::read_from(&root);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(topology, None);
    }

    #[test]
    fn parses_multiple_ranges() {
        assert_eq!(
            parse_cpu_list("0-3,64-67"),
            Some(vec![0, 1, 2, 3, 64, 65, 66, 67])
        );
    }

    #[test]
    fn parses_single_ids_and_mixed_forms() {
        assert_eq!(parse_cpu_list("3"), Some(vec![3]));
        assert_eq!(parse_cpu_list("0,2,4-6"), Some(vec![0, 2, 4, 5, 6]));
        assert_eq!(parse_cpu_list("5-5"), Some(vec![5]));
    }

    #[test]
    fn parses_trailing_newline_and_sorts() {
        assert_eq!(parse_cpu_list("64-65,0-1\n"), Some(vec![0, 1, 64, 65]));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("\n"), None);
        assert_eq!(parse_cpu_list("0-7,"), None);
        assert_eq!(parse_cpu_list("a-b"), None);
        assert_eq!(parse_cpu_list("0..7"), None);
        assert_eq!(parse_cpu_list("7-0"), None);
        assert_eq!(parse_cpu_list("-4"), None);
        // Would allocate ~800 GB if the range were expanded.
        assert_eq!(parse_cpu_list("0-99999999999"), None);
    }

    #[test]
    fn deduplicates_the_domain_every_cpu_reports() {
        let raw = zen5_entries();
        let topology = L3Domains::from_cache_entries(
            raw.iter()
                .map(|(level, kind, list)| entry(level, kind, list)),
        )
        .expect("eight well-formed L3 domains should parse");

        assert_eq!(topology.domains().len(), 8);
        assert_eq!(topology.total_cpus(), 128);
        assert_eq!(topology.domains()[0].cpus().len(), 16);
        assert_eq!(topology.domains()[0].to_string(), "0-7,64-71");
        assert_eq!(topology.domains()[7].to_string(), "56-63,120-127");
    }

    #[test]
    fn ignores_other_levels_and_types() {
        let topology = L3Domains::from_cache_entries([
            entry("1\n", "Data\n", "0,64\n"),
            entry("1\n", "Instruction\n", "0,64\n"),
            entry("2\n", "Unified\n", "0,64\n"),
            entry("3\n", "Unified\n", "0-3\n"),
            // An L3 victim cache exposed as Data would not be a sharing domain.
            entry("3\n", "Data\n", "0-63\n"),
        ])
        .expect("the level-3 unified entry should be found");

        assert_eq!(topology.domains().len(), 1);
        assert_eq!(topology.domains()[0].to_string(), "0-3");
    }

    #[test]
    fn single_domain_machine_is_reported_as_one_domain() {
        let topology = L3Domains::from_cache_entries([
            entry("3", "Unified", "0-15"),
            entry("3", "Unified", "0-15"),
        ])
        .expect("a single-socket single-CCX machine still has one domain");

        assert_eq!(topology.domains().len(), 1);
        assert_eq!(topology.total_cpus(), 16);
    }

    #[test]
    fn domains_are_ordered_by_lowest_cpu_regardless_of_read_order() {
        let topology = L3Domains::from_cache_entries([
            entry("3", "Unified", "16-31"),
            entry("3", "Unified", "0-15"),
            entry("3", "Unified", "48-63"),
            entry("3", "Unified", "32-47"),
        ])
        .expect("four domains should parse");

        let firsts: Vec<usize> = topology
            .domains()
            .iter()
            .map(|domain| domain.cpus()[0])
            .collect();
        assert_eq!(firsts, vec![0, 16, 32, 48]);
    }

    #[test]
    fn no_level_three_entries_is_none() {
        assert_eq!(
            L3Domains::from_cache_entries([entry("1", "Data", "0"), entry("2", "Unified", "0")]),
            None
        );
        assert_eq!(L3Domains::from_cache_entries([]), None);
    }

    #[test]
    fn intersecting_a_domain_keeps_only_allowed_cpus() {
        let topology = L3Domains::from_cache_entries([
            entry("3", "Unified", "0-7,64-71"),
            entry("3", "Unified", "8-15,72-79"),
        ])
        .expect("two domains should parse");
        // What a `taskset -c 0-15` process sees: the first domain survives as
        // its low half, the second as the other low half, and the SMT siblings
        // above 63 are gone from both.
        let allowed: Vec<usize> = (0..16).collect();

        assert_eq!(
            topology.domains()[0]
                .intersect(&allowed)
                .expect("cpus 0-7 are allowed")
                .to_string(),
            "0-7"
        );
        assert_eq!(
            topology.domains()[1]
                .intersect(&allowed)
                .expect("cpus 8-15 are allowed")
                .to_string(),
            "8-15"
        );
    }

    #[test]
    fn a_domain_outside_the_allowed_set_intersects_to_none() {
        let topology = L3Domains::from_cache_entries([
            entry("3", "Unified", "0-7"),
            entry("3", "Unified", "8-15"),
        ])
        .expect("two domains should parse");

        assert_eq!(topology.domains()[1].intersect(&[0, 1, 2]), None);
        assert_eq!(topology.domains()[0].intersect(&[]), None);
    }

    #[test]
    fn unparseable_level_three_domain_is_none_not_a_guess() {
        assert_eq!(
            L3Domains::from_cache_entries([
                entry("3", "Unified", "0-15"),
                entry("3", "Unified", "not-a-list"),
            ]),
            None
        );
    }
}
