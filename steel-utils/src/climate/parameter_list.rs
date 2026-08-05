//! R-Tree backed parameter list for climate-based biome lookup.
//!
//! Implements vanilla's `Climate.ParameterList` with a flattened R-Tree for
//! cache-efficient nearest-neighbor search. The tree is built once at startup
//! using vanilla's algorithm, then flattened into a BFS-ordered contiguous
//! array where children of the same parent occupy adjacent indices.

// Vanilla uses `(a + b) / 2` (integer division, truncates toward zero) for midpoints.
// `i64::midpoint` rounds differently for negative odd sums, so we can't use it.
#![expect(
    clippy::manual_midpoint,
    reason = "must match vanilla's truncate-toward-zero behavior"
)]

use std::cmp::Ordering;
use std::simd::Simd;
use std::simd::cmp::SimdOrd;
use std::simd::num::SimdInt;

use super::PARAMETER_COUNT;
use super::types::{Parameter, ParameterPoint, TargetPoint};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Maximum children per tree node. Matches vanilla's `CHILDREN_PER_NODE` = 6.
const CHILDREN_PER_NODE: usize = 6;

/// Lane count for the distance kernel: `PARAMETER_COUNT` rounded up to a vector
/// width. The 8th lane is padding and is held at zero everywhere.
const LANES: usize = PARAMETER_COUNT.next_power_of_two();

/// A search target laid out for [`FlatNode::distance`].
///
/// Lane 7 is padding and is always zero. Built once per lookup by
/// [`pack_target`] and then carried unchanged through the whole tree walk, so
/// the widening happens once rather than once per visited node.
type PackedTarget = Simd<i64, LANES>;

/// Widen a target to the kernel's lane count, zero-filling the padding lane.
///
/// The target keeps its full `i64` range: `TargetPoint`'s fields are public
/// `i64` and nothing bounds them, so narrowing here would put a precondition on
/// a public API that callers have no way to see.
#[inline]
fn pack_target(target: &[i64; PARAMETER_COUNT]) -> PackedTarget {
    Simd::load_or_default(target)
}

/// R-Tree node used during construction only. After building, the tree is
/// flattened into a `Vec<FlatNode>` for search.
enum RTreeNode {
    /// Leaf node containing a single biome entry.
    Leaf {
        parameter_space: [Parameter; PARAMETER_COUNT],
        value_index: usize,
    },
    /// Internal node with children and a bounding box.
    SubTree {
        parameter_space: [Parameter; PARAMETER_COUNT],
        children: Vec<RTreeNode>,
    },
}

impl RTreeNode {
    /// Get the parameter space (bounding box) of this node.
    const fn parameter_space(&self) -> &[Parameter; PARAMETER_COUNT] {
        match self {
            Self::Leaf {
                parameter_space, ..
            }
            | Self::SubTree {
                parameter_space, ..
            } => parameter_space,
        }
    }
}

/// Build data used during tree construction.
#[derive(Clone)]
struct BuildEntry {
    parameter_space: [Parameter; PARAMETER_COUNT],
    index: usize,
}

/// Build the bounding box for a set of child nodes.
fn build_parameter_space(children: &[RTreeNode]) -> [Parameter; PARAMETER_COUNT] {
    let mut bounds: [Option<Parameter>; PARAMETER_COUNT] = [None; PARAMETER_COUNT];
    for child in children {
        let ps = child.parameter_space();
        for d in 0..PARAMETER_COUNT {
            bounds[d] = Some(ps[d].span_with(bounds[d].as_ref()));
        }
    }
    bounds.map(|b| b.expect("bounds should be initialized"))
}

/// Calculate the cost of a bounding box (sum of range widths).
fn cost(parameter_space: &[Parameter; PARAMETER_COUNT]) -> i64 {
    let mut result = 0i64;
    for p in parameter_space {
        result += (p.max - p.min).abs();
    }
    result
}

/// Build an R-Tree from a list of entries, matching vanilla's algorithm.
fn build_tree(entries: &mut [BuildEntry]) -> RTreeNode {
    assert!(!entries.is_empty());

    if entries.len() == 1 {
        return RTreeNode::Leaf {
            parameter_space: entries[0].parameter_space,
            value_index: entries[0].index,
        };
    }

    if entries.len() <= CHILDREN_PER_NODE {
        // Sort by total magnitude of centers across all dimensions
        entries.sort_by_key(|e| {
            let mut total: i64 = 0;
            for d in 0..PARAMETER_COUNT {
                let p = &e.parameter_space[d];
                total += ((p.min + p.max) / 2).abs();
            }
            total
        });

        let children: Vec<RTreeNode> = entries
            .iter()
            .map(|e| RTreeNode::Leaf {
                parameter_space: e.parameter_space,
                value_index: e.index,
            })
            .collect();
        let ps = build_parameter_space(&children);
        return RTreeNode::SubTree {
            parameter_space: ps,
            children,
        };
    }

    // Try splitting along each dimension, choose minimum cost.
    // Like vanilla, save the bucketized entries when we find a better dimension.
    // This is critical because vanilla sorts `children` in-place for each dimension
    // and `bucketize()` captures a snapshot. Later sorts don't affect saved buckets.
    // Re-sorting after the loop would produce different stable-sort tie-breaking.
    let mut min_cost = i64::MAX;
    let mut best_dim = 0;
    let mut best_buckets: Option<Vec<Vec<BuildEntry>>> = None;

    for d in 0..PARAMETER_COUNT {
        sort_entries(entries, d);
        let (bucket_cost, buckets) = snapshot_buckets(entries);
        if min_cost > bucket_cost {
            min_cost = bucket_cost;
            best_dim = d;
            best_buckets = Some(buckets);
        }
    }

    // Build subtrees from the saved buckets (entries in the order sorted by best_dim)
    let buckets = best_buckets.expect("should have found at least one dimension");

    // Compute bounding box for each bucket, pair with entries for sorting
    let mut bucket_subtrees: Vec<([Parameter; PARAMETER_COUNT], Vec<BuildEntry>)> = buckets
        .into_iter()
        .map(|bucket_entries| {
            let mut bounds: [Option<Parameter>; PARAMETER_COUNT] = [None; PARAMETER_COUNT];
            for e in &bucket_entries {
                #[expect(clippy::needless_range_loop, reason = "dim indexes parallel arrays")]
                for dim in 0..PARAMETER_COUNT {
                    bounds[dim] = Some(e.parameter_space[dim].span_with(bounds[dim].as_ref()));
                }
            }
            let ps = bounds.map(|b| b.expect("bounds should be initialized"));
            (ps, bucket_entries)
        })
        .collect();

    // Sort the bucket subtrees by the best dimension (absolute=true)
    sort_bucket_subtrees(&mut bucket_subtrees, best_dim);
    // For each bucket, recursively build
    let final_children: Vec<RTreeNode> = bucket_subtrees
        .into_par_iter()
        .map(|(_, mut child_entries)| build_tree(&mut child_entries))
        .collect();

    let ps = build_parameter_space(&final_children);
    RTreeNode::SubTree {
        parameter_space: ps,
        children: final_children,
    }
}

/// Sort entries by a dimension, with tiebreaking by subsequent dimensions.
fn sort_entries(entries: &mut [BuildEntry], dimension: usize) {
    entries.sort_by(|a, b| {
        for offset in 0..PARAMETER_COUNT {
            let d = (dimension + offset) % PARAMETER_COUNT;
            let center_a = (a.parameter_space[d].min + a.parameter_space[d].max) / 2;
            let center_b = (b.parameter_space[d].min + b.parameter_space[d].max) / 2;
            let cmp = center_a.cmp(&center_b);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });
}

/// Sort bucket subtrees by a dimension (absolute=true), matching vanilla's
/// `sort(minBuckets, dimensions, minDimension, true)`.
fn sort_bucket_subtrees(
    subtrees: &mut [([Parameter; PARAMETER_COUNT], Vec<BuildEntry>)],
    dimension: usize,
) {
    subtrees.sort_by(|a, b| {
        for offset in 0..PARAMETER_COUNT {
            let d = (dimension + offset) % PARAMETER_COUNT;
            let center_a = (a.0[d].min + a.0[d].max) / 2;
            let center_b = (b.0[d].min + b.0[d].max) / 2;
            let cmp = center_a.abs().cmp(&center_b.abs());
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });
}

/// Compute the expected bucket size from vanilla's formula.
fn expected_children_count(total: usize) -> usize {
    let log_base_6 = ((total as f64) - 0.01).ln() / (CHILDREN_PER_NODE as f64).ln();
    (CHILDREN_PER_NODE as f64).powf(log_base_6.floor()) as usize
}

/// Snapshot the current entry order into buckets and compute total cost.
///
/// This matches vanilla's `bucketize()` which creates `SubTree` objects that
/// capture the children's current sorted order. We return cloned entries so
/// that later sorts of the original slice don't affect the saved buckets.
#[expect(
    clippy::needless_range_loop,
    reason = "indexing into PARAMETER_COUNT parallel arrays; iterator would be less clear"
)]
fn snapshot_buckets(entries: &[BuildEntry]) -> (i64, Vec<Vec<BuildEntry>>) {
    let expected = expected_children_count(entries.len());
    let mut buckets = Vec::new();
    let mut total_cost = 0i64;
    let mut start = 0;
    while start < entries.len() {
        let end = (start + expected).min(entries.len());
        let bucket = entries[start..end].to_vec();
        // Compute bounding box cost for this bucket
        let mut bounds: [Option<Parameter>; PARAMETER_COUNT] = [None; PARAMETER_COUNT];
        for e in &bucket {
            for d in 0..PARAMETER_COUNT {
                bounds[d] = Some(e.parameter_space[d].span_with(bounds[d].as_ref()));
            }
        }
        let ps = bounds.map(|b| b.expect("bounds should be initialized"));
        total_cost += cost(&ps);
        buckets.push(bucket);
        start = end;
    }
    (total_cost, buckets)
}

/// Narrows a bounding-box bound to the width `FlatNode` stores it at.
///
/// Every bound originates in `quantize_coord`, which is
/// `((coord as f32) * 10000.0) as i64` over a climate parameter in roughly
/// [-2, 2], so the real range is about +/-20,000 and `i32` is far wider than it
/// needs to be. This is a build-time conversion, run once at startup per
/// dimension, so the check is free.
///
/// It panics rather than saturating on purpose. A bound that did not fit would
/// mean the quantisation contract changed, and silently clamping it would move a
/// biome's bounding box -- which is a parity break that no test would attribute
/// back to here.
fn narrow_bound(bound: i64) -> i32 {
    i32::try_from(bound).unwrap_or_else(|_| {
        panic!(
            "climate bound {bound} does not fit in i32; quantize_coord is supposed to keep these \
             near +/-20,000, so either the quantisation or the parameter range changed"
        )
    })
}

/// Compact node for the flattened R-Tree.
///
/// Children of the same parent are stored at contiguous indices in a single
/// `Vec<FlatNode>`, enabling cache-efficient iteration during search.
/// The BFS-order layout also means that nodes accessed together during a
/// search tend to be near each other in memory.
struct FlatNode {
    /// Bounding box minimum values for each parameter dimension.
    ///
    /// Narrower than the `i64` these are compared against, so that a node is
    /// exactly one cache line: see the note on the struct's size below.
    mins: [i32; PARAMETER_COUNT],
    /// Bounding box maximum values for each parameter dimension.
    maxs: [i32; PARAMETER_COUNT],
    /// For leaf nodes: index into the values array.
    /// For subtree nodes: start index of the children in the nodes array.
    ///
    /// The two never coexist -- a node is a leaf or a subtree -- so they share
    /// the word, which is what buys the last four bytes. `children_count`
    /// discriminates; read it through `value_index()` or `children_start()`
    /// rather than directly, so the discriminant is always checked.
    payload: u32,
    /// Number of children (0 = leaf, 1..=6 = subtree).
    children_count: u8,
}

// The whole point of the layout above: 7+7 bounds at 4 bytes, a shared 4-byte
// payload and a 1-byte tag is 61 bytes, which pads to exactly one 64-byte cache
// line. At `[i64; 7]` a node was 121 bytes padded to 128 -- two lines, and two
// misses per node visit on a data-dependent walk.
//
// This matters more than one line per node sounds like. `search_nearest` is hit
// 1,536 times per chunk by `create_biomes`, on every generation thread at once,
// and the overworld tree is ~9.1 K nodes. At 128 B that array is ~1.17 MB,
// larger than a core's 1 MiB L2 slice, so the walk misses to L3 indefinitely. At
// 64 B it is ~585 KB and fits with room to spare.
const _: () = assert!(
    size_of::<FlatNode>() == 64,
    "FlatNode must stay one cache line; check the field packing above"
);

impl FlatNode {
    /// Compute the squared distance from a target point to this node's bounding box.
    ///
    /// One 8-lane pass over a 7-dimensional box. `PARAMETER_COUNT` is 7, which
    /// is not a vector width, and left to itself LLVM split each node into a
    /// 4-lane `ymm` chunk, a 2-lane `xmm` chunk and a scalar tail -- two
    /// horizontal reduction chains and two `vpmullq` per node. Rounding the
    /// work up to 8 lanes and eating one dead lane is strictly cheaper than
    /// splitting it three ways.
    ///
    /// The arithmetic stays `i64` end to end. The bounds are *stored* as `i32`
    /// (see the layout note above) but a per-dimension difference reaches
    /// 20,000, its square 4x10^8, and seven of those sum to ~2.8x10^9 -- past
    /// `i32::MAX` (2.147x10^9). Widening the loaded bounds and keeping the
    /// lanes 64-bit means this kernel is bit-identical to the scalar loop it
    /// replaces for *every* `i64` target, with no range precondition on the
    /// caller: same operations, same order, same wrapping behaviour. Narrowing
    /// the lanes to `i32` would be faster still, but only for targets close
    /// enough to the tree, and a target is public API -- `TargetPoint`'s fields
    /// are plain `i64`. `distance_scalar` in the tests below is kept as the
    /// reference this is checked against.
    #[inline]
    fn distance(&self, target: PackedTarget) -> i64 {
        // Lane 7 is padding: `load_or_default` zero-fills it from the 7-element
        // arrays, and `pack_target` zero-fills the target's, so it computes
        // `max(0 - 0, 0 - 0, 0)^2 == 0` and cannot contribute to the sum. On
        // AVX-512 each of these is a single masked load, so the padding costs
        // nothing to suppress.
        let mins: PackedTarget = Simd::<i32, LANES>::load_or_default(&self.mins).cast();
        let maxs: PackedTarget = Simd::<i32, LANES>::load_or_default(&self.maxs).cast();

        let di = (target - maxs)
            .simd_max(mins - target)
            .simd_max(Simd::splat(0));
        (di * di).reduce_sum()
    }

    #[inline]
    const fn is_leaf(&self) -> bool {
        self.children_count == 0
    }

    /// Index into the values array. Leaves only.
    #[inline]
    const fn value_index(&self) -> u32 {
        debug_assert!(self.is_leaf(), "value_index read from a subtree node");
        self.payload
    }

    /// Start index of this node's children. Subtrees only.
    #[inline]
    const fn children_start(&self) -> u32 {
        debug_assert!(!self.is_leaf(), "children_start read from a leaf node");
        self.payload
    }
}

/// Flatten an R-Tree into a contiguous `Vec` using BFS ordering.
///
/// BFS guarantees that all children of the same parent occupy contiguous
/// indices, which is the key property for cache-efficient search.
fn flatten_tree(root: RTreeNode) -> Vec<FlatNode> {
    use std::collections::VecDeque;

    let mut nodes: Vec<FlatNode> = Vec::new();
    // Queue of (children_batch, parent_flat_index).
    // Each batch is a Vec of siblings that will be laid out contiguously.
    let mut queue: VecDeque<(Vec<RTreeNode>, Option<u32>)> = VecDeque::new();
    queue.push_back((vec![root], None));

    while let Some((batch, parent_idx)) = queue.pop_front() {
        let batch_start = nodes.len() as u32;

        // Fix up parent's children_start to point to this batch
        if let Some(pidx) = parent_idx {
            nodes[pidx as usize].payload = batch_start;
        }

        for node in batch {
            let flat_idx = nodes.len() as u32;
            match node {
                RTreeNode::Leaf {
                    parameter_space,
                    value_index,
                } => {
                    nodes.push(FlatNode {
                        mins: parameter_space.map(|p| narrow_bound(p.min)),
                        maxs: parameter_space.map(|p| narrow_bound(p.max)),
                        payload: value_index as u32,
                        children_count: 0,
                    });
                }
                RTreeNode::SubTree {
                    parameter_space,
                    children,
                } => {
                    let children_count = children.len() as u8;
                    nodes.push(FlatNode {
                        mins: parameter_space.map(|p| narrow_bound(p.min)),
                        maxs: parameter_space.map(|p| narrow_bound(p.max)),
                        payload: 0, // fixed up when the children batch is processed
                        children_count,
                    });
                    queue.push_back((children, Some(flat_idx)));
                }
            }
        }
    }

    nodes
}

/// Search the flat R-Tree for the nearest leaf to the target.
///
/// Matches vanilla's `SubTree.search()` which passes the candidate through
/// recursion and checks the returned leaf distance against the local best.
fn search_nearest(
    nodes: &[FlatNode],
    node: &FlatNode,
    target: PackedTarget,
    best_dist: &mut i64,
    best_idx: &mut Option<u32>,
) {
    let start = node.children_start() as usize;
    let end = start + node.children_count as usize;
    let children = &nodes[start..end];

    for child in children {
        let child_dist = child.distance(target);
        // Vanilla uses strict > for pruning (skips equal distance)
        if *best_dist > child_dist {
            if child.is_leaf() {
                // Leaf: child_dist IS the exact distance — no recursion needed
                *best_dist = child_dist;
                *best_idx = Some(child.value_index());
            } else {
                // Subtree: recurse into children
                search_nearest(nodes, child, target, best_dist, best_idx);
            }
        }
    }
}

/// A list of biome parameter points with their associated values.
///
/// Uses an R-Tree for lookup matching vanilla's `Climate.ParameterList`.
pub struct ParameterList<T> {
    /// The biome entries (parameter point, value pairs)
    values: Vec<(ParameterPoint, T)>,
    /// Cached parameter spaces for each value (for distance computation in lastResult)
    param_spaces: Vec<[Parameter; PARAMETER_COUNT]>,
    /// Flat R-Tree nodes in BFS order. Root is at index 0.
    nodes: Vec<FlatNode>,
}

impl<T> ParameterList<T> {
    /// Create a new parameter list from values, building an R-Tree index.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty.
    #[must_use]
    pub fn new(values: Vec<(ParameterPoint, T)>) -> Self {
        assert!(!values.is_empty(), "Need at least one value");

        let param_spaces: Vec<[Parameter; PARAMETER_COUNT]> =
            values.iter().map(|(pp, _)| pp.parameter_space()).collect();

        // Build R-Tree from the parameter points
        let mut entries: Vec<BuildEntry> = values
            .iter()
            .enumerate()
            .map(|(i, (pp, _))| BuildEntry {
                parameter_space: pp.parameter_space(),
                index: i,
            })
            .collect();

        let root = build_tree(&mut entries);
        let nodes = flatten_tree(root);

        Self {
            values,
            param_spaces,
            nodes,
        }
    }

    /// Get the underlying values.
    #[must_use]
    pub fn values(&self) -> &[(ParameterPoint, T)] {
        &self.values
    }

    /// Find the best matching value for a target point (no caching).
    ///
    /// Uses R-Tree search matching vanilla's `Climate.ParameterList.findValue()`.
    ///
    /// Note: Vanilla warm-starts with `lastResult` via `ThreadLocal`, which can
    /// affect tie-breaking on equal-distance candidates. This version starts
    /// from `i64::MAX` (no warm-start). Use `find_value_cached` for the hot
    /// path to match vanilla's tie-breaking behavior.
    ///
    /// # Panics
    ///
    /// Panics if the R-Tree search fails to find any matching value.
    #[must_use]
    pub fn find_value(&self, target: &TargetPoint) -> &T {
        let target_array = target.to_parameter_array();
        let root = &self.nodes[0];
        if root.is_leaf() {
            return &self.values[root.value_index() as usize].1;
        }
        let mut best_dist = i64::MAX;
        let mut best_idx = None;
        search_nearest(
            &self.nodes,
            root,
            pack_target(&target_array),
            &mut best_dist,
            &mut best_idx,
        );
        &self.values[best_idx.expect("R-Tree search should always find a value") as usize].1
    }

    /// Find the best matching value with lastResult caching.
    ///
    /// Matches vanilla's `Climate.ParameterList.findValue()` with `ThreadLocal`
    /// `lastNode` warm-starting. The cache stores the index of the last result,
    /// which is used as the initial candidate for the next search, improving
    /// both performance and tie-breaking behavior.
    ///
    /// # Panics
    ///
    /// Panics if the R-Tree search fails to find any matching value.
    #[must_use]
    pub fn find_value_cached(&self, target: &TargetPoint, cache: &mut Option<usize>) -> &T {
        let target_array = target.to_parameter_array();

        let root = &self.nodes[0];
        if root.is_leaf() {
            let idx = root.value_index() as usize;
            *cache = Some(idx);
            return &self.values[idx].1;
        }

        // Compute initial distance from cached last result
        let (mut best_dist, init_idx) = match *cache {
            Some(idx) => {
                let ps = &self.param_spaces[idx];
                let mut d = 0i64;
                for i in 0..PARAMETER_COUNT {
                    let di = ps[i].distance(target_array[i]);
                    d += di * di;
                }
                (d, Some(idx as u32))
            }
            None => (i64::MAX, None),
        };

        let mut best_idx = init_idx;
        search_nearest(
            &self.nodes,
            root,
            pack_target(&target_array),
            &mut best_dist,
            &mut best_idx,
        );
        let result_idx = best_idx.expect("R-Tree search should always find a value") as usize;

        *cache = Some(result_idx);
        &self.values[result_idx].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    impl FlatNode {
        /// The scalar loop `distance` replaced, kept verbatim as the reference
        /// the vector kernel is checked against.
        ///
        /// Do not "simplify" this to call `distance`: its whole job is to be an
        /// independently written second opinion.
        #[expect(
            clippy::needless_range_loop,
            reason = "indexing into parallel min/max arrays; iterator zip would be less clear"
        )]
        fn distance_scalar(&self, target: &[i64; PARAMETER_COUNT]) -> i64 {
            let mut d = 0i64;
            for i in 0..PARAMETER_COUNT {
                let di = (target[i] - i64::from(self.maxs[i]))
                    .max(i64::from(self.mins[i]) - target[i])
                    .max(0);
                d += di * di;
            }
            d
        }
    }

    /// Build a leaf whose box is `[mins[d], maxs[d]]` per dimension.
    fn node(mins: [i32; PARAMETER_COUNT], maxs: [i32; PARAMETER_COUNT]) -> FlatNode {
        FlatNode {
            mins,
            maxs,
            payload: 0,
            children_count: 0,
        }
    }

    fn assert_same(n: &FlatNode, target: [i64; PARAMETER_COUNT]) {
        let scalar = n.distance_scalar(&target);
        let vector = n.distance(pack_target(&target));
        assert_eq!(
            vector, scalar,
            "vector kernel disagreed with the scalar reference\n  mins   = {:?}\n  maxs   = {:?}\n  target = {target:?}",
            n.mins, n.maxs,
        );
    }

    /// The vector kernel must return *exactly* the scalar result, not a close
    /// one: `search_nearest` prunes with a strict `>`, so a distance that ties
    /// differently picks a different biome.
    #[test]
    fn distance_matches_scalar_reference_over_random_sweep() {
        let mut rng = StdRng::seed_from_u64(0x0DDB_A11C_0FFE_E5EE);

        // Three tiers, widening from the range worldgen actually produces out
        // to the widest range that cannot overflow the reference itself.
        //
        // `quantize_coord` is `((c as f32) * 10000.0) as i64` over a climate
        // parameter in roughly [-2, 2], so real bounds and targets live inside
        // +/-20,000; tier 0 is that, plus slack to land outside every box.
        //
        // The reference squares and sums seven differences in `i64`, so it
        // overflows once a difference passes ~1.1x10^9 (7 * d^2 <= i64::MAX).
        // Tier 2 stays under that: beyond it the reference is not a reference
        // any more, it is UB in debug and a wrap in release.
        for (tier, span) in [25_000i64, 1_000_000, 500_000_000].into_iter().enumerate() {
            for _ in 0..40_000 {
                let mut mins = [0i32; PARAMETER_COUNT];
                let mut maxs = [0i32; PARAMETER_COUNT];
                let mut target = [0i64; PARAMETER_COUNT];
                for d in 0..PARAMETER_COUNT {
                    let a = rng.random_range(-span..=span);
                    let b = rng.random_range(-span..=span);
                    mins[d] = i32::try_from(a.min(b)).expect("tier span fits i32");
                    maxs[d] = i32::try_from(a.max(b)).expect("tier span fits i32");
                    target[d] = rng.random_range(-span..=span);
                }
                assert_same(&node(mins, maxs), target);
            }
            assert!(tier < 3);
        }
    }

    /// Largest per-dimension difference the `i64` reference can square and sum
    /// seven of without overflowing: `sqrt(i64::MAX / 7)`.
    const REFERENCE_LIMIT: i64 = 1_147_878_293;

    /// Values that sit exactly on, just inside and just outside a box edge,
    /// plus the quantisation extremes. Capped so that the *reference* stays
    /// well-defined -- see `distance_matches_scalar_reference_when_it_wraps`
    /// for the range past this.
    const INTERESTING: [i64; 11] = [
        -500_000_000,
        -46_341,
        -46_340, // -floor(sqrt(i32::MAX)): the i32-lane cutoff, if we ever narrow
        -20_001,
        -20_000, // the quantisation range: ((c as f32) * 10000.0) over c in [-2, 2]
        -1,
        0,
        1,
        20_000,
        46_340,
        500_000_000,
    ];

    /// The interesting inputs are the ones on the boundary, where a one-off in
    /// the `max` chain flips a `>` and picks a different biome.
    #[test]
    fn distance_matches_scalar_reference_on_boundaries() {
        for &lo in &INTERESTING {
            for &hi in &INTERESTING {
                if lo > hi {
                    continue;
                }
                let mins = [i32::try_from(lo).expect("INTERESTING fits i32"); PARAMETER_COUNT];
                let maxs = [i32::try_from(hi).expect("INTERESTING fits i32"); PARAMETER_COUNT];
                let n = node(mins, maxs);
                for &t in &INTERESTING {
                    // Straddle each edge in both directions as well.
                    for delta in [-1i64, 0, 1] {
                        let t = t + delta;
                        assert!(
                            (t - lo).abs().max((t - hi).abs()) <= REFERENCE_LIMIT,
                            "test input would overflow the reference, not the kernel",
                        );
                        assert_same(&n, [t; PARAMETER_COUNT]);
                    }
                }
            }
        }
    }

    /// Past `REFERENCE_LIMIT` the original scalar loop overflows `i64` itself:
    /// `i32::MIN` against `i32::MAX` is a difference of 4.3x10^9, whose square
    /// is 1.8x10^19. That is a property of the code this replaces, not of the
    /// replacement -- release builds wrap there and always have.
    ///
    /// Parity still has to hold in that regime, because release is what ships.
    /// Two's-complement add and multiply wrap associatively, so the vector
    /// kernel's tree-shaped reduction wraps to the same value as the scalar
    /// loop's sequential one. This pins that down against a reference written
    /// with explicit wrapping ops, which debug builds will not trap on.
    #[test]
    fn distance_matches_scalar_reference_when_it_wraps() {
        #[expect(
            clippy::needless_range_loop,
            reason = "mirrors the scalar reference's indexing, deliberately"
        )]
        fn wrapping_reference(n: &FlatNode, target: &[i64; PARAMETER_COUNT]) -> i64 {
            let mut d = 0i64;
            for i in 0..PARAMETER_COUNT {
                let di = (target[i].wrapping_sub(i64::from(n.maxs[i])))
                    .max(i64::from(n.mins[i]).wrapping_sub(target[i]))
                    .max(0);
                d = d.wrapping_add(di.wrapping_mul(di));
            }
            d
        }

        let extremes = [
            i64::from(i32::MIN),
            i64::from(i32::MIN) + 1,
            -1,
            0,
            1,
            i64::from(i32::MAX) - 1,
            i64::from(i32::MAX),
        ];

        let mut rng = StdRng::seed_from_u64(0xC0FF_EE15_600D);
        for _ in 0..20_000 {
            let mut mins = [0i32; PARAMETER_COUNT];
            let mut maxs = [0i32; PARAMETER_COUNT];
            let mut target = [0i64; PARAMETER_COUNT];
            for d in 0..PARAMETER_COUNT {
                let a: i32 = rng.random();
                let b: i32 = rng.random();
                mins[d] = a.min(b);
                maxs[d] = a.max(b);
                // Full i64 targets: nothing bounds `TargetPoint`'s fields.
                target[d] = if rng.random_bool(0.5) {
                    extremes[rng.random_range(0..extremes.len())]
                } else {
                    rng.random()
                };
            }
            let n = node(mins, maxs);
            assert_eq!(
                n.distance(pack_target(&target)),
                wrapping_reference(&n, &target),
                "vector kernel and scalar loop wrapped differently\n  mins   = {mins:?}\n  maxs   = {maxs:?}\n  target = {target:?}",
            );
        }
    }

    /// A degenerate box (min == max) and a target inside it must both give
    /// exactly zero, and a zero must survive the padding lane.
    #[test]
    fn distance_inside_the_box_is_zero() {
        let n = node([-100; PARAMETER_COUNT], [100; PARAMETER_COUNT]);
        for t in [-100i64, -50, 0, 50, 100] {
            assert_eq!(n.distance(pack_target(&[t; PARAMETER_COUNT])), 0);
        }

        // Only one dimension escapes the box: the other six, and the padding
        // lane, must contribute nothing.
        let mut target = [0i64; PARAMETER_COUNT];
        target[3] = 130;
        assert_eq!(n.distance(pack_target(&target)), 30 * 30);
        assert_same(&n, target);
    }

    /// Mixed per-dimension boxes, so a bug that broadcasts one lane's bound
    /// across the vector cannot pass.
    #[test]
    fn distance_uses_each_dimension_independently() {
        let mins = [-10, -20, -30, -40, -50, -60, -70];
        let maxs = [10, 20, 30, 40, 50, 60, 70];
        let n = node(mins, maxs);
        let target = [100i64, -200, 300, -400, 500, -600, 700];
        // 90^2 + 180^2 + 270^2 + 360^2 + 450^2 + 540^2 + 630^2
        let expected: i64 = [90i64, 180, 270, 360, 450, 540, 630]
            .iter()
            .map(|d| d * d)
            .sum();
        assert_eq!(n.distance(pack_target(&target)), expected);
        assert_same(&n, target);
    }

    #[test]
    fn test_parameter_list_find_value() {
        let values = vec![
            (
                ParameterPoint::new(
                    Parameter::new(-10000, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    0,
                ),
                "cold",
            ),
            (
                ParameterPoint::new(
                    Parameter::new(0, 10000),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    Parameter::new(0, 0),
                    0,
                ),
                "hot",
            ),
        ];

        let list = ParameterList::new(values);

        // Cold biome should match negative temperature
        let cold_target = TargetPoint::new(-5000, 0, 0, 0, 0, 0);
        assert_eq!(*list.find_value(&cold_target), "cold");

        // Hot biome should match positive temperature
        let hot_target = TargetPoint::new(5000, 0, 0, 0, 0, 0);
        assert_eq!(*list.find_value(&hot_target), "hot");
    }
}
