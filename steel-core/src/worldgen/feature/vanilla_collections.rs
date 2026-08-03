use std::collections::VecDeque;

use super::prelude::*;

/// Block-position set for feature code that vanilla models as `HashSet<BlockPos>`.
///
/// Java `HashSet` iteration order is implementation-defined, so the extractor normalizes these
/// worldgen sets to insertion order. Steel follows that deterministic oracle instead of depending
/// on JVM bucket ordering.
///
/// `entries` holds exactly the present positions, in insertion order: `insert`
/// appends only when `present` gained the position and `remove` drops it from
/// both. Keeping that invariant is what lets iteration skip a hash probe per
/// element and lets `pop_java_ordered_position` take the head directly, which
/// the leaf-distance frontier in `update_tree_leaves` drains one position at a
/// time -- previously by materializing the whole remaining frontier into a
/// `Vec` and then walking `entries` again to erase the head.
#[derive(Default)]
pub(super) struct JavaBlockPosSet {
    entries: VecDeque<BlockPos>,
    present: FxHashSet<BlockPos>,
}

impl JavaBlockPosSet {
    pub(super) fn insert(&mut self, pos: BlockPos) -> bool {
        if !self.present.insert(pos) {
            return false;
        }

        self.entries.push_back(pos);
        true
    }

    /// Removal of an arbitrary position, as opposed to the head that
    /// [`Self::pop_java_ordered_position`] takes. No feature needs it today, but
    /// it is what keeps `entries` free of stale positions -- the invariant the
    /// rest of this type reads from -- so it stays with the type rather than
    /// being reconstructed by whichever feature next models a `HashSet.remove`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "set contract; exercised by this module's tests")
    )]
    pub(super) fn remove(&mut self, pos: BlockPos) -> bool {
        if !self.present.remove(&pos) {
            return false;
        }

        self.entries.retain(|entry| *entry != pos);
        true
    }

    pub(super) fn contains(&self, pos: BlockPos) -> bool {
        self.present.contains(&pos)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    pub(super) fn insertion_order(&self) -> impl Iterator<Item = &BlockPos> {
        self.entries.iter()
    }

    pub(super) fn java_ordered_positions(&self) -> Vec<BlockPos> {
        self.insertion_order().copied().collect()
    }

    pub(super) fn pop_java_ordered_position(&mut self) -> Option<BlockPos> {
        let pos = self.entries.pop_front()?;
        self.present.remove(&pos);
        Some(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_positions_keep_first_insertion() {
        let mut set = JavaBlockPosSet::default();
        assert!(set.insert(BlockPos::new(1, 2, 3)));
        assert!(!set.insert(BlockPos::new(1, 2, 3)));
        assert_eq!(set.java_ordered_positions(), [BlockPos::new(1, 2, 3)]);
    }

    #[test]
    fn removed_positions_do_not_iterate() {
        let mut set = JavaBlockPosSet::default();
        for x in 0..4 {
            assert!(set.insert(BlockPos::new(x, 0, 0)));
        }

        assert!(set.remove(BlockPos::new(1, 0, 0)));

        assert_eq!(
            set.java_ordered_positions(),
            [
                BlockPos::new(0, 0, 0),
                BlockPos::new(2, 0, 0),
                BlockPos::new(3, 0, 0)
            ]
        );
    }

    #[test]
    fn reinserted_position_uses_new_insertion_position() {
        let mut set = JavaBlockPosSet::default();
        let first = BlockPos::new(1, 0, 0);
        let second = BlockPos::new(17, 0, 0);

        assert!(set.insert(first));
        assert!(set.insert(second));
        assert!(set.remove(first));
        assert!(set.insert(first));

        assert_eq!(set.java_ordered_positions(), [second, first]);
    }

    #[test]
    fn pop_uses_insertion_order() {
        let mut set = JavaBlockPosSet::default();
        let first = BlockPos::new(1, 0, 0);
        let second = BlockPos::new(2, 0, 0);
        assert!(set.insert(first));
        assert!(set.insert(second));

        assert_eq!(set.pop_java_ordered_position(), Some(first));
        assert_eq!(set.pop_java_ordered_position(), Some(second));
        assert_eq!(set.pop_java_ordered_position(), None);
    }
}
