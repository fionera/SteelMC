use std::sync::OnceLock;

use rustc_hash::{FxHashMap, FxHashSet};
use steel_utils::Identifier;

#[derive(Debug, Default)]
pub struct RegistryTags {
    tags: FxHashMap<Identifier, RegistryTag>,
}

#[derive(Debug)]
struct RegistryTag {
    ordered_keys: Vec<&'static Identifier>,
    /// Registry ids of the members, in the same order as `ordered_keys`.
    ///
    /// Resolved on first use, and only once the registry is frozen. Iterating a
    /// tag is then a slice walk and a `Vec` index per member; resolving by key
    /// instead is a string-keyed hash lookup per member -- seven of them for a
    /// six-member tag -- and `RuleTest::TagMatch` in the ore feature does that on
    /// the order of a hundred million times over a 601x601 pregeneration. It is
    /// the whole of `__memcmp_evex_movbe` at 0.9% of the machine.
    ///
    /// Resolution waits for the freeze because registering an entry whose key is
    /// already taken appends a *new* id and remaps the key, so a tag's membership
    /// follows the replacement. Ids captured before that would point at the entry
    /// the key no longer resolves to.
    resolved_ids: OnceLock<Vec<usize>>,
    member_keys: FxHashSet<&'static Identifier>,
}

impl RegistryTag {
    fn new(ordered_keys: Vec<&'static Identifier>) -> Self {
        let member_keys = ordered_keys.iter().copied().collect();
        Self {
            ordered_keys,
            resolved_ids: OnceLock::new(),
            member_keys,
        }
    }
}

impl RegistryTags {
    #[doc(hidden)]
    pub fn insert(&mut self, tag: Identifier, ordered_keys: Vec<&'static Identifier>) {
        self.tags.insert(tag, RegistryTag::new(ordered_keys));
    }

    #[doc(hidden)]
    pub fn remove(&mut self, tag: &Identifier) -> Option<Vec<&'static Identifier>> {
        self.tags.remove(tag).map(|tag| tag.ordered_keys)
    }

    #[doc(hidden)]
    #[must_use]
    pub fn contains(&self, tag: &Identifier, entry_key: &Identifier) -> bool {
        self.tags
            .get(tag)
            .is_some_and(|entries| entries.member_keys.contains(entry_key))
    }

    #[doc(hidden)]
    #[must_use]
    pub fn get(&self, tag: &Identifier) -> Option<&[&'static Identifier]> {
        self.tags
            .get(tag)
            .map(|entries| entries.ordered_keys.as_slice())
    }

    /// Registry ids of a tag's members, resolving and memoizing them on first
    /// use once the registry is frozen.
    ///
    /// Returns `None` while the registry still allows registering, because a
    /// later duplicate-key registration would move a member to a new id.
    #[doc(hidden)]
    #[must_use]
    pub fn ids(
        &self,
        tag: &Identifier,
        frozen: bool,
        id_of: impl Fn(&Identifier) -> Option<usize>,
    ) -> Option<&[usize]> {
        if !frozen {
            return None;
        }
        self.tags.get(tag).map(|entries| {
            entries
                .resolved_ids
                .get_or_init(|| entries.ordered_keys.iter().filter_map(|key| id_of(key)).collect())
                .as_slice()
        })
    }

    #[doc(hidden)]
    pub fn keys(&self) -> impl Iterator<Item = &Identifier> + '_ {
        self.tags.keys()
    }
}
