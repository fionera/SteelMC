//! Surface rule JSON parsing and transpilation.
//!
//! Parses surface rule trees from `noise_settings/{dimension}.json` and generates
//! a `try_apply_surface_rule()` function per dimension that inlines all conditions
//! and block outputs as Rust code.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Deserialize;
use std::{mem, slice};

// ── JSON types ──────────────────────────────────────────────────────────────

/// Surface rule source (top-level rule node).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum SurfaceRuleJson {
    #[serde(rename = "minecraft:block")]
    Block { result_state: ResultStateJson },
    #[serde(rename = "minecraft:sequence")]
    Sequence { sequence: Vec<SurfaceRuleJson> },
    #[serde(rename = "minecraft:condition")]
    Condition {
        if_true: SurfaceConditionJson,
        then_run: Box<SurfaceRuleJson>,
    },
    #[serde(rename = "minecraft:bandlands")]
    Bandlands {},
}

/// Block state reference in a surface rule.
///
/// Currently only uses the block name (all vanilla surface rule blocks use
/// default state). If modded surface rules need non-default block states,
/// add a `Properties` field and wire it through the transpiler.
#[derive(Debug, Clone, Deserialize)]
pub struct ResultStateJson {
    #[serde(rename = "Name")]
    pub name: String,
}

/// Surface rule condition.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum SurfaceConditionJson {
    #[serde(rename = "minecraft:stone_depth")]
    StoneDepth {
        offset: i32,
        add_surface_depth: bool,
        secondary_depth_range: i32,
        surface_type: String,
    },
    #[serde(rename = "minecraft:above_preliminary_surface")]
    AbovePreliminarySurface {},
    #[serde(rename = "minecraft:biome")]
    BiomeIs { biome_is: SingleOrList<BiomeIdJson> },
    #[serde(rename = "minecraft:noise_threshold")]
    NoiseThreshold {
        noise: String,
        #[serde(default)]
        is_3d: bool,
        min_threshold: f64,
        max_threshold: f64,
    },
    #[serde(rename = "minecraft:vertical_gradient")]
    VerticalGradient {
        random_name: String,
        true_at_and_below: VerticalAnchorJson,
        false_at_and_above: VerticalAnchorJson,
    },
    #[serde(rename = "minecraft:y_above")]
    YAbove {
        anchor: VerticalAnchorJson,
        surface_depth_multiplier: i32,
        add_stone_depth: bool,
    },
    #[serde(rename = "minecraft:water")]
    Water {
        offset: i32,
        surface_depth_multiplier: i32,
        add_stone_depth: bool,
    },
    #[serde(rename = "minecraft:temperature")]
    Temperature {},
    #[serde(rename = "minecraft:steep")]
    Steep {},
    #[serde(rename = "minecraft:hole")]
    Hole {},
    #[serde(rename = "minecraft:not")]
    Not { invert: Box<SurfaceConditionJson> },
}

/// Biome reference — plain string biome ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct BiomeIdJson(String);

/// Vanilla holder-set JSON accepts either a single ID or a list of IDs.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SingleOrList<T> {
    Single(T),
    List(Vec<T>),
}

impl<T> SingleOrList<T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Single(value) => slice::from_ref(value),
            Self::List(values) => values,
        }
    }
}

impl BiomeIdJson {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Vertical anchor for Y-level resolution.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum VerticalAnchorJson {
    Absolute { absolute: i32 },
    AboveBottom { above_bottom: i32 },
    BelowTop { below_top: i32 },
}

// ── Transpiler ──────────────────────────────────────────────────────────────

/// Context for surface rule transpilation.
pub struct SurfaceRuleTranspiler {
    /// Collected noise IDs referenced by `NoiseThreshold` conditions.
    pub noise_ids: Vec<String>,
    /// Collected random IDs referenced by `VerticalGradient` conditions.
    pub gradient_ids: Vec<String>,
    /// Collected block state names returned by block-result rules.
    pub block_state_names: Vec<String>,
    /// Whether generated conditions read `ctx.biome_id` directly or indirectly.
    pub uses_biome: bool,
    /// Whether generated conditions use `ctx.min_surface_level`.
    pub uses_preliminary_surface: bool,
    /// Whether generated conditions use `ctx.surface_secondary`.
    pub uses_surface_secondary: bool,
    /// Whether generated conditions use `ctx.steep`.
    pub uses_steep: bool,
    /// The largest depth any ceiling-type `StoneDepth` condition compares
    /// `ctx.stone_depth_below` against, or `None` when some comparison bound is
    /// not a compile-time constant.
    ///
    /// Every such condition has the shape `stone_depth_below <= bound`, so a
    /// depth greater than the largest bound is indistinguishable from any other
    /// depth greater than it, and the caller need not count past that. `Some(0)`
    /// means the value is never read at all.
    pub stone_depth_below_bound: Option<i32>,
    /// Facts held true for the copy currently being emitted.
    facts: SurfaceRuleFacts,
    /// Biomes named by `BiomeIs` conditions that survived the current pass.
    referenced_biomes: Vec<String>,
    /// Whether the current pass emitted a condition reading the downward scan's
    /// running state -- stone depth or water height -- rather than position and
    /// per-column values alone.
    reads_scan_state: bool,
    /// Min Y for this dimension.
    min_y: i32,
    /// Height for this dimension.
    height: i32,
}

/// What a specialized copy of a surface rule is allowed to assume.
///
/// Every fact here has to be established by the caller before it uses the copy;
/// the transpiler folds the matching conditions away and emits neither the test
/// nor the branch behind it.
#[derive(Debug, Default, Clone)]
pub struct SurfaceRuleFacts {
    /// Known value of `above_preliminary_surface`, if the caller knows it.
    above_preliminary_surface: Option<bool>,
    /// Biomes known not to occur anywhere the copy will be applied.
    absent_biomes: Vec<String>,
}

/// A transpiled condition, folded to a constant where the facts decide it.
enum Cond {
    Known(bool),
    Expr(TokenStream),
}

impl SurfaceRuleTranspiler {
    pub fn new(min_y: i32, height: i32, uses_preliminary_surface: bool) -> Self {
        Self {
            noise_ids: Vec::new(),
            gradient_ids: Vec::new(),
            block_state_names: Vec::new(),
            uses_biome: false,
            uses_preliminary_surface,
            uses_surface_secondary: false,
            uses_steep: false,
            stone_depth_below_bound: Some(0),
            facts: SurfaceRuleFacts::default(),
            referenced_biomes: Vec::new(),
            reads_scan_state: false,
            min_y,
            height,
        }
    }

    /// Widen the recorded `stone_depth_below` bound to cover one more condition.
    ///
    /// `None` is absorbing: once one condition compares against a runtime value
    /// the exact depth is needed and no later constant bound can take that back.
    /// The bound is otherwise the maximum, and is only ever widened, so passes
    /// that fold conditions away -- the deep-band copies -- cannot narrow what
    /// the full rule established.
    fn note_stone_depth_below_bound(&mut self, bound: Option<i32>) {
        self.stone_depth_below_bound = match (self.stone_depth_below_bound, bound) {
            (Some(seen), Some(bound)) => Some(seen.max(bound)),
            _ => None,
        };
    }

    /// Transpile a surface rule tree into a Rust function body.
    ///
    /// Generated code references `ctx: &mut SurfaceRuleContext` from `steel_utils`.
    pub fn transpile_rule(&mut self, rule: &SurfaceRuleJson) -> TokenStream {
        match rule {
            SurfaceRuleJson::Block { result_state } => {
                let block_name = result_state.name.as_str();
                let block_state_index = if let Some(idx) = self
                    .block_state_names
                    .iter()
                    .position(|name| name == block_name)
                {
                    idx
                } else {
                    let idx = self.block_state_names.len();
                    self.block_state_names.push(block_name.to_owned());
                    idx
                };
                quote! {
                    return Some(ctx.block_state(#block_state_index));
                }
            }
            SurfaceRuleJson::Sequence { sequence } => {
                let stmts: Vec<_> = sequence.iter().map(|r| self.transpile_rule(r)).collect();
                quote! { #(#stmts)* }
            }
            SurfaceRuleJson::Condition { if_true, then_run } => {
                // The condition is evaluated first even when it folds, so that a
                // pass still registers the noise, gradient and block-state ids
                // its live siblings share. Every surface condition is a pure
                // function of position and per-column values -- the vertical
                // gradients take a positional random, the noises are positional
                // and memoized per block -- so dropping one changes nothing but
                // the work done.
                match self.transpile_condition(if_true) {
                    Cond::Known(false) => TokenStream::new(),
                    Cond::Known(true) => self.transpile_rule(then_run),
                    Cond::Expr(cond) => {
                        let body = self.transpile_rule(then_run);
                        if body.is_empty() {
                            TokenStream::new()
                        } else {
                            quote! {
                                if #cond {
                                    #body
                                }
                            }
                        }
                    }
                }
            }
            SurfaceRuleJson::Bandlands {} => {
                quote! {
                    return Some(ctx.system.get_band(ctx.block_x, ctx.block_y, ctx.block_z));
                }
            }
        }
    }

    /// Transpile a condition into a boolean expression.
    #[expect(
        clippy::too_many_lines,
        reason = "surface condition variants are best kept in one dispatch function"
    )]
    fn transpile_condition(&mut self, cond: &SurfaceConditionJson) -> Cond {
        Cond::Expr(match cond {
            SurfaceConditionJson::StoneDepth {
                offset,
                add_surface_depth,
                secondary_depth_range,
                surface_type,
            } => {
                self.reads_scan_state = true;
                let is_floor = surface_type == "floor";
                let depth_field = if is_floor {
                    quote! { ctx.stone_depth_above }
                } else {
                    // Ceiling depth is counted by the caller one block at a
                    // time, so record how far it has to count. A bound that is
                    // not a compile-time constant poisons the answer and the
                    // caller falls back to counting the run out in full.
                    let bound = (!*add_surface_depth && *secondary_depth_range <= 0)
                        .then(|| (1 + *offset).max(0));
                    self.note_stone_depth_below_bound(bound);
                    quote! { ctx.stone_depth_below }
                };

                if *secondary_depth_range > 0 {
                    self.uses_surface_secondary = true;
                    let range = *secondary_depth_range;
                    if *add_surface_depth {
                        quote! {
                            {
                                let extra = ((ctx.surface_secondary + 1.0) / 2.0 * #range as f64) as i32;
                                #depth_field <= 1 + #offset + ctx.surface_depth + extra
                            }
                        }
                    } else {
                        quote! {
                            {
                                let extra = ((ctx.surface_secondary + 1.0) / 2.0 * #range as f64) as i32;
                                #depth_field <= 1 + #offset + extra
                            }
                        }
                    }
                } else if *add_surface_depth {
                    quote! { #depth_field <= 1 + #offset + ctx.surface_depth }
                } else {
                    quote! { #depth_field <= 1 + #offset }
                }
            }
            SurfaceConditionJson::AbovePreliminarySurface {} => {
                self.uses_preliminary_surface = true;
                if let Some(known) = self.facts.above_preliminary_surface {
                    return Cond::Known(known);
                }
                quote! { ctx.block_y >= ctx.min_surface_level }
            }
            SurfaceConditionJson::BiomeIs { biome_is } => {
                self.uses_biome = true;
                let named: Vec<&str> = biome_is.as_slice().iter().map(BiomeIdJson::as_str).collect();
                if !named.is_empty()
                    && named
                        .iter()
                        .all(|name| self.facts.absent_biomes.iter().any(|absent| absent == name))
                {
                    return Cond::Known(false);
                }
                for name in named {
                    if !self.referenced_biomes.iter().any(|seen| seen == name) {
                        self.referenced_biomes.push(name.to_owned());
                    }
                }
                let checks: Vec<_> = biome_is
                    .as_slice()
                    .iter()
                    .map(|b| {
                        let biome_name = b
                            .as_str()
                            .strip_prefix("minecraft:")
                            .unwrap_or(b.as_str());
                        let upper = biome_name.to_uppercase();
                        let biome_ident = Ident::new(&upper, Span::call_site());
                        quote! { biome_id == steel_registry::RegistryEntry::id(&*steel_registry::vanilla_biomes::#biome_ident) as u16 }
                    })
                    .collect();
                let check = if checks.is_empty() {
                    quote! { false }
                } else if checks.len() == 1 {
                    let mut checks = checks;
                    checks.remove(0)
                } else {
                    quote! { ( #(#checks)||* ) }
                };
                let biome_id = if self.uses_preliminary_surface {
                    quote! { ctx.biome_id() }
                } else {
                    quote! { ctx.known_biome_id() }
                };
                quote! { #biome_id.is_some_and(|biome_id| #check) }
            }
            SurfaceConditionJson::NoiseThreshold {
                noise,
                is_3d,
                min_threshold,
                max_threshold,
            } => {
                let noise_key = noise.clone();
                let noise_index =
                    if let Some(idx) = self.noise_ids.iter().position(|k| k == &noise_key) {
                        idx
                    } else {
                        let idx = self.noise_ids.len();
                        self.noise_ids.push(noise_key);
                        idx
                    };
                let min_f = *min_threshold;
                let max_f = *max_threshold;
                let sample = if *is_3d {
                    quote! { ctx.condition_noise_3d(#noise_index) }
                } else {
                    quote! { ctx.condition_noise(#noise_index) }
                };
                quote! {
                    {
                        let v = #sample;
                        v >= #min_f && v <= #max_f
                    }
                }
            }
            SurfaceConditionJson::VerticalGradient {
                random_name,
                true_at_and_below,
                false_at_and_above,
            } => {
                let gradient_index =
                    if let Some(idx) = self.gradient_ids.iter().position(|id| id == random_name) {
                        idx
                    } else {
                        let idx = self.gradient_ids.len();
                        self.gradient_ids.push(random_name.to_owned());
                        idx
                    };
                let true_y = self.resolve_anchor(true_at_and_below);
                let false_y = self.resolve_anchor(false_at_and_above);
                quote! { ctx.system.vertical_gradient(#gradient_index, ctx.block_x, ctx.block_y, ctx.block_z, #true_y, #false_y) }
            }
            SurfaceConditionJson::YAbove {
                anchor,
                surface_depth_multiplier,
                add_stone_depth,
            } => {
                // Vanilla: blockY + (addStoneDepth ? stoneDepthAbove : 0)
                //            >= anchor + surfaceDepth * multiplier
                let anchor_y = self.resolve_anchor(anchor);
                let mul = *surface_depth_multiplier;
                if *add_stone_depth {
                    self.reads_scan_state = true;
                    quote! {
                        ctx.block_y + ctx.stone_depth_above >= #anchor_y + ctx.surface_depth * #mul
                    }
                } else {
                    quote! {
                        ctx.block_y >= #anchor_y + ctx.surface_depth * #mul
                    }
                }
            }
            SurfaceConditionJson::Water {
                offset,
                surface_depth_multiplier,
                add_stone_depth,
            } => {
                // Vanilla: waterHeight == MIN_VALUE
                //   || blockY + (addStoneDepth ? stoneDepthAbove : 0)
                //        >= waterHeight + offset + surfaceDepth * multiplier
                self.reads_scan_state = true;
                let mul = *surface_depth_multiplier;
                if *add_stone_depth {
                    quote! {
                        ctx.water_height == i32::MIN
                            || ctx.block_y + ctx.stone_depth_above >= ctx.water_height + #offset + ctx.surface_depth * #mul
                    }
                } else {
                    quote! {
                        ctx.water_height == i32::MIN
                            || ctx.block_y >= ctx.water_height + #offset + ctx.surface_depth * #mul
                    }
                }
            }
            SurfaceConditionJson::Temperature {} => {
                self.uses_biome = true;
                quote! { ctx.cold_enough_to_snow() }
            }
            SurfaceConditionJson::Steep {} => {
                self.uses_steep = true;
                quote! { ctx.steep }
            }
            SurfaceConditionJson::Hole {} => {
                quote! { ctx.surface_depth <= 0 }
            }
            SurfaceConditionJson::Not { invert } => match self.transpile_condition(invert) {
                Cond::Known(known) => return Cond::Known(!known),
                Cond::Expr(inner) => quote! { !(#inner) },
            },
        })
    }

    /// Evaluates a condition against the current facts without emitting code.
    ///
    /// Mirrors the folding `transpile_condition` does, so the two agree on which
    /// branches are dead.
    fn const_condition(&self, cond: &SurfaceConditionJson) -> Option<bool> {
        match cond {
            SurfaceConditionJson::AbovePreliminarySurface {} => {
                self.facts.above_preliminary_surface
            }
            SurfaceConditionJson::BiomeIs { biome_is } => {
                let named = biome_is.as_slice();
                (!named.is_empty()
                    && named.iter().all(|b| {
                        self.facts
                            .absent_biomes
                            .iter()
                            .any(|absent| absent == b.as_str())
                    }))
                .then_some(false)
            }
            SurfaceConditionJson::Not { invert } => self.const_condition(invert).map(|known| !known),
            _ => None,
        }
    }

    /// The `block_y` at and above which `cond` is always false, if it has one.
    fn condition_false_at_and_above(&self, cond: &SurfaceConditionJson) -> Option<i32> {
        match cond {
            SurfaceConditionJson::VerticalGradient {
                false_at_and_above, ..
            } => Some(self.resolve_anchor(false_at_and_above)),
            _ => None,
        }
    }

    /// The `block_y` at and above which `rule` never returns a block.
    ///
    /// `None` when no such bound can be proven, which is the answer for any rule
    /// that can write at an unbounded height. The bound lets the caller skip the
    /// rule outright over a whole span of a column rather than calling it once
    /// per block to be told `None`.
    fn write_ceiling(&self, rule: &SurfaceRuleJson) -> Option<i32> {
        match rule {
            SurfaceRuleJson::Block { .. } | SurfaceRuleJson::Bandlands {} => None,
            SurfaceRuleJson::Sequence { sequence } => {
                let mut highest = i32::MIN;
                for inner in sequence {
                    highest = highest.max(self.write_ceiling(inner)?);
                }
                Some(highest)
            }
            SurfaceRuleJson::Condition { if_true, then_run } => {
                if self.const_condition(if_true) == Some(false) {
                    // Dead branch: it writes nowhere, so it bounds nothing.
                    return Some(i32::MIN);
                }
                let body = self.write_ceiling(then_run);
                match self.condition_false_at_and_above(if_true) {
                    Some(bound) => Some(body.map_or(bound, |body| body.min(bound))),
                    None => body,
                }
            }
        }
    }

    /// Resolve a vertical anchor to a constant Y value.
    fn resolve_anchor(&self, anchor: &VerticalAnchorJson) -> i32 {
        match anchor {
            VerticalAnchorJson::Absolute { absolute } => *absolute,
            VerticalAnchorJson::AboveBottom { above_bottom } => self.min_y + above_bottom,
            VerticalAnchorJson::BelowTop { below_top } => self.height - 1 + self.min_y - below_top,
        }
    }
}

fn rule_uses_preliminary_surface(rule: &SurfaceRuleJson) -> bool {
    match rule {
        SurfaceRuleJson::Block { .. } | SurfaceRuleJson::Bandlands {} => false,
        SurfaceRuleJson::Sequence { sequence } => {
            sequence.iter().any(rule_uses_preliminary_surface)
        }
        SurfaceRuleJson::Condition { if_true, then_run } => {
            condition_uses_preliminary_surface(if_true) || rule_uses_preliminary_surface(then_run)
        }
    }
}

fn condition_uses_preliminary_surface(condition: &SurfaceConditionJson) -> bool {
    match condition {
        SurfaceConditionJson::AbovePreliminarySurface {} => true,
        SurfaceConditionJson::Not { invert } => condition_uses_preliminary_surface(invert),
        SurfaceConditionJson::StoneDepth { .. }
        | SurfaceConditionJson::BiomeIs { .. }
        | SurfaceConditionJson::NoiseThreshold { .. }
        | SurfaceConditionJson::VerticalGradient { .. }
        | SurfaceConditionJson::YAbove { .. }
        | SurfaceConditionJson::Water { .. }
        | SurfaceConditionJson::Temperature {}
        | SurfaceConditionJson::Steep {}
        | SurfaceConditionJson::Hole {} => false,
    }
}

/// Everything the density-function generator needs from one dimension's surface rule.
pub struct SurfaceRuleFunctionArtifacts {
    /// The generated `apply_surface_rule_impl`, plus the deep-band copy if there is one.
    pub functions: TokenStream,
    pub noise_ids: Vec<String>,
    pub gradient_ids: Vec<String>,
    pub block_state_names: Vec<String>,
    pub uses_biome: bool,
    pub uses_preliminary_surface: bool,
    pub uses_surface_secondary: bool,
    pub uses_steep: bool,
    /// Largest `stone_depth_below` any condition distinguishes, or `None` when
    /// the rule needs the exact depth. See the transpiler field of the same name.
    pub stone_depth_below_bound: Option<i32>,
    /// The deep-band specialization, absent when the rule does not admit one.
    pub deep_band: Option<DeepBandArtifacts>,
}

/// A copy of the surface rule specialized to below the preliminary surface.
///
/// Below that level the rule is a much smaller function: everything guarded by
/// `above_preliminary_surface` is gone, and so is everything that only fires in
/// a biome the caller has proven cannot occur nearby. What survives in vanilla's
/// overworld is two vertical gradients, which read nothing but position -- no
/// biome lookup, no scan state, no preliminary surface.
pub struct DeepBandArtifacts {
    /// Biomes the caller must prove absent before using the specialization.
    pub required_absent_biomes: Vec<String>,
    /// `block_y` at and above which the specialization never writes, so the
    /// caller can skip it over that whole span instead of calling it per block.
    pub write_ceiling: i32,
}

pub fn generate_surface_rule_function(
    rule: &SurfaceRuleJson,
    min_y: i32,
    height: i32,
) -> SurfaceRuleFunctionArtifacts {
    let uses_preliminary_surface = rule_uses_preliminary_surface(rule);
    let mut transpiler = SurfaceRuleTranspiler::new(min_y, height, uses_preliminary_surface);
    let body = transpiler.transpile_rule(rule);

    // The deep-band copy is emitted second so that it can only reuse ids the
    // full rule already registered; folding conditions away never introduces a
    // new noise, gradient or block state.
    let deep = uses_preliminary_surface
        .then(|| generate_deep_band(&mut transpiler, rule))
        .flatten();

    let noise_ids = mem::take(&mut transpiler.noise_ids);
    let gradient_ids = mem::take(&mut transpiler.gradient_ids);
    let block_state_names = mem::take(&mut transpiler.block_state_names);
    let uses_biome = transpiler.uses_biome;
    let uses_preliminary_surface = transpiler.uses_preliminary_surface;
    let uses_surface_secondary = transpiler.uses_surface_secondary;
    let uses_steep = transpiler.uses_steep;
    let stone_depth_below_bound = transpiler.stone_depth_below_bound;

    let deep_function = deep.as_ref().map(|(_, body)| {
        quote! {
            /// Apply this dimension's surface rule below the preliminary surface.
            ///
            /// Only valid where `ctx.block_y < ctx.min_surface_level` and none of
            /// `surface_deep_band_absent_biomes()` occur near the position.
            #[allow(clippy::collapsible_if, clippy::needless_return, clippy::erasing_op, unused_comparisons)]
            fn apply_surface_rule_deep_impl(
                ctx: &mut steel_worldgen::surface::SurfaceRuleContext<'_>,
            ) -> Option<steel_utils::BlockStateId> {
                #body
                None
            }
        }
    });

    let functions = quote! {
        /// Apply this dimension's surface rule at the current context position.
        #[allow(clippy::collapsible_if, clippy::needless_return, clippy::erasing_op, unused_comparisons)]
        fn apply_surface_rule_impl(
            ctx: &mut steel_worldgen::surface::SurfaceRuleContext<'_>,
        ) -> Option<steel_utils::BlockStateId> {
            #body
            None
        }

        #deep_function
    };

    SurfaceRuleFunctionArtifacts {
        functions,
        noise_ids,
        gradient_ids,
        block_state_names,
        uses_biome,
        uses_preliminary_surface,
        uses_surface_secondary,
        uses_steep,
        stone_depth_below_bound,
        deep_band: deep.map(|(artifacts, _)| artifacts),
    }
}

/// Emits the below-preliminary-surface copy of `rule`, if one is worth having.
///
/// Two passes. The first establishes only that `above_preliminary_surface` is
/// false and records which biomes the surviving conditions still test; the
/// second additionally assumes those biomes absent, which is what collapses the
/// rule. Returns `None` when the result would still read the downward scan's
/// running state, or when no height bound can be proven -- in both cases the
/// caller is better off with the full rule.
fn generate_deep_band(
    transpiler: &mut SurfaceRuleTranspiler,
    rule: &SurfaceRuleJson,
) -> Option<(DeepBandArtifacts, TokenStream)> {
    transpiler.facts = SurfaceRuleFacts {
        above_preliminary_surface: Some(false),
        absent_biomes: Vec::new(),
    };
    transpiler.referenced_biomes.clear();
    let _ = transpiler.transpile_rule(rule);
    let required_absent_biomes = mem::take(&mut transpiler.referenced_biomes);

    transpiler.facts = SurfaceRuleFacts {
        above_preliminary_surface: Some(false),
        absent_biomes: required_absent_biomes.clone(),
    };
    transpiler.referenced_biomes.clear();
    transpiler.reads_scan_state = false;
    let body = transpiler.transpile_rule(rule);
    let reads_scan_state = transpiler.reads_scan_state;
    let write_ceiling = transpiler.write_ceiling(rule);
    transpiler.facts = SurfaceRuleFacts::default();
    transpiler.reads_scan_state = false;
    transpiler.referenced_biomes.clear();

    if reads_scan_state {
        return None;
    }
    let write_ceiling = write_ceiling?;

    Some((
        DeepBandArtifacts {
            required_absent_biomes,
            write_ceiling,
        },
        body,
    ))
}
