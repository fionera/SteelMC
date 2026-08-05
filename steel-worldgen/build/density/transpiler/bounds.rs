//! Static bounds analysis for density function subtrees.
//!
//! Computes conservative `(lower, upper)` intervals used during codegen to elide
//! unreachable branches (for example `min`/`max` when one operand is already
//! bounded). Resolves `Reference` nodes through the build-time registry.

use std::collections::BTreeSet;

use crate::density::{
    CubicSpline, DensityFunction, MappedType, MarkerType, SplineValue, TwoArgType,
};

use super::TranspilerInput;

/// Static (lower, upper) bounds for a density function subtree.
///
/// Returned bounds satisfy `lower <= eval(df) <= upper` at runtime for all
/// inputs the function can be sampled at. When tight bounds aren't derivable
/// (e.g., free-form noise with unknown amplitude product, or potentially
/// unbounded operations like reciprocal), the corresponding side is set to
/// `f64::NEG_INFINITY` / `f64::INFINITY` and downstream short-circuit
/// optimizations correctly fall through to the unconditional codegen.
///
/// Mirrors the static-bounds analysis used by C2ME's
/// `MaxShortNode`/`MinShortNode` rewriters, with one extension: we resolve
/// `Reference` nodes through the build-time registry so cross-function
/// bounds propagate.
pub(super) fn compute_bounds(df: &DensityFunction, input: &TranspilerInput) -> (f64, f64) {
    compute_bounds_inner(df, input, &mut Vec::new())
}

#[expect(
    clippy::too_many_lines,
    reason = "one match arm per DensityFunction variant; splitting the dispatch would obscure the per-variant bounds analysis"
)]
pub(super) fn compute_bounds_inner(
    df: &DensityFunction,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
) -> (f64, f64) {
    match df {
        DensityFunction::Constant(c) => (c.value, c.value),

        DensityFunction::Reference(r) => {
            // Avoid infinite recursion through self-referential cycles (shouldn't
            // happen in practice, but DF graphs are cycle-free only by convention).
            if visiting.iter().any(|n| n == &r.id) {
                return (f64::NEG_INFINITY, f64::INFINITY);
            }
            let Some(target) = input.registry.get(&r.id) else {
                return (f64::NEG_INFINITY, f64::INFINITY);
            };
            visiting.push(r.id.clone());
            let bounds = compute_bounds_inner(target, input, visiting);
            visiting.pop();
            bounds
        }

        DensityFunction::YClampedGradient(g) => {
            let lo = g.from_value.min(g.to_value);
            let hi = g.from_value.max(g.to_value);
            (lo, hi)
        }

        DensityFunction::Noise(_)
        | DensityFunction::ShiftedNoise(_)
        | DensityFunction::ShiftA(_)
        | DensityFunction::ShiftB(_)
        | DensityFunction::Shift(_)
        | DensityFunction::Spline(_)
        | DensityFunction::BlendedNoise(_) => (f64::NEG_INFINITY, f64::INFINITY),

        DensityFunction::TwoArgumentSimple(t) => {
            let (a_lo, a_hi) = compute_bounds_inner(&t.argument1, input, visiting);
            let (b_lo, b_hi) = compute_bounds_inner(&t.argument2, input, visiting);
            match t.op {
                TwoArgType::Add => (a_lo + b_lo, a_hi + b_hi),
                TwoArgType::Mul => {
                    // Interval arithmetic for sign-mixed multiplication.
                    let candidates = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi];
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for c in candidates {
                        if c.is_nan() {
                            return (f64::NEG_INFINITY, f64::INFINITY);
                        }
                        if c < lo {
                            lo = c;
                        }
                        if c > hi {
                            hi = c;
                        }
                    }
                    (lo, hi)
                }
                TwoArgType::Min => (a_lo.min(b_lo), a_hi.min(b_hi)),
                TwoArgType::Max => (a_lo.max(b_lo), a_hi.max(b_hi)),
            }
        }

        DensityFunction::Mapped(m) => {
            let (lo, hi) = compute_bounds_inner(&m.input, input, visiting);
            match m.op {
                MappedType::Abs => {
                    if lo >= 0.0 {
                        (lo, hi)
                    } else if hi <= 0.0 {
                        (-hi, -lo)
                    } else {
                        (0.0, lo.abs().max(hi.abs()))
                    }
                }
                MappedType::Square => {
                    if lo >= 0.0 {
                        (lo * lo, hi * hi)
                    } else if hi <= 0.0 {
                        (hi * hi, lo * lo)
                    } else {
                        (0.0, (lo * lo).max(hi * hi))
                    }
                }
                MappedType::Cube => {
                    // x^3 is monotone over the whole real line, so endpoints suffice.
                    (lo * lo * lo, hi * hi * hi)
                }
                MappedType::HalfNegative => {
                    // `if v > 0 { v } else { v * 0.5 }` — monotone non-decreasing
                    // (slope 0.5 below 0, slope 1 above 0).
                    let map = |v: f64| if v > 0.0 { v } else { v * 0.5 };
                    (map(lo), map(hi))
                }
                MappedType::QuarterNegative => {
                    let map = |v: f64| if v > 0.0 { v } else { v * 0.25 };
                    (map(lo), map(hi))
                }
                MappedType::Invert => {
                    // 1/v is unbounded near 0; only safe if input doesn't straddle 0.
                    if lo > 0.0 || hi < 0.0 {
                        let a = 1.0 / lo;
                        let b = 1.0 / hi;
                        (a.min(b), a.max(b))
                    } else {
                        (f64::NEG_INFINITY, f64::INFINITY)
                    }
                }
                MappedType::Squeeze => {
                    // clamp(-1, 1) → c/2 - c³/24. Endpoints: -1/2 + 1/24, 1/2 - 1/24.
                    let map = |v: f64| {
                        let c = v.clamp(-1.0, 1.0);
                        c / 2.0 - c * c * c / 24.0
                    };
                    let lo_c = lo.clamp(-1.0, 1.0);
                    let hi_c = hi.clamp(-1.0, 1.0);
                    (map(lo_c), map(hi_c))
                }
            }
        }

        DensityFunction::Clamp(c) => (c.min, c.max),

        DensityFunction::RangeChoice(rc) => {
            let (in_lo, in_hi) = compute_bounds_inner(&rc.when_in_range, input, visiting);
            let (out_lo, out_hi) = compute_bounds_inner(&rc.when_out_of_range, input, visiting);
            (in_lo.min(out_lo), in_hi.max(out_hi))
        }

        DensityFunction::IntervalSelect(interval) => {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for function in &interval.functions {
                let (function_lo, function_hi) = compute_bounds_inner(function, input, visiting);
                lo = lo.min(function_lo);
                hi = hi.max(function_hi);
            }
            if lo > hi {
                (f64::NEG_INFINITY, f64::INFINITY)
            } else {
                (lo, hi)
            }
        }

        DensityFunction::WeirdScaledSampler(_) => {
            // result = scale * noise.abs() where scale ∈ [0.5, 3.0] and
            // noise.abs() is non-negative. The upper bound is noise-parameter
            // dependent, so leave it unbounded for branch-elision purposes.
            (0.0, f64::INFINITY)
        }

        DensityFunction::EndIslands => (-100.0, 80.0),

        DensityFunction::BlendAlpha(_) => (1.0, 1.0),
        DensityFunction::BlendOffset(_) => (0.0, 0.0),
        DensityFunction::BlendDensity(bd) => compute_bounds_inner(&bd.input, input, visiting),

        DensityFunction::Marker(m) => compute_bounds_inner(&m.wrapped, input, visiting),

        DensityFunction::FindTopSurface(fts) => {
            // Returns a Y coordinate in [lower_bound, upper_bound rounded down].
            // upper_bound is itself a DF — its static upper bound caps the result.
            let (_, upper) = compute_bounds_inner(&fts.upper_bound, input, visiting);
            (f64::from(fts.lower_bound), upper)
        }
    }
}

/// Whether `final_density <= 0` is implied by its first interpolated channel
/// being `<= 0`.
///
/// This is what lets `NoiseChunk::fill` prove a whole vertical run is air from
/// one channel, without evaluating the combine per block. Each channel is affine
/// in y between its two cell corners, so the run's channel interval is exact;
/// this function supplies the second half of the argument — that a non-positive
/// channel 0 forces a non-positive density.
///
/// Deliberately structural and conservative: it recognizes only operations whose
/// sign-at-zero behaviour is provable, and returns `false` for anything else, so
/// an upstream JSON change degrades to "never skip" rather than to wrong terrain.
///
/// Sound because, for `x <= 0`:
/// - `squeeze(x) = clamp(x,-1,1)/2 - clamp(x,-1,1)^3/24` is monotone increasing
///   with `squeeze(0) = 0`, hence `<= 0`;
/// - `min(a, b) <= a`, so one non-positive operand suffices;
/// - the marker itself is the channel.
pub(super) fn density_nonpositive_when_first_channel_nonpositive(
    df: &DensityFunction,
    input: &TranspilerInput,
) -> bool {
    nonpositive_inner(df, input, &mut Vec::new(), &mut true)
}

fn nonpositive_inner(
    df: &DensityFunction,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
    first_marker: &mut bool,
) -> bool {
    match df {
        // The interpolated channel itself. Only the first one encountered is
        // channel 0, which is the one the runtime test bounds.
        DensityFunction::Marker(m) if m.kind == MarkerType::Interpolated => {
            std::mem::take(first_marker)
        }
        // Other markers are transparent wrappers.
        DensityFunction::Marker(m) => nonpositive_inner(&m.wrapped, input, visiting, first_marker),
        DensityFunction::Mapped(m) if m.op == MappedType::Squeeze => {
            nonpositive_inner(&m.input, input, visiting, first_marker)
        }
        DensityFunction::TwoArgumentSimple(t) if t.op == TwoArgType::Min => {
            // `min` needs only one non-positive side, but the channel must be
            // reachable through it, so try each in turn.
            let mut first = *first_marker;
            if nonpositive_inner(&t.argument1, input, visiting, &mut first) {
                *first_marker = first;
                return true;
            }
            let mut second = *first_marker;
            if nonpositive_inner(&t.argument2, input, visiting, &mut second) {
                *first_marker = second;
                return true;
            }
            false
        }
        DensityFunction::Reference(r) => {
            if visiting.iter().any(|name| name == &r.id) {
                return false;
            }
            let Some(resolved) = input.registry.get(&r.id) else {
                return false;
            };
            visiting.push(r.id.clone());
            let result = nonpositive_inner(resolved, input, visiting, first_marker);
            visiting.pop();
            result
        }
        _ => false,
    }
}

// ── Blended-noise reach analysis ────────────────────────────────────────────

/// How far a `BlendedNoise` leaf's influence reaches up a subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlendedReach {
    /// No `BlendedNoise` occurs anywhere in the subtree, so its value never
    /// depends on one.
    Absent,
    /// `BlendedNoise` occurs below, but for every `y >= .0` this subtree
    /// evaluates to the same bits whatever value the blended noise takes.
    NeutralAtOrAbove(i32),
    /// `BlendedNoise` occurs below and for every `y >= .0` this subtree
    /// evaluates to a zero — but possibly `-0.0` rather than `+0.0`, because
    /// the sign follows the annihilated operand. The zero must still be
    /// absorbed before the sign can be forgotten.
    SignedZeroAtOrAbove(i32),
    /// Nothing proven: the blended noise may reach the value at any `y`.
    Live,
}

/// Combine child reaches for a node that is an arbitrary pure function of its
/// children.
///
/// `SignedZeroAtOrAbove` degrades to [`BlendedReach::Live`] here: only the
/// `Add` rule below knows how to discharge the sign, and other operations
/// (`abs`, `cube`, a comparison, …) can turn `-0.0` and `+0.0` into different
/// results.
fn combine_reach(children: impl IntoIterator<Item = BlendedReach>) -> BlendedReach {
    let mut acc = BlendedReach::Absent;
    for child in children {
        acc = match (acc, child) {
            (BlendedReach::Live | BlendedReach::SignedZeroAtOrAbove(_), _)
            | (_, BlendedReach::Live | BlendedReach::SignedZeroAtOrAbove(_)) => BlendedReach::Live,
            (BlendedReach::Absent, BlendedReach::Absent) => BlendedReach::Absent,
            (BlendedReach::Absent, BlendedReach::NeutralAtOrAbove(y))
            | (BlendedReach::NeutralAtOrAbove(y), BlendedReach::Absent) => {
                BlendedReach::NeutralAtOrAbove(y)
            }
            (BlendedReach::NeutralAtOrAbove(a), BlendedReach::NeutralAtOrAbove(b)) => {
                BlendedReach::NeutralAtOrAbove(a.max(b))
            }
        };
    }
    acc
}

/// The Y at or above which this node evaluates to exactly a zero, if it is a
/// `y_clamped_gradient` that slides down to zero.
///
/// `map_clamped(y, from_y, to_y, from_value, 0.0)` is exact at and above
/// `to_y`:
/// - for `y > to_y` the factor `t` exceeds 1 and the clamp returns `to_value`
///   itself, a zero;
/// - at `y == to_y` the factor is `(to_y - from_y) / (to_y - from_y)`, i.e.
///   exactly `1.0` (both operands are the same non-zero f64), so the lerp is
///   `from_value + 1.0 * (0.0 - from_value)` — and `x + (-x)` is `+0.0` for
///   every finite `x`.
///
/// Both the scalar `map_clamped` and the SIMD mask-select form the transpiler
/// emits compute exactly this, so the result holds for either code path.
fn zeroing_gradient_at_or_above(
    df: &DensityFunction,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
) -> Option<i32> {
    match df {
        DensityFunction::Marker(m) => zeroing_gradient_at_or_above(&m.wrapped, input, visiting),
        DensityFunction::Reference(r) => {
            if visiting.iter().any(|name| name == &r.id) {
                return None;
            }
            let target = input.registry.get(&r.id)?;
            visiting.push(r.id.clone());
            let result = zeroing_gradient_at_or_above(target, input, visiting);
            visiting.pop();
            result
        }
        DensityFunction::YClampedGradient(g) => {
            // `to_value == 0.0` matches both `+0.0` and `-0.0`; either way the
            // product below is a zero, and its sign is already treated as
            // unknown by `SignedZeroAtOrAbove`.
            (g.to_value == 0.0 && g.from_y < g.to_y && g.from_value.is_finite()).then_some(g.to_y)
        }
        _ => None,
    }
}

/// Whether a subtree provably evaluates to a finite (non-NaN, non-infinite)
/// value at every position.
///
/// Needed because the annihilation argument rests on `0.0 * v` being a zero,
/// which fails when `v` is infinite or NaN. Deliberately structural and
/// conservative: leaves whose runtime implementation carries an explicit finite
/// bound are accepted, and anything able to divide by a non-constant is
/// rejected.
fn is_finite_valued(
    df: &DensityFunction,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
) -> bool {
    match df {
        DensityFunction::Constant(c) => c.value.is_finite(),

        // Clamped lerp between two finite endpoints. `from_y != to_y` keeps the
        // interpolation factor's divisor non-zero.
        DensityFunction::YClampedGradient(g) => {
            g.from_y != g.to_y && g.from_value.is_finite() && g.to_value.is_finite()
        }

        // `NormalNoise` carries an explicit finite `max_value` (its
        // `expected_deviation` is bounded away from zero) and reads gradients
        // from a fixed table, so every sampler is finite by construction.
        DensityFunction::Noise(_)
        | DensityFunction::ShiftA(_)
        | DensityFunction::ShiftB(_)
        | DensityFunction::Shift(_) => true,
        DensityFunction::ShiftedNoise(sn) => {
            is_finite_valued(&sn.shift_x, input, visiting)
                && is_finite_valued(&sn.shift_y, input, visiting)
                && is_finite_valued(&sn.shift_z, input, visiting)
        }

        // Ends in `clamped_lerp(a, b, t) / 128.0` over finite endpoints.
        DensityFunction::BlendedNoise(_) => true,

        // Transpiled as constants / bounded helpers.
        DensityFunction::EndIslands
        | DensityFunction::BlendAlpha(_)
        | DensityFunction::BlendOffset(_) => true,

        DensityFunction::Marker(m) => is_finite_valued(&m.wrapped, input, visiting),
        DensityFunction::BlendDensity(bd) => is_finite_valued(&bd.input, input, visiting),
        DensityFunction::Clamp(c) => {
            c.min.is_finite() && c.max.is_finite() && is_finite_valued(&c.input, input, visiting)
        }

        // `Invert` is `1.0 / v` — the one unary op that can divide by zero.
        DensityFunction::Mapped(m) => {
            m.op != MappedType::Invert && is_finite_valued(&m.input, input, visiting)
        }

        DensityFunction::TwoArgumentSimple(t) => {
            is_finite_valued(&t.argument1, input, visiting)
                && is_finite_valued(&t.argument2, input, visiting)
        }

        DensityFunction::RangeChoice(rc) => {
            is_finite_valued(&rc.input, input, visiting)
                && is_finite_valued(&rc.when_in_range, input, visiting)
                && is_finite_valued(&rc.when_out_of_range, input, visiting)
        }

        DensityFunction::IntervalSelect(sel) => {
            is_finite_valued(&sel.input, input, visiting)
                && sel
                    .functions
                    .iter()
                    .all(|f| is_finite_valued(f, input, visiting))
        }

        // Rarity mappers are inlined as if-chains of constant multipliers, and
        // the sampled noise is a bounded `NormalNoise`.
        DensityFunction::WeirdScaledSampler(ws) => is_finite_valued(&ws.input, input, visiting),

        // Returns a Y coordinate between a finite lower bound and its
        // upper-bound function.
        DensityFunction::FindTopSurface(fts) => {
            is_finite_valued(&fts.density, input, visiting)
                && is_finite_valued(&fts.upper_bound, input, visiting)
        }

        DensityFunction::Spline(s) => spline_is_finite(&s.spline, input, visiting),

        DensityFunction::Reference(r) => {
            if visiting.iter().any(|name| name == &r.id) {
                return false;
            }
            let Some(target) = input.registry.get(&r.id) else {
                return false;
            };
            visiting.push(r.id.clone());
            let result = is_finite_valued(target, input, visiting);
            visiting.pop();
            result
        }
    }
}

/// Whether a cubic spline evaluates finitely: finite coordinate, finite point
/// data, and strictly increasing locations so no segment divides by a
/// zero-width span.
fn spline_is_finite(
    spline: &CubicSpline,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
) -> bool {
    if !is_finite_valued(&spline.coordinate, input, visiting) {
        return false;
    }
    if spline
        .points
        .windows(2)
        .any(|w| w[0].location >= w[1].location)
    {
        return false;
    }
    spline.points.iter().all(|p| {
        p.location.is_finite()
            && p.derivative.is_finite()
            && match &p.value {
                SplineValue::Constant(v) => v.is_finite(),
                SplineValue::Spline(nested) => spline_is_finite(nested, input, visiting),
            }
    })
}

/// Compute how far blended noise reaches up a single subtree.
fn blended_reach(
    df: &DensityFunction,
    input: &TranspilerInput,
    visiting: &mut Vec<String>,
) -> BlendedReach {
    match df {
        DensityFunction::BlendedNoise(_) => BlendedReach::Live,

        DensityFunction::Constant(_)
        | DensityFunction::YClampedGradient(_)
        | DensityFunction::Noise(_)
        | DensityFunction::ShiftA(_)
        | DensityFunction::ShiftB(_)
        | DensityFunction::Shift(_)
        | DensityFunction::EndIslands
        | DensityFunction::BlendAlpha(_)
        | DensityFunction::BlendOffset(_) => BlendedReach::Absent,

        DensityFunction::Marker(m) => blended_reach(&m.wrapped, input, visiting),
        DensityFunction::BlendDensity(bd) => blended_reach(&bd.input, input, visiting),

        // Pure unary functions of one child: a value that no longer depends on
        // the blended noise stays independent through them.
        DensityFunction::Clamp(c) => combine_reach([blended_reach(&c.input, input, visiting)]),
        DensityFunction::Mapped(m) => combine_reach([blended_reach(&m.input, input, visiting)]),
        DensityFunction::WeirdScaledSampler(ws) => {
            combine_reach([blended_reach(&ws.input, input, visiting)])
        }

        DensityFunction::ShiftedNoise(sn) => combine_reach([
            blended_reach(&sn.shift_x, input, visiting),
            blended_reach(&sn.shift_y, input, visiting),
            blended_reach(&sn.shift_z, input, visiting),
        ]),

        DensityFunction::RangeChoice(rc) => combine_reach([
            blended_reach(&rc.input, input, visiting),
            blended_reach(&rc.when_in_range, input, visiting),
            blended_reach(&rc.when_out_of_range, input, visiting),
        ]),

        DensityFunction::IntervalSelect(sel) => combine_reach(
            std::iter::once(blended_reach(&sel.input, input, visiting)).chain(
                sel.functions
                    .iter()
                    .map(|f| blended_reach(f, input, visiting)),
            ),
        ),

        // A spline's value depends on its coordinate in ways this analysis does
        // not model, and `FindTopSurface` samples its density at many Y values
        // rather than the current one — so neither can carry a per-Y neutrality
        // claim upward. Accept them only when no blended noise is involved.
        DensityFunction::Spline(_) | DensityFunction::FindTopSurface(_) => {
            if super::graph::has_blended_noise(df, &input.registry, &mut BTreeSet::new()) {
                BlendedReach::Live
            } else {
                BlendedReach::Absent
            }
        }

        DensityFunction::TwoArgumentSimple(t) => {
            let reach1 = blended_reach(&t.argument1, input, visiting);
            let reach2 = blended_reach(&t.argument2, input, visiting);

            match t.op {
                // `gradient * v` is a zero at and above the gradient's `to_y`
                // whenever `v` is finite — whatever `v` is, blended noise and
                // all. That is what erases the blended-noise contribution.
                TwoArgType::Mul => {
                    for (gradient, other) in
                        [(&t.argument1, &t.argument2), (&t.argument2, &t.argument1)]
                    {
                        if let Some(threshold) =
                            zeroing_gradient_at_or_above(gradient, input, visiting)
                            && is_finite_valued(other, input, visiting)
                        {
                            return BlendedReach::SignedZeroAtOrAbove(threshold);
                        }
                    }
                    combine_reach([reach1, reach2])
                }

                // `x + (±0.0)` is exactly `x` when `x` is finite and non-zero,
                // which discharges the unknown sign of the annihilated product.
                TwoArgType::Add => {
                    for (zeroed, sibling, sibling_reach) in [
                        (reach1, &t.argument2, reach2),
                        (reach2, &t.argument1, reach1),
                    ] {
                        if let BlendedReach::SignedZeroAtOrAbove(threshold) = zeroed
                            && sibling_reach == BlendedReach::Absent
                            && is_nonzero_finite(sibling, input)
                        {
                            return BlendedReach::NeutralAtOrAbove(threshold);
                        }
                    }
                    combine_reach([reach1, reach2])
                }

                TwoArgType::Min | TwoArgType::Max => combine_reach([reach1, reach2]),
            }
        }

        DensityFunction::Reference(r) => {
            if visiting.iter().any(|name| name == &r.id) {
                return BlendedReach::Live;
            }
            let Some(target) = input.registry.get(&r.id) else {
                return BlendedReach::Live;
            };
            visiting.push(r.id.clone());
            let result = blended_reach(target, input, visiting);
            visiting.pop();
            result
        }
    }
}

/// Whether a subtree's static bounds prove it is finite and never zero.
fn is_nonzero_finite(df: &DensityFunction, input: &TranspilerInput) -> bool {
    let (lo, hi) = compute_bounds(df, input);
    lo.is_finite() && hi.is_finite() && (lo > 0.0 || hi < 0.0)
}

/// The Y at or above which blended noise cannot affect *any* interpolated
/// channel, if there is one.
///
/// `inners` must be exactly the `Interpolated` marker inner trees the
/// transpiler emits into `fill_cell_corner_densities`, because those are the
/// only expressions that read the hoisted `blended_noise_value` parameter that
/// `NoiseChunk` fills from the blended-noise column.
///
/// Returns `None` — disabling the optimization — unless every channel either
/// contains no blended noise at all or provably ignores it at and above a
/// common Y. `None` reproduces the unoptimized behaviour exactly.
pub(super) fn blended_noise_irrelevant_at_or_above_y(
    inners: &[DensityFunction],
    input: &TranspilerInput,
) -> Option<i32> {
    let mut threshold: Option<i32> = None;
    for inner in inners {
        match blended_reach(inner, input, &mut Vec::new()) {
            // No blended noise in this channel: no constraint on the column.
            BlendedReach::Absent => {}
            BlendedReach::NeutralAtOrAbove(y) => {
                threshold = Some(threshold.map_or(y, |t: i32| t.max(y)));
            }
            BlendedReach::SignedZeroAtOrAbove(_) | BlendedReach::Live => return None,
        }
    }
    threshold
}
