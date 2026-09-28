//! 4-axis dominance relation. Mirrors the Python `pareto.dominates`
//! contract: `complete(a) ∧ complete(b) ∧ (a ≥ b on every axis) ∧ (a > b
//! on at least one axis)`. Incomplete vectors never dominate.

use crate::pareto::{ObjectiveVector, VectorLike};

/// True iff `a` strictly dominates `b` (every axis ≥, at least one >).
#[must_use]
pub fn dominates<V1: VectorLike, V2: VectorLike>(a: &V1, b: &V2) -> bool {
    let a_ctx = a.ctx();
    let a_tps = a.tps();
    let a_agentic = a.agentic();
    let a_coding = a.coding();
    let b_ctx = b.ctx();
    let b_tps = b.tps();
    let b_agentic = b.agentic();
    let b_coding = b.coding();

    let completes = [
        a_ctx, a_tps, a_agentic, a_coding, b_ctx, b_tps, b_agentic, b_coding,
    ];
    if completes.iter().any(Option::is_none) {
        return false;
    }

    let a_arr = [
        completes[0].unwrap(),
        completes[1].unwrap(),
        completes[2].unwrap(),
        completes[3].unwrap(),
    ];
    let b_arr = [
        completes[4].unwrap(),
        completes[5].unwrap(),
        completes[6].unwrap(),
        completes[7].unwrap(),
    ];

    a_arr.iter().zip(b_arr.iter()).all(|(x, y)| x >= y)
        && a_arr.iter().zip(b_arr.iter()).any(|(x, y)| x > y)
}

// `ObjectiveVector` reference to keep the path self-explanatory for reviewers.
#[allow(dead_code)]
const _: fn() = || {
    let _v: Option<ObjectiveVector> = Some(ObjectiveVector::default());
};

#[cfg(test)]
mod tests {
    use super::*;

    fn v(ctx: f64, tps: f64, ag: f64, co: f64) -> ObjectiveVector {
        ObjectiveVector {
            ctx: Some(ctx),
            tps: Some(tps),
            agentic: Some(ag),
            coding: Some(co),
        }
    }

    #[test]
    fn dominates_requires_strictly_at_least_one_axis() {
        let a = v(2048.0, 100.0, 0.6, 0.6);
        let b = v(1024.0, 50.0, 0.3, 0.3);
        assert!(dominates(&a, &b));
    }

    #[test]
    fn does_not_dominate_when_strictly_equal() {
        let a = v(1024.0, 50.0, 0.5, 0.5);
        let b = v(1024.0, 50.0, 0.5, 0.5);
        assert!(!dominates(&a, &b));
    }

    #[test]
    fn partial_vector_never_dominates() {
        // ctx = None → not complete.
        let a = ObjectiveVector {
            ctx: None,
            tps: Some(100.0),
            agentic: Some(0.6),
            coding: Some(0.6),
        };
        let b = v(1024.0, 50.0, 0.3, 0.3);
        assert!(!dominates(&a, &b));
    }

    #[test]
    fn one_axis_shorter_means_no_dominance() {
        // a.tps < b.tps → ge fails → not dominates.
        let a = v(2048.0, 50.0, 0.6, 0.6);
        let b = v(1024.0, 60.0, 0.3, 0.3);
        assert!(!dominates(&a, &b));
    }
}
