//! Merge `Trial`s by fingerprint (the `fp` field), keeping the maximum
//! value of each axis. The resulting list is ordered by fingerprint
//! (deterministic, byte-stable).
//!
//! Mirrors the Python `pareto.merge(trials)` contract: only one merged
//! `Trial` per `fp`; axes filled in across Trials; partial vectors stay
//! partial until all four axes are measured.

use crate::pareto::{ObjectiveVector, Trial};
use std::collections::BTreeMap;

#[must_use]
pub fn merge(trials: &[Trial]) -> Vec<Trial> {
    let mut grouped: BTreeMap<String, ObjectiveVector> = BTreeMap::new();

    for t in trials {
        let entry = grouped
            .entry(t.fp.clone())
            .or_insert_with(ObjectiveVector::default);
        entry.ctx = pick_max_f64(entry.ctx, t.vector.ctx);
        entry.tps = pick_max_f64(entry.tps, t.vector.tps);
        entry.agentic = pick_max_f64(entry.agentic, t.vector.agentic);
        entry.coding = pick_max_f64(entry.coding, t.vector.coding);
    }

    grouped
        .into_iter()
        .map(|(fp, vector)| Trial { fp, vector })
        .collect()
}

fn pick_max_f64(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (None, Some(y)) => Some(y),
        (Some(x), None) => Some(x),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(ctx: Option<f64>, tps: Option<f64>) -> ObjectiveVector {
        ObjectiveVector {
            ctx,
            tps,
            agentic: None,
            coding: None,
        }
    }

    fn t(fp: &str, vector: ObjectiveVector) -> Trial {
        Trial {
            fp: fp.into(),
            vector,
        }
    }

    #[test]
    fn empty_input_yields_empty_list() {
        let out = merge(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn same_fingerprint_collapses_to_one_with_max_axes() {
        let input = vec![
            t("fp1", v(Some(1024.0), Some(50.0))),
            t("fp1", v(Some(2048.0), Some(40.0))),
        ];
        let out = merge(&input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fp, "fp1");
        assert_eq!(out[0].vector.ctx, Some(2048.0));
        assert_eq!(out[0].vector.tps, Some(50.0));
    }

    #[test]
    fn partial_vector_stays_partial_until_full() {
        let input = vec![
            t("fp1", v(Some(1024.0), Some(50.0))),
            t("fp1", v(None, None)),
        ];
        let out = merge(&input);
        assert!(!out[0].vector.complete());
    }

    #[test]
    fn different_fingerprints_stay_distinct() {
        let input = vec![
            t("fp_a", v(Some(1024.0), Some(50.0))),
            t("fp_b", v(Some(2048.0), Some(40.0))),
        ];
        assert_eq!(merge(&input).len(), 2);
    }
}
