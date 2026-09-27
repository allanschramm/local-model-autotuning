//! Extract the Pareto set (non-dominated subset) from a list of
//! `ObjectiveVector`s, preserving insertion order. Incomplete vectors are
//! dropped silently — matching the Python contract.

use crate::pareto::{dominates, ObjectiveVector};

#[must_use]
pub fn pareto_set(vectors: &[ObjectiveVector]) -> Vec<ObjectiveVector> {
    let complete: Vec<ObjectiveVector> = vectors.iter().filter(|v| v.complete()).cloned().collect();
    let mut out = Vec::with_capacity(complete.len());
    for (i, cand) in complete.iter().enumerate() {
        let mut dominated = false;
        for (j, other) in complete.iter().enumerate() {
            if i == j {
                continue;
            }
            if dominates(other, cand) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            out.push(cand.clone());
        }
    }
    out
}

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
    fn empty_input_yields_empty_set() {
        let out = pareto_set(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn single_complete_vector_is_kept() {
        let input = vec![v(1024.0, 50.0, 0.5, 0.5)];
        assert_eq!(pareto_set(&input).len(), 1);
    }

    #[test]
    fn incomplete_vector_dropped_silently() {
        let input = vec![ObjectiveVector::default()];
        assert!(pareto_set(&input).is_empty());
    }

    #[test]
    fn two_dominating_vectors_keeps_only_one() {
        let input = vec![v(1024.0, 50.0, 0.5, 0.5), v(2048.0, 100.0, 0.6, 0.6)];
        let out = pareto_set(&input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ctx, Some(2048.0));
    }
}
