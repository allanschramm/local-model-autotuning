//! Native pareto module — port of `autoresearch/core/pareto.py`.
//!
//! Pure-Rust implementation of the 4-axis dominance relation, the Pareto
//! set extraction, the merge operation, and the canonical fingerprint
//! (sha256 over canonical JSON of engine + sampler dicts).

pub mod dominates;
pub mod fingerprint_hash;
pub mod merge;
pub mod pareto_set;
pub mod types;

pub use dominates::dominates;
pub use fingerprint_hash::fingerprint_hash;
pub use merge::merge;
pub use pareto_set::pareto_set;
pub use types::{ObjectiveVector, Trial, VectorLike};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_re_exports_match_python_names() {
        // The Python `from autoresearch.core.pareto import …` imports look up
        // each of these. The names must remain stable across the port.
        let _name: fn(&ObjectiveVector, &ObjectiveVector) -> bool = dominates;
        let _name2: fn(&serde_json::Value, &serde_json::Value) -> String =
            fingerprint_hash;
        let _name3: fn(&[Trial]) -> Vec<Trial> = merge;
        let _name4: fn(&[ObjectiveVector]) -> Vec<ObjectiveVector> = pareto_set;
    }
}
