//! Core data types mirroring the Python `@dataclass(frozen=True)` in
//! `autoresearch/core/pareto.py`. Public PyO3 bindings expose
//! `ObjectiveVector` (`#[pyclass]` frozen + `complete` property) so the
//! Python tests and consumers keep working unchanged.

/// 4-axis objective vector. All axes are `Optional` (None = not measured
/// yet). A vector is `complete` only when every axis is populated.
///
/// `ctx` is modelled as `f64` (instead of `u32`) so the Python shape
/// `int | None` accepts both `int` and `float` consistently — the project
/// uses `ctx` as a memory budget and a float value never loses precision
/// for the values used in practice (1024–131072).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveVector {
    pub ctx: Option<f64>,
    pub tps: Option<f64>,
    pub agentic: Option<f64>,
    pub coding: Option<f64>,
}

impl Default for ObjectiveVector {
    fn default() -> Self {
        Self {
            ctx: None,
            tps: None,
            agentic: None,
            coding: None,
        }
    }
}

impl ObjectiveVector {
    #[must_use]
    pub fn complete(&self) -> bool {
        self.ctx.is_some() && self.tps.is_some() && self.agentic.is_some() && self.coding.is_some()
    }
}

/// Trial summary used by `merge(trials)`. Mirrors the Python `Trial` —
/// `(fp: str, vector: ObjectiveVector)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub fp: String,
    pub vector: ObjectiveVector,
}

impl Default for Trial {
    fn default() -> Self {
        Self {
            fp: String::new(),
            vector: ObjectiveVector::default(),
        }
    }
}

/// Type-erased vector — Python uses a `Protocol` here. Public helper trait.
pub trait VectorLike {
    fn ctx(&self) -> Option<f64>;
    fn tps(&self) -> Option<f64>;
    fn agentic(&self) -> Option<f64>;
    fn coding(&self) -> Option<f64>;
}

impl VectorLike for ObjectiveVector {
    fn ctx(&self) -> Option<f64> {
        self.ctx
    }
    fn tps(&self) -> Option<f64> {
        self.tps
    }
    fn agentic(&self) -> Option<f64> {
        self.agentic
    }
    fn coding(&self) -> Option<f64> {
        self.coding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector_is_incomplete() {
        let v = ObjectiveVector::default();
        assert!(!v.complete());
    }

    #[test]
    fn partially_populated_vector_is_incomplete() {
        let v = ObjectiveVector {
            ctx: Some(1024.0),
            tps: None,
            agentic: None,
            coding: None,
        };
        assert!(!v.complete());
    }

    #[test]
    fn fully_populated_vector_is_complete() {
        let v = ObjectiveVector {
            ctx: Some(1024.0),
            tps: Some(20.0),
            agentic: Some(0.5),
            coding: Some(0.4),
        };
        assert!(v.complete());
    }

    #[test]
    fn trial_default_is_empty() {
        let t = Trial::default();
        assert_eq!(t.fp, "");
        assert!(!t.vector.complete());
    }
}
