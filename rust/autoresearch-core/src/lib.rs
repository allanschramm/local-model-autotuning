//! Native core for `ailocal-model-autotuning`. Phase 0 ports three Python
//! modules (`fingerprint`, `state`, `pareto`) to pure Rust and exposes them
//! to Python through PyO3 (`#[pymodule] autoresearch_core`). Phase 1+ adds
//! the `autoresearch-loop` binary that wraps the rust-native subprocess
//! pipeline.
//!
//! See `rust/README.md` for build instructions and the Phase 0 acceptance
//! gates; see `…/artifacts/plan.md` for the strategic roadmap.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod fingerprint;
pub mod pareto;
pub mod state;

#[cfg(any(feature = "pyo3", not(feature = "no-pyo3")))]
pub mod py;

/// Crate version constant (mirrors `Cargo.toml` package version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library MSRV (matches the `rust-version` field in `Cargo.toml`).
pub const MSRV: &str = "1.82";

#[cfg(test)]
mod smoke {
    use super::*;

    #[test]
    fn version_is_set() {
        // Sanity check the version is propagated.
        assert!(VERSION.starts_with("0."));
    }

    #[test]
    fn msrv_is_documented() {
        // MSRV string must round-trip; tests fail loudly if it is removed.
        assert_eq!(MSRV, "1.82");
    }
}
