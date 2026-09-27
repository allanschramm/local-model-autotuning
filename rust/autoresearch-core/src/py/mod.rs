//! PyO3 bindings surface. Phase 0 Sprint 0.2/0.3/0.4 fill in the binding
//! modules; this scaffold exposes the module so `cargo build` succeeds.

use pyo3::prelude::*;

mod fingerprint;
mod pareto;
mod state;

/// Module entry point. Bound as `autoresearch_core` in Python
/// (after the maturin wheel is installed).
#[pymodule]
fn autoresearch_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;

    let pareto_module = PyModule::new_bound(py, "pareto")?;
    pareto::register(py, &pareto_module)?;
    m.add_submodule(&pareto_module)?;
    modules.set_item("autoresearch_core.pareto", &pareto_module)?;

    let fingerprint_module = PyModule::new_bound(py, "fingerprint")?;
    fingerprint::register(py, &fingerprint_module)?;
    m.add_submodule(&fingerprint_module)?;
    modules.set_item("autoresearch_core.fingerprint", &fingerprint_module)?;

    let state_module = PyModule::new_bound(py, "state")?;
    state::register(py, &state_module)?;
    m.add_submodule(&state_module)?;
    modules.set_item("autoresearch_core.state", &state_module)?;

    Ok(())
}
