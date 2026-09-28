//! PyO3 bindings for the `pareto` module. Mirrors the public Python surface
//! of `autoresearch.core.pareto` so existing imports keep working unchanged.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::pareto::{ObjectiveVector as RustObjectiveVector, Trial as RustTrial};

/// Python-facing handle for `ObjectiveVector`. All four axes are `Optional`
/// (None default), mirroring the `@dataclass(frozen=True)` semantics.
#[pyclass(name = "ObjectiveVector", frozen)]
#[derive(Debug, Clone, Default)]
pub struct PyObjectiveVector {
    inner: RustObjectiveVector,
}

#[pymethods]
impl PyObjectiveVector {
    #[new]
    #[pyo3(signature = (ctx=None, tps=None, agentic=None, coding=None))]
    fn new(ctx: Option<f64>, tps: Option<f64>, agentic: Option<f64>, coding: Option<f64>) -> Self {
        Self {
            inner: RustObjectiveVector {
                ctx,
                tps,
                agentic,
                coding,
            },
        }
    }

    #[getter]
    fn ctx(&self) -> Option<f64> {
        self.inner.ctx
    }

    #[getter]
    fn tps(&self) -> Option<f64> {
        self.inner.tps
    }

    #[getter]
    fn agentic(&self) -> Option<f64> {
        self.inner.agentic
    }

    #[getter]
    fn coding(&self) -> Option<f64> {
        self.inner.coding
    }

    /// Mirrors the Python `complete` property. `True` iff all four axes are
    /// populated.
    #[getter]
    fn complete(&self) -> bool {
        self.inner.complete()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObjectiveVector(ctx={:?}, tps={:?}, agentic={:?}, coding={:?})",
            self.inner.ctx, self.inner.tps, self.inner.agentic, self.inner.coding,
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = other.downcast::<PyObjectiveVector>()?;
        Ok(self.inner == other.get().inner)
    }
}

/// Python-facing handle for `Trial` (fp, vector).
#[pyclass(name = "Trial", frozen)]
#[derive(Debug, Clone, Default)]
pub struct PyTrial {
    inner: RustTrial,
}

#[pymethods]
impl PyTrial {
    #[new]
    fn new(fp: String, vector: &Bound<'_, PyObjectiveVector>) -> PyResult<Self> {
        let inner = RustTrial {
            fp,
            vector: vector.get().inner.clone(),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn fp(&self) -> String {
        self.inner.fp.clone()
    }

    #[getter]
    fn vector(&self, py: Python<'_>) -> PyResult<Py<PyObjectiveVector>> {
        Py::new(
            py,
            PyObjectiveVector {
                inner: self.inner.vector.clone(),
            },
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Trial(fp={:?}, vector={:?})",
            self.inner.fp, self.inner.vector
        )
    }
}

/// `pareto.fingerprint(engine, sampler) -> str` — sha256 hex over the
/// canonical JSON of `{"engine": …, "sampler": …}` (sorted keys, compact
/// separators).
#[pyfunction]
#[pyo3(signature = (engine, sampler=None))]
fn fingerprint(
    engine: &Bound<'_, PyDict>,
    sampler: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let engine_value = pydict_to_json(engine)?;
    let sampler_value = match sampler {
        Some(s) => pydict_to_json(s)?,
        None => serde_json::Value::Null,
    };
    Ok(crate::pareto::fingerprint_hash(
        &engine_value,
        &sampler_value,
    ))
}

/// `pareto.dominates(a, b) -> bool` — 4-axis strict dominance.
#[pyfunction]
fn dominates(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let a_v = to_objective_vector(a)?;
    let b_v = to_objective_vector(b)?;
    Ok(crate::pareto::dominates(&a_v, &b_v))
}

/// `pareto.pareto_set(vectors) -> list[ObjectiveVector]` — non-dominated
/// subset, preserves order. Accepts any iterable (list, generator, ...).
#[pyfunction]
fn pareto_set<'py>(py: Python<'py>, vectors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut rust: Vec<RustObjectiveVector> = Vec::new();
    for item in vectors.iter()? {
        let item = item?;
        rust.push(to_objective_vector(&item)?);
    }
    let result = crate::pareto::pareto_set(&rust);
    let out = PyList::empty_bound(py);
    for v in result {
        let py_v = wrap_objective_vector(py, v)?;
        out.append(py_v)?;
    }
    Ok(out)
}

/// `pareto.merge(trials) -> list[Trial]` — per-fingerprint merge, taking
/// the maximum value of each axis. Accepts any iterable.
#[pyfunction]
fn merge<'py>(py: Python<'py>, trials: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut rust: Vec<RustTrial> = Vec::new();
    for item in trials.iter()? {
        let item = item?;
        rust.push(to_trial(&item)?);
    }
    let result = crate::pareto::merge(&rust);
    let out = PyList::empty_bound(py);
    for t in result {
        let py_t = wrap_trial(py, t)?;
        out.append(py_t)?;
    }
    Ok(out)
}

// Helpers for Python ↔ Rust conversion.

fn pydict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    pythonize_to_json(dict)
}

fn pythonize_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::from(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(serde_json::Value::from(u));
    }
    if let Ok(f) = obj.extract::<f64>() {
        let n = serde_json::Number::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("non-finite float"))?;
        return Ok(serde_json::Value::Number(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(pythonize_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        for (key, value) in dict.iter() {
            let k = key.extract::<String>()?;
            map.insert(k, pythonize_to_json(&value)?);
        }
        return Ok(serde_json::Value::Object(
            map.into_iter().collect::<serde_json::Map<String, _>>(),
        ));
    }
    Err(PyValueError::new_err(format!(
        "unsupported Python type for pareto JSON conversion: {}",
        obj.get_type().name()?
    )))
}

fn to_objective_vector(obj: &Bound<'_, PyAny>) -> PyResult<RustObjectiveVector> {
    let py_obj: &PyObjectiveVector = obj.downcast::<PyObjectiveVector>()?.get();
    Ok(py_obj.inner.clone())
}

fn to_trial(obj: &Bound<'_, PyAny>) -> PyResult<RustTrial> {
    let py_obj: &PyTrial = obj.downcast::<PyTrial>()?.get();
    Ok(py_obj.inner.clone())
}

fn wrap_objective_vector<'py>(
    py: Python<'py>,
    v: RustObjectiveVector,
) -> PyResult<Bound<'py, PyObjectiveVector>> {
    Bound::new(py, PyObjectiveVector { inner: v })
}

fn wrap_trial<'py>(py: Python<'py>, t: RustTrial) -> PyResult<Bound<'py, PyTrial>> {
    Bound::new(py, PyTrial { inner: t })
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    parent.add_class::<PyObjectiveVector>()?;
    parent.add_class::<PyTrial>()?;
    parent.add_function(wrap_pyfunction!(fingerprint, parent)?)?;
    parent.add_function(wrap_pyfunction!(dominates, parent)?)?;
    parent.add_function(wrap_pyfunction!(pareto_set, parent)?)?;
    parent.add_function(wrap_pyfunction!(merge, parent)?)?;
    Ok(())
}
