//! PyO3 bindings for the `state` module — Phase 0 Sprint 0.3.
//! Mirrors `autoresearch.core.state.SearchState`. Baseline I/O stays
//! Python-side; the visited/morris memory + atomic JSON persistence are
//! pure Rust.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use serde_json::{Map, Value};

use crate::error::{CoreError, CoreResult};
use crate::state::SearchState as RustSearchState;

/// Python-facing handle for `autoresearch.core.state.SearchState`.
#[pyclass(name = "SearchState")]
#[derive(Debug, Clone)]
pub struct PySearchState {
    inner: RustSearchState,
}

#[pymethods]
impl PySearchState {
    /// `__init__(state_path=None)` — when `state_path` is `None`, the
    /// Python shim fills in `config.STATE_FILE`.
    #[new]
    fn new(state_path: Option<&str>) -> PyResult<Self> {
        let path = match state_path {
            Some(s) => std::path::PathBuf::from(s),
            None => return Err(PyValueError::new_err(
                "SearchState requires an explicit state_path; use the Python shim for the config.STATE_FILE default",
            )),
        };
        let inner = RustSearchState::load(path).map_err(core_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn state_path(&self) -> String {
        self.inner.path.to_string_lossy().to_string()
    }

    #[getter]
    fn visited<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty_bound(py);
        for v in &self.inner.visited {
            list.append(v.clone())?;
        }
        Ok(list)
    }

    fn is_visited(&self, key: &str) -> bool {
        self.inner.is_visited(key)
    }

    fn mark_visited(&mut self, key: &str, persist: Option<bool>) -> PyResult<()> {
        let persist = persist.unwrap_or(true);
        self.inner.mark_visited(key, persist).map_err(core_to_py)
    }

    fn morris_pins_for<'py>(&self, py: Python<'py>, model: &str) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in self.inner.morris_pins_for(model) {
            dict.set_item(k, json_to_python(py, &v))?;
        }
        Ok(dict)
    }

    fn set_morris<'py>(
        &mut self,
        py: Python<'py>,
        model: &str,
        pins: &Bound<'py, PyDict>,
        effects: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let pins_map = pythonize_obj_map(pins)?;
        let effects_map = pythonize_obj_map(effects)?;
        self.inner
            .set_morris(model, pins_map, effects_map)
            .map_err(core_to_py)
    }

    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(core_to_py)
    }

    fn persist(&self) -> PyResult<()> {
        self.inner.persist().map_err(core_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchState(visited={}, morris_models={}, path={:?})",
            self.inner.visited.len(),
            self.inner.morris.len(),
            self.inner.path
        )
    }
}

fn core_to_py(e: CoreError) -> PyErr {
    use pyo3::exceptions::PyValueError;
    PyValueError::new_err(e.to_string())
}

fn pythonize_obj_map(obj: &Bound<'_, PyDict>) -> PyResult<Map<String, Value>> {
    let mut map: Map<String, Value> = Map::new();
    for (key, value) in obj.iter() {
        let k = key.extract::<String>()?;
        map.insert(k, pythonize_value(&value)?);
    }
    Ok(map)
}

fn pythonize_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::from(u));
    }
    if let Ok(f) = obj.extract::<f64>() {
        let n = serde_json::Number::from_f64(f)
            .ok_or_else(|| PyValueError::new_err("non-finite float"))?;
        return Ok(Value::Number(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(pythonize_value(&item)?);
        }
        return Ok(Value::Array(arr));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return Ok(Value::Object(pythonize_obj_map(dict)?));
    }
    Err(PyValueError::new_err(format!(
        "unsupported type for state mapping: {}",
        obj.get_type().name()?.to_string()
    )))
}

fn json_to_python(py: Python<'_>, v: &Value) -> Py<PyAny> {
    match v {
        Value::Null => py.None().into(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else if let Some(u) = n.as_u64() {
                u.into_py(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py(py)
            } else {
                py.None()
            }
        }
        Value::String(s) => s.clone().into_py(py),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_python(py, item)).ok();
            }
            list.into()
        }
        Value::Object(o) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in o {
                dict.set_item(k, json_to_python(py, v)).ok();
            }
            dict.into()
        }
    }
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    parent.add_class::<PySearchState>()?;
    Ok(())
}

use pyo3::IntoPy;
