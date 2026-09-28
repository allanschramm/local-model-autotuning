//! PyO3 bindings for the `fingerprint` module — Phase 0 Sprint 0.2.
//! Mirrors the public surface of `autoresearch.core.fingerprint`:
//! `FINGERPRINT_SCHEMA_VERSION`, `SCHEMA_VERSION`, `dump`, `load`,
//! `apply` (Python shim), `mismatch_reason`, `path_for`, and the
//! `FingerprintError` exception (mapped to Python `ValueError`).

use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::IntoPy;

use serde_json::{Map, Value};

use crate::error::{CoreError, CoreResult};
use crate::fingerprint;
use crate::fingerprint::dump::dump;
use crate::fingerprint::load::load;
use crate::fingerprint::mismatch::mismatch_reason;

// Mirror the Python `class FingerprintError(ValueError)`. Bound as a
// Python class via PyO3; consumers catch it the same way they always have.
create_exception!(
    autoresearch_core,
    FingerprintError,
    PyValueError,
    "Raised when a Fingerprint payload leaks private data or breaks schema."
);

fn scrub_or_py(e: CoreError) -> PyErr {
    match e {
        CoreError::ScrubViolation { path, reason } => {
            FingerprintError::new_err(format!("scrub violation at '{path}': {reason}"))
        }
        CoreError::SchemaMismatch {
            file_version,
            expected,
        } => FingerprintError::new_err(format!(
            "unsupported Fingerprint schema_version {file_version} (want {expected})"
        )),
        CoreError::Io { source, .. } => PyOSError::new_err(source.to_string()),
        other => PyErr::new::<PyValueError, _>(other.to_string()),
    }
}

/// `dump(path, *, model, engine, sampler=None) -> Path` — port of the Python
/// `fingerprint.dump`.
#[pyfunction(name = "dump")]
#[pyo3(signature = (path, *, model, engine, sampler=None))]
fn py_dump(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    model: &str,
    engine: &Bound<'_, PyDict>,
    sampler: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let engine_map = pythonize_obj_map(engine)?;
    let sampler_map: Option<Map<String, Value>> = match sampler {
        Some(s) => Some(pythonize_obj_map(s)?),
        None => None,
    };
    let target_path = path_for_callable(path)?;
    let written =
        dump(&target_path, model, &engine_map, sampler_map.as_ref()).map_err(scrub_or_py)?;
    path_from(&written, path, py)
}

#[pyfunction(name = "load")]
fn py_load(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let target = path_for_callable(path)?;
    let map = load(&target).map_err(scrub_or_py)?;
    json_map_to_python(py, &map)
}

#[pyfunction(name = "mismatch_reason")]
#[pyo3(signature = (model_basename, baseline_engine, *, directory=None, _target_path=None))]
fn py_mismatch_reason(
    model_basename: &str,
    baseline_engine: &Bound<'_, PyDict>,
    directory: Option<&str>,
    _target_path: Option<&str>,
) -> PyResult<Option<String>> {
    let baseline = pythonize_obj_map(baseline_engine)?;
    let r#override = _target_path.map(std::path::Path::new);
    let result =
        mismatch_reason(model_basename, &baseline, directory, r#override).map_err(scrub_or_py)?;
    Ok(result)
}

#[pyfunction(name = "path_for")]
#[pyo3(signature = (model_basename, directory=None))]
fn py_path_for(model_basename: &str, directory: Option<&str>) -> PyResult<String> {
    let stem = std::path::Path::new(model_basename.trim())
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let dir = directory
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("fingerprints"));
    let p = dir.join(format!("{stem}.json"));
    Ok(p.to_string_lossy().to_string())
}

// Helpers

fn pythonize_obj_map(obj: &Bound<'_, PyAny>) -> PyResult<Map<String, Value>> {
    if obj.is_none() {
        return Ok(Map::new());
    }
    let v: Value = pythonize_to_json(obj)?;
    let m = v
        .as_object()
        .cloned()
        .ok_or_else(|| PyValueError::new_err("engine must be a mapping of ENGINE_DEFAULTS"))?;
    Ok(m)
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
        let mut map: Map<String, serde_json::Value> = Map::new();
        for (key, value) in dict.iter() {
            let k = key.extract::<String>()?;
            map.insert(k, pythonize_to_json(&value)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(serde_json::Value::String(s.to_string()));
    }
    Err(PyValueError::new_err(format!(
        "unsupported Python type for fingerprint mapping: {}",
        obj.get_type().name()?
    )))
}

fn json_map_to_python(py: Python<'_>, map: &Map<String, Value>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    for (k, v) in map.iter() {
        dict.set_item(k, json_value_to_python(py, v)?)?;
    }
    Ok(dict.into())
}

fn json_value_to_python(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    match v {
        Value::Null => Ok(py.None().into()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Err(PyValueError::new_err("unsupported number"))
            }
        }
        Value::String(s) => Ok(s.clone().into_py(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_value_to_python(py, item)?)?;
            }
            Ok(list.into())
        }
        Value::Object(o) => json_map_to_python(py, o),
    }
}

fn path_for_callable(arg: &Bound<'_, PyAny>) -> PyResult<std::path::PathBuf> {
    // Accept either `pathlib.Path` (extract to string then `PathBuf`) or a
    // string. We deliberately avoid extracting the `Path` via `extract` so
    // os.fspath semantics apply uniformly.
    if let Ok(p) = arg.extract::<std::path::PathBuf>() {
        return Ok(p);
    }
    if let Ok(s) = arg.extract::<String>() {
        return Ok(std::path::PathBuf::from(s));
    }
    Err(PyValueError::new_err(format!(
        "expected a path-like object, got {}",
        arg.get_type().name()?.to_string()
    )))
}

fn path_from(
    p: &std::path::Path,
    original: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    // Try to mirror the input's type (string vs Path). The Python
    // `dump()` always re-emits a string for portability; pathlib.Path
    // round-tripping through PyO3 is noisy enough that we keep things
    // simple by always returning a string for now.
    let s = p.to_string_lossy().to_string();
    if original.is_instance_of::<PyString>() {
        Ok(PyString::new_bound(py, &s).into())
    } else {
        Ok(PyString::new_bound(py, &s).into())
    }
}

// `engine` parameter for `load` would accept an already-parsed dict; we
// don't have a `load_into` variant, so this helper is reserved for the
// future shim that wants to surface parsed JSON as a Python dict (the
// current `py_load` already does that).
#[allow(dead_code)]
fn _ensure_load_path(py: Python<'_>) -> CoreResult<()> {
    let _ = py;
    Ok(())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add(
        "FingerprintError",
        parent.py().get_type_bound::<FingerprintError>(),
    )?;
    parent.add(
        "FINGERPRINT_SCHEMA_VERSION",
        fingerprint::FINGERPRINT_SCHEMA_VERSION,
    )?;
    parent.add("SCHEMA_VERSION", fingerprint::FINGERPRINT_SCHEMA_VERSION)?;
    parent.add_function(wrap_pyfunction!(py_dump, parent)?)?;
    parent.add_function(wrap_pyfunction!(py_load, parent)?)?;
    parent.add_function(wrap_pyfunction!(py_mismatch_reason, parent)?)?;
    parent.add_function(wrap_pyfunction!(py_path_for, parent)?)?;
    let _ = py;
    Ok(())
}
