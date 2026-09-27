//! `Fingerprint.mismatch_reason(model_basename, baseline_engine, *, directory=None)`
//! — port of `autoresearch/core/fingerprint.py::mismatch_reason`. Returns
//! `None` when the Baseline-aligned Trial may proceed, or a human-readable
//! reject reason string otherwise.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::error::CoreResult;
use crate::fingerprint::load::load;

/// Public key sets carried by the Python module (ADR 0014). Kept as
/// `BTreeSet`s so iteration is stable across runs.
pub fn server_engine_keys() -> BTreeSet<&'static str> {
    [
        "CTX_SIZE",
        "BATCH_SIZE",
        "UBATCH_SIZE",
        "THREADS",
        "PARALLEL",
        "N_GPU_LAYERS",
        "NUMA",
        "KV_CACHE",
        "KV_CACHE_K",
        "KV_CACHE_V",
        "FLASH_ATTN",
        "THREADS_BATCH",
        "NO_MMAP",
        "MLOCK",
        "JINJA",
        "REASONING_BUDGET",
        "REASONING_BUDGET_MESSAGE",
        "REASONING",
        "REASONING_PRESERVE",
        "REASONING_EFFORT",
        "CONT_BATCHING",
        "CACHE_REUSE",
        "SPEC_TYPE",
        "SPEC_DRAFT_N_MAX",
        "SPEC_DRAFT_MODEL",
        "MOE_CACHE_PROFILE",
        "MOE_CACHE_SLOTS",
        "N_CPU_MOE",
    ]
    .into_iter()
    .collect()
}

pub fn harness_only_engine_keys() -> BTreeSet<&'static str> {
    [
        "VRAM_LIMIT_MB",
        "VRAM_HEADROOM_MB",
        "HOST_MEMORY_HEADROOM_MB",
        "FREE_RAM_FLOOR_MB",
        "RAM_WATCHDOG_POLL_S",
        "RAM_WATCHDOG_RESERVE_MB",
        "RAM_PREFLIGHT_MARGIN_MB",
        "TPS_FLOOR",
        "TPS_REPS",
        "THERMAL_WAIT",
    ]
    .into_iter()
    .collect()
}

/// Strip POSIX + Windows directory components to a basename. Mirrors the
/// Python `_basename` helper.
fn basename(name: &str) -> String {
    let t = name.trim();
    if t.is_empty() {
        return String::new();
    }
    let cleaned = t.replace('\\', "/");
    std::path::Path::new(&cleaned)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| t.to_string())
}

/// Return the on-disk Fingerprint path for `model_basename`.
fn path_for(model_basename: &str, directory: Option<&str>) -> PathBuf {
    let stem = std::path::Path::new(basename(model_basename).as_str())
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let dir = directory.map_or_else(default_dir, std::path::PathBuf::from);
    dir.join(format!("{stem}.json"))
}

fn default_dir() -> PathBuf {
    // `<repo>/fingerprints/` — Phase 0 keeps the Python default directory
    // in lock-step; the autoloop invents nothing new here.
    PathBuf::from("fingerprints")
}

/// Trial gate. None = proceed, Some(reason) = reject. When
/// `target_override` is `Some`, the resolved path is used directly
/// (otherwise the path is derived from `model_basename` + `directory`).
pub fn mismatch_reason(
    model_basename: &str,
    baseline_engine: &Map<String, Value>,
    directory: Option<&str>,
    target_override: Option<&Path>,
) -> CoreResult<Option<String>> {
    let base = basename(model_basename);
    if base.is_empty() {
        return Ok(Some(
            "Fingerprint check: Trial has no model basename; refusing to score".into(),
        ));
    }

    let baseline: BTreeMap<String, String> = baseline_engine
        .iter()
        .map(|(k, v)| (k.to_uppercase(), json_to_str(v)))
        .collect();

    let target = target_override
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| path_for(&base, directory));
    if !target.is_file() {
        return Ok(None);
    }
    let data = match load(&target) {
        Ok(d) => d,
        Err(crate::error::CoreError::ScrubViolation { reason, .. }) => {
            return Ok(Some(format!(
                "Fingerprint {} for {base} is invalid ({reason}); delete it or re-climb",
                target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            )));
        }
        Err(crate::error::CoreError::SchemaMismatch {
            file_version,
            expected,
        }) => {
            return Ok(Some(format!(
                "Fingerprint {} for {base} uses schema_version {file_version} (want {expected}); delete it or re-climb",
                target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            )));
        }
        Err(crate::error::CoreError::Io { source, .. }) => {
            return Ok(Some(format!(
                "Fingerprint {} for {base} is unreadable ({source}); delete it or re-climb",
                target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            )));
        }
        Err(other) => {
            return Ok(Some(format!(
                "Fingerprint {} for {base} is invalid ({other}); delete it or re-climb",
                target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            )));
        }
    };

    let stored_model = data
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if stored_model != base {
        return Ok(Some(format!(
            "Fingerprint {} serves {stored_model:?}, not the Trial model {base:?}; delete it or re-climb",
            target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        )));
    }

    let file_engine = match data.get("engine").and_then(Value::as_object) {
        Some(o) => o.clone(),
        None => Map::new(),
    };
    let unknown: Vec<String> = {
        let mut all_known: std::collections::BTreeSet<&'static str> = server_engine_keys();
        let harness = harness_only_engine_keys();
        for k in harness {
            all_known.insert(k);
        }
        file_engine
            .keys()
            .filter(|k| !all_known.contains(k.as_str()) && k.as_str() != "MODEL")
            .cloned()
            .collect()
    };
    if !unknown.is_empty() {
        let mut u = unknown;
        u.sort();
        return Ok(Some(format!(
            "Fingerprint {} for {base} has unknown engine keys {u:?}; delete it or re-climb",
            target.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        )));
    }

    let server = server_engine_keys();
    let mut diffs: Vec<String> = Vec::new();
    let mut keys: Vec<&'static str> = server.into_iter().collect();
    keys.push("MODEL");
    keys.sort();
    for key in keys {
        let from_file = file_engine
            .get(key)
            .map(|v| json_to_str(v))
            .unwrap_or_default();
        let from_baseline = baseline.get(key).cloned().unwrap_or_default();
        if from_baseline != from_file {
            diffs.push(format!(
                "{key} Baseline={from_baseline:?} file={from_file:?}"
            ));
        }
    }
    if diffs.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "Fingerprint mismatch for {base} ({}): {} (Baseline engine differs from the TPS-climbed flags; apply the Fingerprint or re-climb before scoring)",
        target.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
        diffs.join("; ")
    )))
}

fn json_to_str(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn engine_from(pairs: &[(&str, &str)]) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, v) in pairs {
            m.insert((*k).into(), Value::String((*v).into()));
        }
        m
    }

    #[test]
    fn empty_baseline_basename_rejects() {
        let engine = engine_from(&[("CTX_SIZE", "2048")]);
        let reason = mismatch_reason("", &engine, None, None).unwrap();
        assert!(reason.unwrap().contains("no model basename"));
    }

    #[test]
    fn keys_match_python_module_constants() {
        let server = server_engine_keys();
        assert!(server.contains("CTX_SIZE"));
        assert!(server.contains("FLASH_ATTN"));
        let harness = harness_only_engine_keys();
        assert!(harness.contains("TPS_FLOOR"));
        assert!(harness.contains("VRAM_LIMIT_MB"));
    }
}
