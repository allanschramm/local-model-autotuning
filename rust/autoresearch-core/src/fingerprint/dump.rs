//! `Fingerprint.dump(path, *, model, engine, sampler=None)` — port of
//! `autoresearch/core/fingerprint.py::dump`. Writes the JSON payload with
//! `sort_keys=True, indent=2`, pre-scrubs private data, and rejects
//! anything that is not a bare GGUF basename for `model`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::error::{CoreError, CoreResult};
use crate::fingerprint::scrub::{check_payload, SCHEMA_VERSION};

/// Strip POSIX + Windows directory components to a basename. Mirrors the
/// Python `PureWindowsPath(name).name or PurePath(name).name or name` fallback.
fn basename(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let cleaned = trimmed.replace('\\', "/");
    std::path::Path::new(&cleaned)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| trimmed.to_string())
}

/// Write a Fingerprint file. Returns the path actually written (after
/// atomic creation).
pub fn dump(
    target: &Path,
    model: &str,
    engine: &Map<String, Value>,
    sampler: Option<&Map<String, Value>>,
) -> CoreResult<PathBuf> {
    let base = basename(model);
    if base.is_empty() {
        return Err(CoreError::ScrubViolation {
            path: "$".into(),
            reason: "model must be a non-empty GGUF basename".into(),
        });
    }

    let mut engine_clean = engine.clone();
    engine_clean.insert("MODEL".to_string(), Value::String(base.clone()));
    scrub_obj(&engine_clean)?;

    let sampler_clean: Option<Map<String, Value>> = sampler.map(|s| s.clone());
    if let Some(s) = sampler_clean.as_ref() {
        scrub_obj(s)?;
    }

    let mut payload = Map::new();
    payload.insert(
        "schema_version".to_string(),
        Value::Number(SCHEMA_VERSION.into()),
    );
    payload.insert("model".to_string(), Value::String(base));
    payload.insert("engine".to_string(), Value::Object(engine_clean));
    if let Some(s) = sampler_clean {
        payload.insert("sampler".to_string(), Value::Object(s));
    }

    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut file = fs::File::create(target)?;
    serde_json::to_writer_pretty(&mut file, &payload)?;
    writeln!(file)?;
    Ok(target.to_path_buf())
}

fn scrub_obj(obj: &Map<String, Value>) -> CoreResult<()> {
    let value = Value::Object(obj.clone());
    check_payload(&value, "$")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn dump_writes_canonical_json() {
        let tmp = tempdir().expect("tempdir");
        let target = tmp.path().join("qwen.json");

        let mut engine = Map::new();
        engine.insert("CTX_SIZE".into(), json!(2048));
        engine.insert("KV_CACHE".into(), json!("q4_0"));

        let written = dump(&target, "Qwen.gguf", &engine, None).expect("dump");
        let content = std::fs::read_to_string(&written).expect("read");
        assert!(content.contains("\"schema_version\": 1"));
        assert!(content.contains("\"engine\": {"));
        assert!(content.contains("\"CTX_SIZE\": 2048"));
    }

    #[test]
    fn dump_rejects_absolute_paths_in_engine() {
        let tmp = tempdir().expect("tempdir");
        let target = tmp.path().join("qwen.json");
        let mut engine = Map::new();
        engine.insert("CTX_SIZE".into(), json!("/home/u/runs"));
        let result = dump(&target, "Qwen.gguf", &engine, None);
        assert!(result.is_err());
    }

    #[test]
    fn dump_strips_model_directory_components() {
        let tmp = tempdir().expect("tempdir");
        let target = tmp.path().join("m.json");
        let mut engine = Map::new();
        engine.insert("CTX_SIZE".into(), json!(1024));
        let written = dump(&target, "C:\\Users\\allan\\models\\Qwen.gguf", &engine, None).unwrap();
        let content = std::fs::read_to_string(&written).unwrap();
        // Basename lands in the model field.
        assert!(content.contains("\"model\": \"Qwen.gguf\""));
        assert!(!content.contains("C:\\\\Users"));
    }

    #[test]
    fn basename_handles_separators() {
        assert_eq!(basename("foo.gguf"), "foo.gguf");
        assert_eq!(basename("path/to/foo.gguf"), "foo.gguf");
        assert_eq!(basename("path\\to\\foo.gguf"), "foo.gguf");
        assert_eq!(basename(""), "");
    }
}
