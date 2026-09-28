//! `Fingerprint.load(path)` — port of
//! `autoresearch/core/fingerprint.py::load`. Validates schema, scrubs
//! private data, returns a normalized dict (canonical JSON shape with
//! sort_keys-friendly field order).

use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{CoreError, CoreResult};
use crate::fingerprint::scrub::{check_payload, SCHEMA_VERSION};

fn starts_with_absolute(s: &str) -> bool {
    let t = s.trim_start();
    if t.starts_with('/') || t.starts_with("~/") || t.starts_with('~') || t.starts_with("\\\\") {
        return true;
    }
    // Drive-letter prefix: `C:\` or `C:/`.
    let bytes = t.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let sep = bytes[2];
        if sep == b'\\' || sep == b'/' {
            return true;
        }
    }
    false
}

/// Load a Fingerprint JSON file, returning the validated payload as a
/// fresh dict. Mirrors `autoresearch.core.fingerprint.load`.
pub fn load(target: &Path) -> CoreResult<Map<String, Value>> {
    let raw = fs::read_to_string(target).map_err(|source| CoreError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let payload: Value = serde_json::from_str(&raw)?;
    let obj = payload
        .as_object()
        .ok_or_else(|| CoreError::ScrubViolation {
            path: "$".into(),
            reason: format!("invalid Fingerprint payload in {target:?}: not an object"),
        })?;

    // Schema version check.
    let version = obj
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| CoreError::ScrubViolation {
            path: "$.schema_version".into(),
            reason: format!(
                "unsupported Fingerprint schema_version in {target:?} (want {SCHEMA_VERSION}); schema_version is required"
            ),
        })?;
    if version != u64::from(SCHEMA_VERSION) {
        return Err(CoreError::SchemaMismatch {
            file_version: version as u32,
            expected: SCHEMA_VERSION,
        });
    }

    let model =
        obj.get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| CoreError::ScrubViolation {
                path: "$.model".into(),
                reason: format!("invalid Fingerprint model in {target:?}: GGUF basename only"),
            })?;
    if !looks_like_basename(model) {
        return Err(CoreError::ScrubViolation {
            path: "$.model".into(),
            reason: format!(
                "invalid Fingerprint model {model:?} in {target:?}: GGUF basename only"
            ),
        });
    }

    // Engine is required.
    let engine_obj = obj
        .get("engine")
        .and_then(Value::as_object)
        .ok_or_else(|| CoreError::ScrubViolation {
            path: "$.engine".into(),
            reason: format!("invalid Fingerprint engine in {target:?}: mapping required"),
        })?;
    if engine_obj.is_empty() {
        return Err(CoreError::ScrubViolation {
            path: "$.engine".into(),
            reason: format!("invalid Fingerprint engine in {target:?}: mapping required"),
        });
    }
    check_payload(&Value::Object(engine_obj.clone()), "$.engine")?;

    // Sampler is optional.
    let sampler_value = match obj.get("sampler") {
        Some(Value::Null) | None => None,
        Some(Value::Object(s)) => {
            if s.is_empty() {
                return Err(CoreError::ScrubViolation {
                    path: "$.sampler".into(),
                    reason: format!("invalid Fingerprint sampler in {target:?}: mapping required"),
                });
            }
            check_payload(&Value::Object(s.clone()), "$.sampler")?;
            Some(Value::Object(s.clone()))
        }
        Some(_) => {
            return Err(CoreError::ScrubViolation {
                path: "$.sampler".into(),
                reason: format!("invalid Fingerprint sampler in {target:?}: mapping required"),
            });
        }
    };

    let mut result: Map<String, Value> = Map::new();
    result.insert(
        "schema_version".to_string(),
        Value::Number(SCHEMA_VERSION.into()),
    );
    result.insert("model".to_string(), Value::String(model.to_string()));
    result.insert("engine".to_string(), Value::Object(engine_obj.clone()));
    result.insert("sampler".to_string(), sampler_value.unwrap_or(Value::Null));
    Ok(result)
}

fn looks_like_basename(name: &str) -> bool {
    let s = name;
    if s.contains('/') || s.contains('\\') {
        return false;
    }
    if starts_with_absolute(s) {
        return false;
    }
    // Mimic `PureWindowsPath(name).name == name and PurePath(name).name == name`.
    let posix = std::path::Path::new(s).file_name().and_then(|x| x.to_str()) == Some(s);
    let win = match s.rfind('\\') {
        Some(idx) => &s[idx + 1..] == s,
        None => true,
    };
    posix && win
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(tmp: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = tmp.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn load_roundtrips_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let content = r#"{
  "schema_version": 1,
  "model": "Qwen.gguf",
  "engine": {
    "CTX_SIZE": 2048,
    "MODEL": "Qwen.gguf"
  }
}"#;
        let path = write(tmp.path(), "f.json", content);
        let out = load(&path).unwrap();
        assert_eq!(out["model"], "Qwen.gguf");
        assert_eq!(out["engine"]["CTX_SIZE"], 2048);
    }

    #[test]
    fn load_rejects_unknown_schema_version() {
        let tmp = tempfile::tempdir().unwrap();
        let content = r#"{"schema_version": 99, "model": "Q.gguf", "engine": {"MODEL": "Q.gguf"}}"#;
        let path = write(tmp.path(), "f.json", content);
        assert!(matches!(load(&path), Err(CoreError::SchemaMismatch { .. })));
    }

    #[test]
    fn load_rejects_private_data_in_engine() {
        let tmp = tempfile::tempdir().unwrap();
        // `email` is in the private-key set.
        let mut map = Map::new();
        map.insert("email".to_string(), json!("operator@example.com"));
        map.insert("MODEL".to_string(), json!("Q.gguf"));
        let payload = json!({"schema_version": 1, "model": "Q.gguf", "engine": map});
        let path = write(
            tmp.path(),
            "f.json",
            &serde_json::to_string_pretty(&payload).unwrap(),
        );
        let result = load(&path);
        assert!(result.is_err());
    }
}
