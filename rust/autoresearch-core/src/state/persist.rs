//! Atomic JSON persistence for `SearchState`. Port of the Python
//! `_write_to_disk` (`tempfile.mkstemp` + `os.replace` with `fsync`).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{CoreError, CoreResult};

/// Legacy schema versions accepted on load (issue #5 / config.STATE_SCHEMA_VERSION).
/// Phase 0 keeps the same set so existing files round-trip without migration.
const ACCEPTED_SCHEMAS: &[u32] = &[1, 2, 3];

/// Atomic JSON write: tmpfile + fsync + os.replace. Same semantics as the
/// Python `tempfile.mkstemp` + `os.replace` path.
pub fn write_atomic(target: &Path, data: &serde_json::Value) -> CoreResult<()> {
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    let dir = target.parent().filter(|p| !p.as_os_str().is_empty());
    let dir_owned = dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let prefix = format!(
        ".{}.",
        target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("state")
    );
    let mut tmp = tempfile::Builder::new()
        .prefix(&prefix)
        .suffix(".tmp")
        .tempfile_in(&dir_owned)
        .map_err(|source| CoreError::Io {
            path: dir_owned,
            source,
        })?;
    let bytes = serde_json::to_vec_pretty(data)?;
    tmp.write_all(&bytes)?;
    tmp.write_all(b"\n")?;
    tmp.flush().map_err(|source| CoreError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    tmp.as_file().sync_all().map_err(|source| CoreError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    tmp.persist(target).map_err(|err| CoreError::Io {
        path: target.to_path_buf(),
        source: err.error,
    })?;
    Ok(())
}

/// Read a state file, validating the schema version. Returns the parsed
/// JSON value (a `Map`) on success.
pub fn read_validated(target: &Path) -> CoreResult<serde_json::Value> {
    let raw = fs::read_to_string(target).map_err(|source| CoreError::Io {
        path: target.to_path_buf(),
        source,
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let obj = value.as_object().ok_or_else(|| CoreError::ScrubViolation {
        path: target.to_string_lossy().into_owned(),
        reason: "state payload is not an object".into(),
    })?;
    let version = obj
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| CoreError::ScrubViolation {
            path: target.to_string_lossy().into_owned(),
            reason: "missing schema_version".into(),
        })?;
    let v = version as u32;
    if !ACCEPTED_SCHEMAS.contains(&v) {
        return Err(CoreError::ScrubViolation {
            path: target.to_string_lossy().into_owned(),
            reason: format!("unsupported state schema: {v}"),
        });
    }
    Ok(value)
}

/// Convenience: append a single key to `visited` and re-persist atomically.
pub fn append_visited(target: &Path, key: &str) -> CoreResult<serde_json::Value> {
    let mut value = if target.exists() {
        read_validated(target)?
    } else {
        serde_json::json!({"schema_version": 3u32, "visited": [], "morris": {}})
    };
    let obj = value.as_object_mut().expect("fresh object");
    let visited = obj
        .entry("visited")
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    let arr = visited.as_array_mut().expect("visited is array");
    if !arr
        .iter()
        .any(|v| v == &serde_json::Value::String(key.into()))
    {
        arr.push(serde_json::Value::String(key.into()));
        arr.sort_by(|a, b| match (a.as_str(), b.as_str()) {
            (Some(x), Some(y)) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        });
    }
    write_atomic(target, &value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn write_atomic_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("state.json");
        let payload = json!({"schema_version": 3u32, "visited": ["a", "b"], "morris": {}});
        write_atomic(&target, &payload).unwrap();
        let read = read_validated(&target).unwrap();
        assert_eq!(read, payload);
    }

    #[test]
    fn read_validated_rejects_unknown_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("state.json");
        std::fs::write(
            &target,
            br#"{"schema_version": 99, "visited": [], "morris": {}}"#,
        )
        .unwrap();
        let result = read_validated(&target);
        assert!(matches!(result, Err(CoreError::ScrubViolation { .. })));
    }

    #[test]
    fn read_validated_accepts_legacy_schemas() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("state.json");
        std::fs::write(
            &target,
            br#"{"schema_version": 1, "visited": ["x"], "morris": {}}"#,
        )
        .unwrap();
        let value = read_validated(&target).unwrap();
        assert_eq!(value["schema_version"], 1);
    }

    #[test]
    fn append_visited_dedupes_and_sorts() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("state.json");
        write_atomic(
            &target,
            &json!({"schema_version": 3u32, "visited": ["b"], "morris": {}}),
        )
        .unwrap();
        append_visited(&target, "a").unwrap();
        append_visited(&target, "b").unwrap(); // already present
        let value = read_validated(&target).unwrap();
        let arr = value["visited"].as_array().unwrap();
        let s: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(s, vec!["a", "b"]);
    }
}
