//! Private-data scrub rules — ported 1:1 from
//! `autoresearch/core/fingerprint.py`. Six regex-driven rules plus a
//! private-key set guard the JSON payload of every Fingerprint file (and
//! the in-memory Baseline before it is written).

use std::sync::LazyLock;

use regex::Regex;

use crate::error::{CoreError, CoreResult};

/// Private JSON keys (case-insensitive) that must never appear in a
/// Fingerprint payload. Mirrors the Python `_PRIVATE_KEYS` frozenset.
pub const PRIVATE_KEYS: &[&str] = &[
    "hostname",
    "host",
    "machine",
    "user",
    "username",
    "email",
    "gpu",
    "gpu_sku",
    "gpu_model",
    "alias",
    "alias_name",
    "model_alias",
];

pub const SCHEMA_VERSION: u32 = 1;

const _ABSOLUTE_PATH_SRC: &str = r"^(?:[A-Za-z]:[\\/]|\\\\|/|~[\\/]?)";
const _EMAIL_SRC: &str = r"[^@\s]+@[^@\s]+\.[^@\s]+";
const _URL_SRC: &str = r"[a-zA-Z][a-zA-Z0-9+.-]*://";
const _GPU_SKU_SRC: &str = r"\b(?:RTX|GTX|GT|RX|ARC|A100|H100|H200|B200)\b";
const _HOSTNAME_SRC: &str =
    r"(?i)^(localhost|([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+(local|lan|internal|[a-z]{2,}))$";
const _FILENAME_EXT_SRC: &str = r"(?i)\.(gguf|bin|safetensors|csv|json|md|txt|yaml|yml|log|ggml)$";

pub static ABSOLUTE_PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_ABSOLUTE_PATH_SRC).expect("ABSOLUTE_PATH_RE compile"));
pub static EMAIL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_EMAIL_SRC).expect("EMAIL_RE compile"));
pub static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_URL_SRC).expect("URL_RE compile"));
pub static GPU_SKU_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_GPU_SKU_SRC).expect("GPU_SKU_RE compile"));
pub static HOSTNAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_HOSTNAME_SRC).expect("HOSTNAME_RE compile"));
pub static FILENAME_EXT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(_FILENAME_EXT_SRC).expect("FILENAME_EXT_RE compile"));

/// Walk a JSON payload, raising [`CoreError::ScrubViolation`] at the
/// first sign of private data. Mirrors Python `_check_mapping` and
/// `_check_value`.
pub fn check_payload(payload: &serde_json::Value, path: &str) -> CoreResult<()> {
    match payload {
        serde_json::Value::Object(obj) => {
            check_mapping(obj.iter().map(|(k, v)| (k.clone(), v)), path)
        }
        serde_json::Value::Array(arr) => {
            for (i, item) in arr.iter().enumerate() {
                check_payload(item, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        serde_json::Value::String(s) => check_str_value(s, "(unknown_key)", path),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            Ok(())
        }
    }
}

fn check_str_value(s: &str, key: &str, path: &str) -> CoreResult<()> {
    check_str_message(s, key).map_err(|reason| violation_at(path, reason))
}

/// Recursive helper for object payloads.
pub fn check_mapping<'a, I>(items: I, path: &str) -> CoreResult<()>
where
    I: IntoIterator<Item = (String, &'a serde_json::Value)>,
{
    for (key, value) in items {
        let key_str = key.trim().to_lowercase();
        if PRIVATE_KEYS.iter().any(|p| *p == key_str) {
            return Err(violation_at(
                path,
                format!("private key {key:?} must not be fingerprinted"),
            ));
        }
        let child_path = format!("{path}.{key}");
        check_value(key.as_str(), value, &child_path)?;
    }
    Ok(())
}

fn check_value(key: &str, value: &serde_json::Value, path: &str) -> CoreResult<()> {
    match value {
        serde_json::Value::String(s) => {
            check_str_message(s, key).map_err(|reason| violation_at(path, reason))
        }
        serde_json::Value::Object(obj) => {
            check_mapping(obj.iter().map(|(k, v)| (k.clone(), v)), path)
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                check_value(key, item, path)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// String scrubber that mirrors the Python messages verbatim so the
/// existing tests can `match="absolute path"`/`"contact/host"`/`"GPU SKU"`/
/// `"hostname"` against the raised `FingerprintError`.
fn check_str_message(s: &str, key: &str) -> Result<(), String> {
    if ABSOLUTE_PATH_RE.is_match(s.trim()) {
        return Err(format!("private absolute path in {key:?}: {s:?}"));
    }
    if EMAIL_RE.is_match(s) || URL_RE.is_match(s) {
        return Err(format!("private contact/host value in {key:?}: {s:?}"));
    }
    if GPU_SKU_RE.is_match(s) {
        return Err(format!("private GPU SKU value in {key:?}: {s:?}"));
    }
    let stripped = s.trim();
    let is_dotted = stripped.contains('.') || stripped.to_lowercase() == "localhost";
    let no_separator = !stripped.contains('/') && !stripped.contains('\\');
    if is_dotted
        && no_separator
        && !FILENAME_EXT_RE.is_match(stripped)
        && HOSTNAME_RE.is_match(stripped)
    {
        return Err(format!("private hostname value in {key:?}: {s:?}"));
    }
    Ok(())
}

fn violation_at(path: &str, reason: String) -> CoreError {
    CoreError::ScrubViolation {
        path: path.to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_object_passes() {
        let v = json!({});
        assert!(check_payload(&v, "$").is_ok());
    }

    #[test]
    fn absolute_path_posix_blocked() {
        let v = json!({"CTX_SIZE": "/home/u/runs"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn absolute_path_windows_blocked() {
        let v = json!({"CTX_SIZE": "C:\\Users\\operator\\x"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn email_blocked() {
        let v = json!({"CTX_SIZE": "operator@example.com"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn url_blocked() {
        let v = json!({"CTX_SIZE": "https://example.com"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn gpu_sku_blocked() {
        let v = json!({"CTX_SIZE": "RTX 4060"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn hostname_in_baseline_blocked() {
        let v = json!({"name": "host.example.com"});
        assert!(check_payload(&v, "$").is_err());
    }

    #[test]
    fn filename_extension_allowed_through_filter() {
        // `MODEL.gguf` must not trip the hostname check.
        let v = json!({"MODEL": "Qwen2.5-7B.gguf"});
        assert!(check_payload(&v, "$").is_ok());
    }

    #[test]
    fn private_key_blocks_payload() {
        let v = json!({"hostname": "ok"});
        assert!(check_payload(&v, "$").is_err());
    }
}
