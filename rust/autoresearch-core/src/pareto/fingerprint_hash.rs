//! SHA‑256 over canonical JSON of `{"engine": …, "sampler": …}` — same
//! dict shape as the Python `pareto.fingerprint`.
//!
//! `json.dumps(..., sort_keys=True, separators=(",", ":"))` is the canonical
//! form; we use a hand-rolled formatter so the bytes are identical even for
//! nested dicts with mixed key orders.

use sha2::{Digest, Sha256};

/// Hash the canonical (sorted, compact) JSON of `{"engine": …, "sampler":
/// …}`. Returns the lowercase hex digest.
#[must_use]
pub fn fingerprint_hash(engine: &serde_json::Value, sampler: &serde_json::Value) -> String {
    let wrapper = serde_json::json!({"engine": engine, "sampler": sampler});
    let bytes = canonical_json(&wrapper);
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    hex_lower(&digest)
}

/// Emit the value's canonical form: object keys sorted (recursively),
/// compact separators. Matches `json.dumps(..., sort_keys=True,
/// separators=(",", ":"))` 1:1.
#[must_use]
pub fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    write_value(value, &mut out);
    out
}

fn write_value(value: &serde_json::Value, out: &mut Vec<u8>) {
    use serde_json::Value;
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            // serde_json::Number renders itself without spaces.
            out.extend_from_slice(n.to_string().as_bytes());
        }
        Value::String(s) => write_string(s, out),
        Value::Array(arr) => {
            out.push(b'[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(item, out);
            }
            out.push(b']');
        }
        Value::Object(obj) => {
            out.push(b'{');
            // BTreeMap sorts keys (dicts are Python-mapped so order is
            // canonical sorted).
            use std::collections::BTreeMap;
            let sorted: BTreeMap<&String, &serde_json::Value> = obj.iter().collect();
            for (i, (k, v)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(k, out);
                out.push(b':');
                write_value(v, out);
            }
            out.push(b'}');
        }
    }
}

fn write_string(s: &str, out: &mut Vec<u8>) {
    // `serde_json::to_string` already wraps the value in `"…"`, escapes
    // inner control characters, and unicode-escapes non-ASCII. Reuse it
    // verbatim so the byte output matches `json.dumps(..., separators=(",", ":"))`.
    let escaped = serde_json::to_string(s).expect("string serialize should not fail");
    out.extend_from_slice(escaped.as_bytes());
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_64_hex_chars() {
        let engine = serde_json::json!({"threads": 4});
        let sampler = serde_json::json!({"top_k": 40});
        let h = fingerprint_hash(&engine, &sampler);
        assert_eq!(h.len(), 64);
        assert!(h
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn hash_is_deterministic_under_key_order() {
        let engine_a = serde_json::json!({"a": 1, "b": 2});
        let engine_b = serde_json::json!({"b": 2, "a": 1});
        assert_eq!(
            fingerprint_hash(&engine_a, &serde_json::json!({})),
            fingerprint_hash(&engine_b, &serde_json::json!({})),
        );
    }

    #[test]
    fn different_inputs_yield_different_hashes() {
        let e1 = serde_json::json!({"threads": 4});
        let e2 = serde_json::json!({"threads": 8});
        assert_ne!(
            fingerprint_hash(&e1, &serde_json::json!({})),
            fingerprint_hash(&e2, &serde_json::json!({})),
        );
    }

    #[test]
    fn canonical_payload_matches_python_dict_not_list() {
        // Regression: Python uses {"engine": ..., "sampler": ...} not
        // [engine, sampler]. The byte payload must reflect the dict form.
        let engine = serde_json::json!({"a": 1});
        let sampler = serde_json::json!({"b": 2});
        let bytes = canonical_json(&serde_json::json!({"engine": engine, "sampler": sampler}));
        assert_eq!(bytes, br#"{"engine":{"a":1},"sampler":{"b":2}}"#);
    }
}
