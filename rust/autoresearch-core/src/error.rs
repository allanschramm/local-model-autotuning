//! Crate-wide error type. Wraps the various failure modes of the ported
//! Python modules (scrub violations, schema mismatches, IO errors) into a
//! single `enum` that maps cleanly to a PyO3 exception class.

use std::fmt;
use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// All errors raised by `autoresearch-core`. Each variant has a 1:1 mapping
/// to a Python exception class — see [`PyO3 mapping`](../py/index.html).
#[derive(Debug, Error)]
pub enum CoreError {
    /// A fingerprint payload contained a value that violates the private-data
    /// scrub rules (absolute path, e-mail, URL, GPU SKU, hostname, disallowed
    /// extension). Maps to `FingerprintError` on the Python side.
    #[error("scrub violation at path '{path}': {reason}")]
    ScrubViolation {
        /// JSON-pointer-like path through the offending payload.
        path: String,
        /// Human-readable violation description.
        reason: String,
    },

    /// The `schema_version` field of a fingerprint file did not match
    /// `FINGERPRINT_SCHEMA_VERSION`. Maps to `FingerprintError`.
    #[error(
        "fingerprint schema version mismatch: file has {file_version}, library expects {expected}"
    )]
    SchemaMismatch {
        /// The version found on disk.
        file_version: u32,
        /// The version the library was compiled with (`FINGERPRINT_SCHEMA_VERSION`).
        expected: u32,
    },

    /// An IO error during atomic state write / fingerprint dump.
    #[error("io error at {path}: {source}")]
    Io {
        /// File path that triggered the IO error.
        path: PathBuf,
        /// Underlying [`io::Error`].
        #[source]
        source: io::Error,
    },

    /// JSON parsing / serialization error.
    #[error("serde_json error: {0}")]
    SerdeJson(String),

    /// Catch-all when a Python interop call fails. Wraps the PyO3 error.
    #[error("python interop error: {0}")]
    PythonInterop(String),
}

impl CoreError {
    /// Cheap constructor for a `ScrubViolation` with a formatted reason.
    #[must_use]
    pub fn scrub<S: fmt::Display>(path: S, reason: impl Into<String>) -> Self {
        Self::ScrubViolation {
            path: path.to_string(),
            reason: reason.into(),
        }
    }
}

impl From<io::Error> for CoreError {
    fn from(source: io::Error) -> Self {
        Self::Io {
            path: PathBuf::new(),
            source,
        }
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(source: serde_json::Error) -> Self {
        Self::SerdeJson(source.to_string())
    }
}

/// Crate result alias — every fallible API in the native lib returns
/// `Result<T, CoreError>`.
pub type CoreResult<T> = std::result::Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_violation_message_contains_path() {
        let err = CoreError::scrub("$.engine.thread_pool", "absolute path detected");
        let msg = format!("{err}");
        assert!(msg.contains("$.engine.thread_pool"));
        assert!(msg.contains("absolute path detected"));
    }

    #[test]
    fn schema_mismatch_message_contains_both_versions() {
        let err = CoreError::SchemaMismatch {
            file_version: 2,
            expected: 1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("file has 2"));
        assert!(msg.contains("library expects 1"));
    }

    #[test]
    fn serde_json_error_roundtrips() {
        let serde_err = serde_json::from_str::<serde_json::Value>("not json")
            .expect_err("must fail");
        let err: CoreError = serde_err.into();
        match err {
            CoreError::SerdeJson(_) => {}
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
