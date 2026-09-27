//! Native fingerprint module — port of `autoresearch/core/fingerprint.py`.
//!
//! Phase 0 Sprint 0.2 implements the full 6-rule scrubber + dump/load
//! pipeline. The bind-side `apply` is a thin Python shim around Rust
//! `load` + Python `config.write_baseline`.

pub mod dump;
pub mod load;
pub mod mismatch;
pub mod scrub;

pub const FINGERPRINT_SCHEMA_VERSION: u32 = 1;

/// Convenience accessor for the schema version, mirroring the Python
/// module-level constant of the same name.
#[inline]
pub fn schema_version() -> u32 {
    FINGERPRINT_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_current() {
        // Bump this only when the on-disk format changes; see ADR Phase 0.
        assert_eq!(schema_version(), 1);
    }
}
