//! Native search-state module — port of `autoresearch/core/state.py`.
//!
//! `SearchState` tracks visited neighbors + Morris pin dictionaries, with
//! atomic JSON persistence (tmpfile + `fsync` + `os.replace`). Phase 0
//! Sprint 0.3 wires the full persistence path.

pub mod persist;

use std::path::PathBuf;

use serde_json::Value;

use crate::error::CoreResult;

#[derive(Debug, Clone)]
pub struct SearchState {
    pub path: PathBuf,
    pub visited: Vec<String>,
    pub morris: serde_json::Map<String, Value>,
}

impl Default for SearchState {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            visited: Vec::new(),
            morris: serde_json::Map::new(),
        }
    }
}

impl SearchState {
    /// Build an in-memory state by loading from `path` (or empty if absent).
    pub fn load(path: PathBuf) -> CoreResult<Self> {
        if !path.exists() {
            return Ok(Self {
                path,
                visited: Vec::new(),
                morris: serde_json::Map::new(),
            });
        }
        let value = persist::read_validated(&path)?;
        let obj = value.as_object().expect("validated object");
        let visited = obj
            .get("visited")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let morris = obj
            .get("morris")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(serde_json::Map::new);
        Ok(Self {
            path,
            visited,
            morris,
        })
    }

    /// Create an empty in-memory state without touching disk.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_visited(&self, key: &str) -> bool {
        self.visited.iter().any(|k| k == key)
    }

    pub fn persist(&self) -> CoreResult<()> {
        let payload = serde_json::json!({
            "schema_version": 3u32,
            "visited": self.visited,
            "morris": self.morris,
        });
        persist::write_atomic(&self.path, &payload)
    }

    pub fn mark_visited(&mut self, key: &str, persist: bool) -> CoreResult<()> {
        if !self.is_visited(key) {
            self.visited.push(key.to_string());
            self.visited.sort();
        }
        if persist {
            self.persist()?;
        }
        Ok(())
    }

    pub fn morris_pins_for(&self, model: &str) -> serde_json::Map<String, Value> {
        let entry = self.morris.get(model).and_then(Value::as_object);
        let Some(entry) = entry else {
            return serde_json::Map::new();
        };
        let pins = entry.get("pins").and_then(Value::as_object);
        pins.cloned().unwrap_or_default()
    }

    pub fn set_morris(
        &mut self,
        model: &str,
        pins: serde_json::Map<String, Value>,
        effects: serde_json::Map<String, Value>,
    ) -> CoreResult<()> {
        let mut entry = serde_json::Map::new();
        entry.insert("pins".to_string(), Value::Object(pins));
        entry.insert("effects".to_string(), Value::Object(effects));
        self.morris
            .insert(model.to_string(), Value::Object(entry));
        self.persist()
    }

    pub fn reset(&mut self) -> CoreResult<()> {
        self.visited.clear();
        self.morris.clear();
        self.persist()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_starts_with_no_visited() {
        let st = SearchState::new();
        assert!(st.visited.is_empty());
        assert!(st.morris.is_empty());
    }
}

