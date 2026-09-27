"""Visited-memory module for the Search (issue #53, ADR 0014).

The visited set + Morris pin dictionary + atomic JSON persistence now
live in `autoresearch_core.state` (a pure-Rust crate exposed via PyO3 +
maturin). Baseline I/O still goes through `autoresearch.core.config` so
the Python contract is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch_core.state import SearchState as _RustSearchState

from autoresearch.core import config as _config
from autoresearch.core.config import (  # noqa: F401  (re-exported for tests + shim mocking)
    ConfigError,
    validate_config,
    write_baseline,
)


class SearchState:
    """Deep module for visited memory. Baseline read/write goes through config.py."""

    def __init__(self, state_path: Path | str | None = None) -> None:
        path = Path(state_path) if state_path is not None else _config.STATE_FILE
        self._rust = _RustSearchState(str(path))
        self.state_path = path

    def get_baseline(self) -> dict[str, Any]:
        """Return current Baseline from config.py."""
        return _config.load_config()

    def update_baseline(self, new_cfg: dict[str, Any]) -> None:
        """Merge into Baseline and persist via config.write_baseline."""
        merged = _config.load_config()
        for key in _config.CONFIG_KEYS:
            if key in new_cfg:
                merged[key] = new_cfg[key]
        write_baseline(validate_config(merged))

    @property
    def visited(self) -> set[str]:
        """Return a copy of the visited configurations set."""
        return set(self._rust.visited)

    def is_visited(self, config_key: str) -> bool:
        """Check if a specific config key has been marked as visited."""
        return self._rust.is_visited(config_key)

    def mark_visited(self, config_key: str, persist: bool = True) -> None:
        """Mark a configuration key as visited, optionally persisting to disk."""
        self._rust.mark_visited(config_key, persist)

    def morris_pins_for(self, model: str) -> dict:
        """Return stored Morris pins for a model basename, or {}."""
        return dict(self._rust.morris_pins_for(model))

    def set_morris(self, model: str, pins: dict, effects: dict) -> None:
        """Persist Morris pins and elementary-effects for a model."""
        self._rust.set_morris(model, dict(pins), dict(effects))

    def reset(self) -> None:
        """Clear visited history and Morris pins. Baseline stays in config.py."""
        self._rust.reset()


__all__ = ["SearchState"]
