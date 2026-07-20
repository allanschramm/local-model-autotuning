# state.py
# Core state management module for Search baseline overrides and visited memory.

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Set

from autoresearch.core import config
from autoresearch.core.config import ConfigError, validate_config

class SearchState:
    """
    Deep module encapsulating baseline overrides and visited memory.
    Loads overrides and visited history on initialization, and writes updates to disk.
    """

    def __init__(self, state_path: Path | str | None = None):
        self.state_path = Path(state_path) if state_path is not None else config.STATE_FILE
        self._defaults = config.DEFAULTS
        self._baseline_overrides = {}
        self._visited = set()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load state overrides and visited history from disk and populate memory cache."""
        if not self.state_path.exists():
            self._baseline_overrides = {}
            self._visited = set()
            return
        
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigError(f"Failed to read state file: {exc}")

        schema_version = data.get("schema_version")
        if schema_version != config.STATE_SCHEMA_VERSION:
            raise ConfigError(f"Unsupported state schema: {schema_version}")

        baseline = data.get("baseline", {})
        overrides = {k: baseline[k] for k in config.CONFIG_KEYS if k in baseline}
        self._baseline_overrides = self._filter_overrides(overrides)
        self._visited = set(data.get("visited", []))

    def _write_to_disk(self) -> None:
        """Atomically serialize cache state to disk. Sync write ensures crash resilience."""
        data = {
            "schema_version": config.STATE_SCHEMA_VERSION,
            "baseline": self._baseline_overrides,
            "visited": sorted(list(self._visited))
        }

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.state_path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _merge_and_validate(self, overrides_dict: dict[str, Any]) -> dict[str, Any]:
        """Merge overrides onto defaults and validate the full config."""
        merged = dict(self._defaults)
        merged.update(overrides_dict)
        return validate_config(merged)

    def _filter_overrides(self, overrides_dict: dict[str, Any]) -> dict[str, Any]:
        """Merge overrides onto defaults, validate, and return actual overrides."""
        validated = self._merge_and_validate(overrides_dict)
        return {
            k: validated[k]
            for k in config.CONFIG_KEYS
            if k in validated and validated[k] != self._defaults.get(k)
        }

    def get_baseline(self) -> dict[str, Any]:
        """Return baseline overrides overlaid on defaults."""
        merged = dict(self._defaults)
        merged.update(self._baseline_overrides)
        return merged

    def update_baseline(self, new_cfg: dict[str, Any]) -> None:
        """Filter, merge, and persist the overrides that differ from defaults."""
        merged_overrides = dict(self._baseline_overrides)
        filtered_new = {k: new_cfg[k] for k in config.CONFIG_KEYS if k in new_cfg}
        merged_overrides.update(filtered_new)
        self._baseline_overrides = self._filter_overrides(merged_overrides)
        self._write_to_disk()

    @property
    def visited(self) -> Set[str]:
        """Return a copy of the visited configurations set."""
        return set(self._visited)

    def is_visited(self, config_key: str) -> bool:
        """Check if a specific config key has been marked as visited."""
        return config_key in self._visited

    def mark_visited(self, config_key: str, persist: bool = True) -> None:
        """Mark a configuration key as visited, optionally persisting to disk."""
        self._visited.add(config_key)
        if persist:
            self._write_to_disk()

    def reset(self) -> None:
        """Reset baseline overrides and clear visited history."""
        self._baseline_overrides = {}
        self._visited = set()
        self._write_to_disk()
