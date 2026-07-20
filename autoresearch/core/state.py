# state.py
# Visited-memory module for the Search. Baseline lives in config.py.

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Set

from autoresearch.core import config
from autoresearch.core.config import ConfigError, validate_config, write_baseline


class SearchState:
    """Deep module for visited memory. Baseline read/write goes through config.py."""

    def __init__(self, state_path: Path | str | None = None):
        self.state_path = Path(state_path) if state_path is not None else config.STATE_FILE
        self._visited = set()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load visited history from disk. Ignore legacy baseline payloads."""
        if not self.state_path.exists():
            self._visited = set()
            return

        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigError(f"Failed to read state file: {exc}")

        schema_version = data.get("schema_version")
        if schema_version not in (1, config.STATE_SCHEMA_VERSION):
            raise ConfigError(f"Unsupported state schema: {schema_version}")

        self._visited = set(data.get("visited", []))

    def _write_to_disk(self) -> None:
        """Atomically serialize visited memory. Sync write for crash resilience."""
        data = {
            "schema_version": config.STATE_SCHEMA_VERSION,
            "visited": sorted(list(self._visited)),
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

    def get_baseline(self) -> dict[str, Any]:
        """Return current Baseline from config.py."""
        return config.load_config()

    def update_baseline(self, new_cfg: dict[str, Any]) -> None:
        """Merge into Baseline and persist via config.write_baseline."""
        merged = config.load_config()
        for key in config.CONFIG_KEYS:
            if key in new_cfg:
                merged[key] = new_cfg[key]
        write_baseline(validate_config(merged))

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
        """Clear visited history only. Baseline stays in config.py."""
        self._visited = set()
        self._write_to_disk()
