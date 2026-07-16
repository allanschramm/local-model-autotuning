# config.py
# Immutable runtime defaults + invariants. Mutable Baseline lives in
# .autoresearch_state.json (see load_state / write_state). Never rewrite this
# file from the Search loop.
# NOTE: CTX_SIZE is frozen at 131072. Min ctx floor = 100k. Never lower it.

from typing import Any
from pathlib import Path
import json
import math
import os
import tempfile

MODEL = 'Qwythos-9B-v2-Q4_K_M.gguf'
CTX_SIZE = 131072
KV_CACHE = 'q4_0'
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
# Sweet spot for RTX 4060 8GB (llama-bench 2026-07-05): ub=128 pp1861 t/s tg43.4 t/s, ub=256 pp1768 t/s tg40.8 t/s
BATCH_SIZE = 1024
UBATCH_SIZE = 128
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
NO_MMAP = False
JINJA = False
REASONING_BUDGET = None
REASONING_BUDGET_MESSAGE = None
REASONING = None
CONT_BATCHING = True
N_CPU_MOE = 32

# Generation options (Unsloth-corrected for Qwen3.5 thinking mode)
TEMP = 0.4
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
REPEAT_PENALTY = 1.05
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = None

MIN_CTX_SIZE = 100_000
STATE_SCHEMA_VERSION = 1
STATE_FILE = Path(__file__).resolve().parents[2] / ".autoresearch_state.json"
CONFIG_KEYS = (
    "MODEL", "CTX_SIZE", "KV_CACHE", "KV_CACHE_K", "KV_CACHE_V", "BATCH_SIZE",
    "UBATCH_SIZE", "THREADS", "THREADS_BATCH", "FLASH_ATTN", "SPEC_TYPE",
    "SPEC_DRAFT_N_MAX", "NO_MMAP", "JINJA", "REASONING_BUDGET",
    "REASONING_BUDGET_MESSAGE", "REASONING", "CONT_BATCHING", "N_CPU_MOE",
    "TEMP", "TOP_P", "TOP_K", "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY",
    "FREQUENCY_PENALTY",
)


class ConfigError(ValueError):
    """Configuration violates a non-negotiable runtime invariant."""


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy, or fail before any model process starts."""
    normalized = dict(cfg)
    ctx = int(normalized.get("ctx_size", normalized.get("CTX_SIZE", CTX_SIZE)))
    flash = normalized.get("flash_attn", normalized.get("FLASH_ATTN", FLASH_ATTN))
    if ctx < MIN_CTX_SIZE:
        raise ConfigError(f"CTX_SIZE must be >= {MIN_CTX_SIZE}; got {ctx}")
    if flash != "on":
        raise ConfigError("FLASH_ATTN must be 'on'")
    batch = int(normalized.get("batch_size", normalized.get("BATCH_SIZE", BATCH_SIZE)))
    ubatch = int(normalized.get("ubatch_size", normalized.get("UBATCH_SIZE", UBATCH_SIZE)))
    if batch <= 0 or ubatch <= 0 or ubatch > batch:
        raise ConfigError("Require BATCH_SIZE > 0 and 0 < UBATCH_SIZE <= BATCH_SIZE")
    for key, value in normalized.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{key} must be finite")
    normalized["CTX_SIZE"] = ctx
    normalized["FLASH_ATTN"] = flash
    return normalized


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    """Load the local Baseline, initializing it from immutable defaults."""
    state_path = Path(path) if path else STATE_FILE
    if not state_path.exists():
        return {"schema_version": STATE_SCHEMA_VERSION, "baseline": load_config(), "visited": []}
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ConfigError(f"Unsupported state schema: {data.get('schema_version')}")
    data["baseline"] = validate_config(data.get("baseline", {}))
    return data


def write_state(state: dict[str, Any], path: str | Path | None = None) -> None:
    """Atomically persist local Search state without rewriting Python source."""
    state_path = Path(path) if path else STATE_FILE
    payload = dict(state)
    payload["schema_version"] = STATE_SCHEMA_VERSION
    payload["baseline"] = validate_config(payload.get("baseline", {}))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=state_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, state_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)



def load_config(params: list[str] | None = None) -> dict[str, Any]:
    """Return immutable Python defaults as a dictionary."""
    import sys
    mod = sys.modules[__name__]
    if params is not None:
        return {p: getattr(mod, p, None) for p in params}
    return {k: getattr(mod, k) for k in CONFIG_KEYS}

