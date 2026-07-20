# config.py
# Immutable runtime defaults + invariants. Mutable Baseline lives in
# .autoresearch_state.json (see SearchState). Never rewrite this
# file from the Search loop.
# NOTE: CTX_SIZE default is 131072. User may lower it to trade context for speed.
# Minimum floor is 2048 (llama.cpp practical minimum).

from typing import Any
from pathlib import Path
import json
import math
import os
import tempfile

class ConfigError(ValueError):
    """Custom exception raised for invalid or unsupported configurations."""
    pass


# Hardware/Engine/Inference configuration parameters (affect speed and VRAM usage)
ENGINE_DEFAULTS = {
    "MODEL": 'Qwythos-9B-v2-Q4_K_M.gguf',
    "CTX_SIZE": 131072,
    "KV_CACHE": 'q4_0',
    "KV_CACHE_K": 'q4_0',
    "KV_CACHE_V": 'q4_0',
    "BATCH_SIZE": 1024,
    "UBATCH_SIZE": 128,
    "THREADS": 8,
    "THREADS_BATCH": 8,
    "FLASH_ATTN": 'on',
    "SPEC_TYPE": None,
    "SPEC_DRAFT_N_MAX": 0,
    "SPEC_DRAFT_MODEL": None,
    "NO_MMAP": False,
    "JINJA": False,
    "REASONING_BUDGET": None,
    "REASONING_BUDGET_MESSAGE": None,
    "REASONING": None,
    "CONT_BATCHING": True,
    "N_CPU_MOE": 32,
}

# Generation/Sampling configuration parameters (affect quality and token selection)
SAMPLER_DEFAULTS = {
    "TEMP": 0.4,
    "TOP_P": 0.95,
    "TOP_K": 20,
    "MIN_P": 0.0,
    "REPEAT_PENALTY": 1.05,
    "PRESENCE_PENALTY": 0.0,
    "FREQUENCY_PENALTY": None,
}

# Centralized defaults registry (merged for backwards compatibility)
DEFAULTS = {**ENGINE_DEFAULTS, **SAMPLER_DEFAULTS}

MIN_CTX_SIZE = 2048


def validate_config(cfg: dict) -> dict:
    """Return a normalized copy, or fail before any model process starts."""
    normalized = dict(cfg)
    ctx = int(normalized.get("ctx_size", normalized.get("CTX_SIZE", DEFAULTS["CTX_SIZE"])))
    flash = normalized.get("flash_attn", normalized.get("FLASH_ATTN", DEFAULTS["FLASH_ATTN"]))
    if ctx < MIN_CTX_SIZE:
        raise ConfigError(f"CTX_SIZE must be >= {MIN_CTX_SIZE}; got {ctx}")
    if flash != "on":
        raise ConfigError("FLASH_ATTN must be 'on'")
    batch = int(normalized.get("batch_size", normalized.get("BATCH_SIZE", DEFAULTS["BATCH_SIZE"])))
    ubatch = int(normalized.get("ubatch_size", normalized.get("UBATCH_SIZE", DEFAULTS["UBATCH_SIZE"])))
    if batch <= 0 or ubatch <= 0 or ubatch > batch:
        raise ConfigError("Require BATCH_SIZE > 0 and 0 < UBATCH_SIZE <= BATCH_SIZE")
    for key, value in normalized.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{key} must be finite")
    normalized["CTX_SIZE"] = ctx
    normalized["FLASH_ATTN"] = flash
    return normalized

STATE_SCHEMA_VERSION = 1
STATE_FILE = Path(__file__).resolve().parents[2] / ".autoresearch_state.json"

# Derived dynamically from DEFAULTS keys (replaces static CONFIG_KEYS tuple)
CONFIG_KEYS = tuple(DEFAULTS.keys())


def __getattr__(name: str) -> Any:
    """Expose variables from DEFAULTS dynamically for backwards compatibility."""
    if name in DEFAULTS:
        return DEFAULTS[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose all names in DEFAULTS for dynamic directory lookups."""
    return sorted(list(globals().keys()) + list(DEFAULTS.keys()))


def load_config(params: list[str] | None = None) -> dict[str, Any]:
    """Return default configuration as a dictionary."""
    if params is not None:
        return {p: DEFAULTS.get(p) for p in params if p in DEFAULTS}
    return dict(DEFAULTS)

