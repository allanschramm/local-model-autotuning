"""Coding measurement identity shared by runners, store and reports."""

import json
from typing import Any

CODING_PROTOCOL = "coding-v2-32768-final-only"
LEGACY_CODING_PROTOCOL = "coding-legacy-unversioned"
CODING_MAX_TOKENS = 32768
# Allows a full answer at ~10 t/s plus prompt processing; not a reasoning cap.
CODING_TIMEOUT_S = 3600.0


def coding_protocol_from_config(config_json: Any) -> str:
    """Unversioned historical rows retain legacy identity; never relabel them."""
    try:
        cfg = json.loads(config_json) if isinstance(config_json, str) else config_json
    except (TypeError, ValueError):
        return LEGACY_CODING_PROTOCOL
    if not isinstance(cfg, dict):
        return LEGACY_CODING_PROTOCOL
    return str(cfg.get("coding_protocol") or cfg.get("CODING_PROTOCOL") or LEGACY_CODING_PROTOCOL)
