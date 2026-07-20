"""Seed local Baseline from the tracked template when missing."""
from __future__ import annotations

from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "autoresearch" / "core"
_CFG = _CORE / "config.py"
_EXAMPLE = _CORE / "config.py.example"

if not _CFG.exists() and _EXAMPLE.exists():
    _CFG.write_text(_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
