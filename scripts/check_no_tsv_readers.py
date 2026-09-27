#!/usr/bin/env python3
"""check_no_tsv_readers.py — guard rail for the SQLite-first migration.

The results store is canonical SQLite (`results.db`) with an append-only
mirror `results.tsv` for legacy export. Production readers must route through
``autoresearch.core.results_db.load_rows()`` (DB-first, TSV fallback). This
script fails (exit 1) if a new production code path reads ``results.tsv``
directly outside the small intentional allowlist.

Usage::

    python scripts/check_no_tsv_readers.py

Intentionally-allowed direct TSV readers (legacy seams):

* ``autoresearch/core/results_db.py`` — SQLite-mirror fallback path inside
  ``_read_tsv`` (only hits when the DB is missing/empty).
* ``autoresearch/runners/run.py`` — mirror-writer (``_replace_tsv``) and
  ``get_previous_best`` (non-SQL fallback when the DB is empty).
* ``tests/`` — every test monkeypatches a ``tmp_path / "results.tsv"`` and
  exercises the legacy format directly. Legitimate.
* ``docs/sessions/`` — historical analyses reference ``results.tsv`` by name
  to reconstruct graphs from past trials.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Direct reader/writer sites that are part of the legacy mirror contract.
INTENTIONAL_SITES: tuple[Path, ...] = (
    REPO_ROOT / "autoresearch" / "core" / "results_db.py",
    REPO_ROOT / "autoresearch" / "runners" / "run.py",
)

# Directories whose TSV usage is allowed (tests, historical docs).
ALLOWLIST_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "tests",
    REPO_ROOT / "docs" / "sessions",
    REPO_ROOT / ".git",
    REPO_ROOT / "rust",  # Future Rust bindings won't open TSV at all.
)

# Patterns that indicate an active TSV read or open.
READER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"csv\.DictReader\s*\([^)]*delimiter\s*=\s*[\"']\\t[\"']"),
    re.compile(r"read_csv\s*\([^)]*sep\s*=\s*[\"']\\t[\"']"),
    # Opening a path that ends with `.tsv` to read it (rough heuristic).
    re.compile(r"open\s*\([^)]*\.tsv", re.IGNORECASE),
)


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_allowed(path: Path, self_path: Path) -> bool:
    if path == self_path:
        return True
    if path in INTENTIONAL_SITES:
        return True
    return any(_is_under(path, d) for d in ALLOWLIST_DIRS)


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        # Skip caches and venvs anyway (rg-style ignores); we are explicit.
        parts = set(path.parts)
        if parts & {"__pycache__", ".venv", "venv", ".git"}:
            continue
        yield path


def main() -> int:
    self_path = Path(__file__).resolve()
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files(REPO_ROOT):
        if _is_allowed(path, self_path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in READER_PATTERNS:
            for m in pat.finditer(content):
                line_no = content.count("\n", 0, m.start()) + 1
                violations.append((path, line_no, m.group(0).strip()))

    if violations:
        print(
            "ERROR: direct results.tsv reader detected outside the SQLite-first",
            "allowlist.",
        )
        print(
            "Production code must read results.db via",
            "autoresearch.core.results_db.load_rows().",
        )
        print()
        for rel, line, snippet in violations:
            print(f"  {rel.relative_to(REPO_ROOT)}:{line}: {snippet}")
        return 1
    print("OK: no direct results.tsv readers found outside allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
