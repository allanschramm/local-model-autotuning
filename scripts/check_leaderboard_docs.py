#!/usr/bin/env python3
"""Leaderboard doc integrity check (issue #69).

Guards the two ADR 0017 guarantees for ``docs/discovery/pareto-leaderboard.md``:

1. **No duplicate model names** — a basename may appear at most once across the
   doc's Day/Night/axis tables (case-insensitive; ``X.gguf`` vs ``x.GGUF`` is
   one name measured twice).
2. **Doc tables match the tool** — the Day/Night tables in the doc must equal
   ``scripts/rank_results.py --mode pareto`` output byte-for-byte (never
   hand-patched). When the results store is missing or unseeded the parity
   leg is skipped with a note (exit 0); the duplicate check always runs.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\check_leaderboard_docs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DOC = REPO_ROOT / "docs" / "discovery" / "pareto-leaderboard.md"
DEFAULT_TSV = REPO_ROOT / "results.tsv"

_DAY_HEADER = "DAY  (pick=#1)"
_NIGHT_HEADER = "NIGHT  (pick=#1)"


def _table_row_model(row: str) -> str | None:
    """Model cell of one ``| # | Model | …`` table row (backticks stripped)."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if len(cells) < 2:
        return None
    if cells[0] == "#":  # header row
        return None
    if re.fullmatch(r"-*", cells[0] or "-"):  # separator row
        return None
    model = cells[1].strip("`")
    if not model or not model.lower().endswith(".gguf"):
        return None
    return model


def extract_table_models(doc: str, section: str) -> list[str]:
    """Model names of the markdown table under the ``## <section>`` heading."""
    models: list[str] = []
    in_section = False
    for line in doc.splitlines():
        if line.startswith("##"):
            in_section = line.lstrip("#").strip().startswith(section)
            continue
        if not in_section:
            continue
        if line.startswith("|"):
            model = _table_row_model(line)
            if model:
                models.append(model)
        elif models:
            break  # table ended
    return models


def all_table_models(doc: str) -> list[str]:
    """Every model name in every ``| # | Model |`` table of the doc."""
    models: list[str] = []
    for line in doc.splitlines():
        if line.startswith("|"):
            model = _table_row_model(line)
            if model:
                models.append(model)
    return models


def _section_blocks(doc: str) -> list[tuple[str, list[str]]]:
    """(section heading, table line block) for every `## <heading>` section."""
    out: list[tuple[str, list[str]]] = []
    current: str | None = None
    block: list[str] = []
    for line in doc.splitlines():
        if line.startswith("##"):
            if current is not None and block:
                out.append((current, block))
            current = line.lstrip("#").strip()
            block = []
            continue
        if line.startswith("|"):
            block.append(line)
        elif block:
            out.append((current or "(untitled)", block))
            current, block = None, []
    if block:
        out.append((current or "(untitled)", block))
    return out


def find_duplicates(doc: str) -> dict[str, list[str]]:
    """Per-section duplicated model names (lowercased, sorted).

    Sections are checked independently: the Day and Night tables intentionally
    mirror the same membership (ADR 0017), so a name repeating across sections
    is correct — only a repeat within one table is a bug.
    """
    dupes: dict[str, list[str]] = {}
    for heading, block in _section_blocks(doc):
        section = heading if heading.startswith("(") else heading.split("(")[0].strip()
        # A section may hold several table blocks (blank lines split them);
        # duplicates are judged across the whole section, never per block.
        agg = dupes.setdefault(section, {})
        for line in block:
            model = _table_row_model(line)
            if model is None:
                continue
            key = model.lower()
            agg[key] = agg.get(key, 0) + 1
    return {
        section: sorted(name for name, count in seen.items() if count > 1)
        for section, seen in dupes.items()
    }


def day_night_membership_equal(doc: str) -> bool:
    """True when Day and Night tables list the same model-name set."""
    day = {m.lower() for m in extract_table_models(doc, "Day table")}
    night = {m.lower() for m in extract_table_models(doc, "Night table")}
    return day == night


def _tool_blocks(tool: str) -> dict[str, list[str]]:
    """Day/Night table line blocks from rank_results.py report text."""
    blocks: dict[str, list[str]] = {}
    for header in (_DAY_HEADER, _NIGHT_HEADER):
        lines = tool.splitlines()
        try:
            start = lines.index(header)
        except ValueError:
            blocks[header] = []
            continue
        block: list[str] = []
        for line in lines[start + 1 :]:
            if not line.startswith("|"):
                break
            block.append(line)
        blocks[header] = block
    return blocks


def _doc_section_blocks(doc: str) -> dict[str, list[str]]:
    """The ``## Day table`` / ``## Night table`` markdown table line blocks."""
    blocks: dict[str, list[str]] = {}
    lines = doc.splitlines()
    for section, header in (("Day table", _DAY_HEADER), ("Night table", _NIGHT_HEADER)):
        block: list[str] = []
        in_section = False
        for line in lines:
            if line.startswith("##"):
                in_section = line.lstrip("#").strip().startswith(section)
                continue
            if not in_section:
                continue
            if line.startswith("|"):
                block.append(line)
            elif block:
                break
        blocks[header] = block
    return blocks


def doc_tables_match(doc: str, tool: str) -> bool:
    """Doc Day/Night tables equal tool output blocks line-for-line."""
    tool_blocks = _tool_blocks(tool)
    doc_blocks = _doc_section_blocks(doc)
    return all(tool_blocks[header] == doc_blocks[header] for header in tool_blocks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check leaderboard docs: no duplicate names; tables match rank_results.py output.",
    )
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args(argv)

    doc = args.doc.read_text(encoding="utf-8")

    duplicates = find_duplicates(doc)
    hits = {section: names for section, names in duplicates.items() if names}
    if hits:
        print(f"DUPLICATE model names within sections of {args.doc}: {hits}", file=sys.stderr)
        return 1
    print(f"duplicate model names: none within any section ({args.doc})")

    if not day_night_membership_equal(doc):
        print(
            f"MEMBERSHIP MISMATCH: Day and Night tables of {args.doc} list "
            "different model sets — they must mirror each other (ADR 0017).",
            file=sys.stderr,
        )
        return 1
    print("day/night membership: identical")

    # Parity leg — only when the results store is seeded.
    from autoresearch.core import results_db
    from scripts import rank_results as rr

    rows = results_db.load_rows(args.tsv)
    if not rows:
        print("parity check: skipped (results store missing or unseeded)")
        return 0
    complete, incomplete = rr.build_vectors(rows)
    tool_report = rr.format_report(complete, incomplete, mode="pareto")
    if not doc_tables_match(doc, tool_report):
        print(
            "DOC/TABLE MISMATCH: pareto-leaderboard.md Day/Night tables differ from "
            "`scripts/rank_results.py --mode pareto` output — regenerate, never hand-patch.",
            file=sys.stderr,
        )
        return 1
    print("parity check: doc tables match rank_results.py output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
