#!/usr/bin/env python3
"""Rank models from the results store (Pareto / Day / Night / claw / coding).

Reads the canonical SQLite store (``results.db``) first and falls back to the
legacy ``results.tsv`` append-log when the DB is missing or unseeded. This CLI
is the agent-facing query surface so ranking never needs ad-hoc temp scripts.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode claw
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode coding
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_root_on_sys_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_sys_path()

from autoresearch.core import results_db
from autoresearch.core.classify import MORRIS_SCREEN_PROFILE, fp_from_config_json

DEFAULT_TSV = REPO_ROOT / "results.tsv"

_DESC_TPS_RE = re.compile(r"(?:bench_tg|TPS)=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_DESC_CTX_RE = re.compile(r"\bctx=([0-9]+)\b", re.IGNORECASE)

# ADR 0017: models whose iq_min sits within ±0.05 of a neighbor form a
# near-tie band; speed (Day) or ctx (Night) breaks ties among non-dominated
# candidates, but never demotes a model clearly superior in agentic and coding.
NEAR_TIE_BAND = 0.05

OK_OUTCOMES = {"", "OK"}


@dataclass(frozen=True)
class Point:
    model: str
    ctx: int
    tps: float
    agentic: float
    coding: float
    fp: str | None = None
    agentic_coding: float | None = None

    @property
    def iq_min(self) -> float:
        return min(self.agentic, self.coding)

    @property
    def complete(self) -> bool:
        return self.agentic >= 0.0 and self.coding >= 0.0


def _fnum(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _valid_score(raw: Any) -> float | None:
    value = _fnum(raw)
    if value is None or value < 0.0 or value > 1.0:
        return None
    return value


def _ctx_from_description(description: str) -> int | None:
    match = _DESC_CTX_RE.search(description or "")
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _tps_from_description(description: str) -> float | None:
    # Prefer bench_tg= over TPS= when both appear.
    matches = _DESC_TPS_RE.findall(description or "")
    if not matches:
        return None
    # Last numeric wins when both TPS= and bench_tg= present (bench_tg usually last).
    value = float(matches[-1])
    return value if value > 0 else None


def _ctx_of(row: dict[str, str]) -> int | None:
    direct = _fnum(row.get("ctx"))
    if direct is not None and direct > 0:
        return int(direct)
    raw = (row.get("config_json") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            for key in ("CTX_SIZE", "ctx_size", "ctx"):
                value = _fnum(data.get(key))
                if value is not None and value > 0:
                    return int(value)
    return _ctx_from_description(row.get("description") or "")


def _tps_of(row: dict[str, str]) -> float | None:
    for key in ("bench_tg", "tps"):
        value = _fnum(row.get(key))
        if value is not None and value > 0:
            return value
    return _tps_from_description(row.get("description") or "")


def _is_measurement_row(row: dict[str, str]) -> bool:
    outcome = (row.get("outcome") or "").strip()
    return outcome in OK_OUTCOMES


def load_rows(path: Path) -> list[dict[str, str]]:
    """Canonical-first store read (results.db, legacy results.tsv fallback)."""
    return results_db.load_rows(path)


def _axis_values(row: dict[str, str]) -> tuple[float | None, float | None]:
    """agentic/coding axis values of one row: columns when populated, else category.

    The modern write path records a combined vector (agentic-full + coding-10
    measured in one run) with both columns populated; the split form stores
    agentic-full / 10-task category rows. Legacy rows have neither.
    """
    agentic = _valid_score(row.get("agentic"))
    coding = _valid_score(row.get("coding"))
    agentic_coding = _valid_score(row.get("agentic_coding"))
    score = _valid_score(row.get("val_score"))
    category = (row.get("category") or "").strip()
    if agentic is None and category == "agentic-full":
        agentic = score
    if coding is None and category == "10-task":
        coding = score
    if agentic_coding is None and category == "agentic-coding":
        agentic_coding = score
    return agentic, coding, agentic_coding


def build_vectors(
    rows: Sequence[dict[str, str]],
) -> tuple[list[Point], list[Point]]:
    """One representative Point per GGUF basename (ADR 0017).

    A display run = rows sharing one Fingerprint (config_json) inside the
    basename; rows without a config_json (legacy) merge by basename. Best
    per-axis values are merged within a Fingerprint group only. The
    basename's representative run is the complete group with the highest
    iq_min = min(agentic, coding) — its own ctx/TPS/agentic/coding are
    shown, never a composite stitched from different runs. Split-config
    basenames (agentic and coding measured at different Fingerprints) have
    no complete display run and stay incomplete until re-measured.
    """
    # (model, fp) → group accumulator. fp=None groups legacy fingerprint-less rows.
    groups: dict[tuple[str, str | None], dict[str, Any]] = {}

    def _group(model: str, fp: str | None) -> dict[str, Any]:
        return groups.setdefault(
            (model, fp),
            {"agentic": None, "coding": None, "agentic_coding": None, "tps": None, "ctx": None},
        )

    for row in rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        if (row.get("evaluation_profile") or "").strip() == MORRIS_SCREEN_PROFILE:
            continue  # ADR 0016: reps=1 screen TPS must not set the basename axis
        if not _is_measurement_row(row):
            continue
        fp = fp_from_config_json(row.get("config_json"))
        group = _group(model, fp)
        tps = _tps_of(row)
        if tps is not None and (group["tps"] is None or tps > group["tps"]):
            group["tps"] = tps
        ctx = _ctx_of(row)
        if ctx is not None and ctx > 0 and (group["ctx"] is None or ctx > group["ctx"]):
            group["ctx"] = ctx
        agentic, coding, agentic_coding = _axis_values(row)
        if agentic is not None and (group["agentic"] is None or agentic > group["agentic"]):
            group["agentic"] = agentic
        if coding is not None and (group["coding"] is None or coding > group["coding"]):
            group["coding"] = coding
        if agentic_coding is not None and (
            group["agentic_coding"] is None or agentic_coding > group["agentic_coding"]
        ):
            group["agentic_coding"] = agentic_coding

    complete: list[Point] = []
    incomplete: list[Point] = []
    by_model: dict[str, list[tuple[str | None, dict[str, Any]]]] = {}
    for (model, fp), group in groups.items():
        by_model.setdefault(model, []).append((fp, group))

    for model in sorted(by_model):
        candidates = by_model[model]
        complete_groups = [
            (fp, g) for fp, g in candidates if g["agentic"] is not None and g["coding"] is not None
        ]
        if complete_groups:
            # Representative run = highest iq_min; deterministic tie-breaks.
            fp, g = min(
                complete_groups,
                key=lambda item: (
                    -min(item[1]["agentic"], item[1]["coding"]),
                    -(item[1]["tps"] or 0.0),
                    -(item[1]["ctx"] or 0),
                    item[0] or "",
                ),
            )
            complete.append(
                Point(
                    model=model,
                    ctx=int(g["ctx"] or 0),
                    tps=float(g["tps"] or 0.0),
                    agentic=float(g["agentic"]),
                    coding=float(g["coding"]),
                    fp=fp,
                    agentic_coding=g["agentic_coding"],
                )
            )
            continue
        # Incomplete: keep the legacy best-per-axis merge so a partially
        # measured basename still surfaces its partial vector.
        agentic = max(
            (g["agentic"] for _, g in candidates if g["agentic"] is not None), default=-1.0
        )
        coding = max((g["coding"] for _, g in candidates if g["coding"] is not None), default=-1.0)
        tps = max((g["tps"] for _, g in candidates if g["tps"] is not None), default=0.0)
        ctx = max((g["ctx"] for _, g in candidates if g["ctx"] is not None), default=0)
        ac = max(
            (g["agentic_coding"] for _, g in candidates if g["agentic_coding"] is not None),
            default=None,
        )
        fp = next((f for f, g in candidates if f and g["agentic"] is not None), None)
        incomplete.append(
            Point(
                model=model,
                ctx=int(ctx),
                tps=float(tps),
                agentic=max(agentic, 0.0),
                coding=max(coding, 0.0),
                fp=fp,
                agentic_coding=ac,
            )
        )
    return complete, incomplete


# Rank membership is a leaderboard, not a frontier (ADR 0017): every complete
# point passes through — the alias keeps callers/tests stable.
def pareto_front(points: Sequence[Point]) -> list[Point]:
    return [p for p in points if p.complete]


def pick_day(front: Sequence[Point]) -> Point | None:
    ranked = day_table(front)
    return ranked[0] if ranked else None


def pick_night(front: Sequence[Point]) -> Point | None:
    ranked = night_table(front)
    return ranked[0] if ranked else None


def quality_dominates(a: Point, b: Point) -> bool:
    """True if a is >= b on both agentic and coding, and > on at least one."""
    return (a.agentic >= b.agentic and a.coding >= b.coding) and (
        a.agentic > b.agentic or a.coding > b.coding
    )


def _sort_band(band: Sequence[Point], tie_key: Any) -> list[Point]:
    """Sort a near-tie band by tie_key while strictly respecting quality dominance.

    If a quality-dominates b (a is clearly superior in agentic and coding),
    a must always rank before b (ADR 0017). Tie-breaking (speed for Day,
    context for Night) resolves ties and non-dominated trade-offs.
    """
    in_degree = {p: 0 for p in band}
    for a in band:
        for b in band:
            if a is not b and quality_dominates(a, b):
                in_degree[b] += 1

    remaining = list(band)
    result: list[Point] = []
    while remaining:
        eligible = [p for p in remaining if in_degree[p] == 0]
        eligible.sort(key=tie_key)
        chosen = eligible[0]
        result.append(chosen)
        remaining.remove(chosen)
        for b in remaining:
            if quality_dominates(chosen, b):
                in_degree[b] -= 1
    return result


def _band_sorted(points: Sequence[Point], tie_key: Any) -> list[Point]:
    """IQ-first sort with the ADR 0017 ±0.05 near-tie band.

    Points sorted by iq_min descending; consecutive neighbors whose iq_min
    differs by <= NEAR_TIE_BAND chain into one near-tie band, ordered by the
    lens tie_key while strictly respecting quality dominance. Outside a band,
    iq_min strictly rules. Deterministic: model name is the final tie-break.
    """
    order = sorted(points, key=lambda p: (-p.iq_min, p.model))
    out: list[Point] = []
    band: list[Point] = []
    for p in order:
        if band and p.iq_min >= band[0].iq_min - NEAR_TIE_BAND:
            band.append(p)
        else:
            if band:
                out.extend(_sort_band(band, tie_key))
            band = [p]
    if band:
        out.extend(_sort_band(band, tie_key))
    return out


def day_table(front: Sequence[Point]) -> list[Point]:
    """Every complete model, IQ-first; near-ties broken by TPS (ADR 0017)."""
    return _band_sorted(
        [p for p in front if p.complete],
        lambda p: (-p.tps, -p.ctx, -p.iq_min, -(p.agentic + p.coding), p.model),
    )


def night_table(front: Sequence[Point]) -> list[Point]:
    """Every complete model, IQ-first; near-ties broken by ctx (ADR 0017)."""
    return _band_sorted(
        [p for p in front if p.complete],
        lambda p: (-p.ctx, -p.iq_min, -(p.agentic + p.coding), -p.tps, p.model),
    )


def _rank_axis(
    rows: Sequence[dict[str, str]],
    category: str,
    *,
    score_column: str | None = None,
) -> list[tuple[str, float, float | None, int | None]]:
    best: dict[str, tuple[float, float | None, int | None]] = {}
    for row in rows:
        if not _is_measurement_row(row):
            continue
        model = (row.get("model") or "").strip()
        score = _valid_score(row.get(score_column)) if score_column else None
        if score is None and (row.get("category") or "").strip() == category:
            score = _valid_score(row.get("val_score"))
        if not model or score is None:
            continue
        prev = best.get(model)
        if prev is None or score > prev[0]:
            best[model] = (score, _tps_of(row), _ctx_of(row))
    ranked = [(model, score, tps, ctx) for model, (score, tps, ctx) in best.items()]
    ranked.sort(key=lambda item: (-item[1], -(item[2] or 0.0), item[0]))
    return ranked


def _fmt_ctx(ctx: int) -> str:
    if ctx <= 0:
        return "-"
    # Repo leaderboards use 65k/32k/131k (= ctx // 1000), not KiB.
    if ctx >= 1000:
        return f"{ctx // 1000}k"
    return str(ctx)


def _md_table(points: Sequence[Point]) -> list[str]:
    if not points:
        return ["(none)"]
    headers = ("#", "Model", "ctx", "TPS", "agentic", "coding")
    rows: list[tuple[str, ...]] = []
    for i, p in enumerate(points, 1):
        rows.append(
            (
                str(i),
                p.model,
                _fmt_ctx(p.ctx),
                f"{p.tps:.1f}",
                f"{p.agentic:.4f}",
                f"{p.coding:.4f}",
            )
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            # left-align model; right-align numeric-ish cols
            if i == 1:
                parts.append(f" {cell:<{widths[i]}} ")
            else:
                parts.append(f" {cell:>{widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return out


def _fmt_axis_row(
    rank: int,
    model: str,
    score: float,
    tps: float | None,
    ctx: int | None,
) -> tuple[str, str, str, str, str]:
    tps_s = f"{tps:.1f}" if tps is not None else "-"
    ctx_s = _fmt_ctx(ctx) if ctx is not None else "-"
    return (str(rank), model, ctx_s, tps_s, f"{score:.4f}")


def _md_axis_table(
    ranked: Sequence[tuple[str, float, float | None, int | None]],
) -> list[str]:
    if not ranked:
        return ["(none)"]
    headers = ("#", "Model", "ctx", "TPS", "score")
    rows = [_fmt_axis_row(i, m, s, t, c) for i, (m, s, t, c) in enumerate(ranked, 1)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 1:
                parts.append(f" {cell:<{widths[i]}} ")
            else:
                parts.append(f" {cell:>{widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return out


def format_report(
    complete: Sequence[Point],
    incomplete: Sequence[Point],
    *,
    mode: str,
    rows: Sequence[dict[str, str]] | None = None,
) -> str:
    del incomplete  # kept in signature for callers; default view is Day/Night only
    lines: list[str] = []
    front = pareto_front(complete)

    if mode in ("pareto", "day", "all"):
        day_rows = day_table(front)
        lines.append("DAY  (pick=#1)")
        lines.extend(_md_table(day_rows))

    if mode in ("pareto", "night", "all"):
        if lines:
            lines.append("")
        night_rows = night_table(front)
        lines.append("NIGHT  (pick=#1)")
        lines.extend(_md_table(night_rows))

    if mode in ("claw", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("CLAW-FULL")
        lines.extend(_md_axis_table(_rank_axis(rows, "agentic-full")))

    if mode in ("coding", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("CODING-10")
        lines.extend(_md_axis_table(_rank_axis(rows, "10-task")))

    if mode in ("agentic-coding", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("AGENTIC-CODING")
        lines.extend(
            _md_axis_table(_rank_axis(rows, "agentic-coding", score_column="agentic_coding"))
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models from the results store (results.db, legacy TSV fallback; ADR 0006/0009).",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help="Base path of the store: reads <base>.db first, <base> TSV as fallback (default: repo root results.tsv)",
    )
    parser.add_argument(
        "--mode",
        choices=("pareto", "day", "night", "claw", "coding", "agentic-coding", "all"),
        default="pareto",
        help="Report view (default: pareto = Day + Night markdown tables)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.tsv
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 1
    rows = load_rows(path)
    complete, incomplete = build_vectors(rows)
    report = format_report(
        complete,
        incomplete,
        mode=args.mode,
        rows=rows,
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
