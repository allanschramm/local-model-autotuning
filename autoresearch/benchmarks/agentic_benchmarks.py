"""Catalog of approved agentic coding benchmarks with tiered evaluation.

Quick tier (5 min):   5 easy tasks, rule-based scoring, single trial. Smoke test.
Full tier  (15 min): 15 tasks (easy+medium), single trial. Agentic quality gate.

Task selection policy: prefer rule-based scoring (tool_called, keywords_present,
categories_present, min_length) over LLM-judge tasks. This keeps evaluation
fully local — no external API keys needed for grading.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Root paths ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLAW_EVAL_DIR = _PROJECT_ROOT / "claw-eval"
CLAW_TASKS_DIR = CLAW_EVAL_DIR / "tasks"


# ── Tier constants ────────────────────────────────────────────────────────────

QUICK_TASK_COUNT = 5   # ~5-minute smoke test
FULL_TASK_COUNT = 15   # ~15-minute agentic quality gate


# ── Spec dataclass ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgenticBenchmarkSpec:
    """Stable description of an external agentic benchmark target."""

    key: str
    name: str
    status: str
    scope: str
    harness: str
    source_url: str
    notes: str
    priority: int


# ── Catalog ───────────────────────────────────────────────────────────────────

AGENTIC_BENCHMARKS: tuple[AgenticBenchmarkSpec, ...] = (
    AgenticBenchmarkSpec(
        key="claw-eval",
        name="Claw-Eval",
        status="adopt-next",
        scope=(
            "300 human-verified autonomous-agent tasks across general, multimodal, "
            "and multi-turn splits. Quick tier: 5 easy rule-based tasks. "
            "Full tier: 15 easy+medium tasks."
        ),
        harness="claw-eval",
        source_url="https://github.com/claw-eval/claw-eval",
        notes=(
            "External harness with Pass^3 scoring. Quick/full tiers use "
            "deterministic rule-based tasks (no LLM judge needed). "
            "Needs a local-model adapter before execution."
        ),
        priority=10,
    ),
)


# ── Task discovery ────────────────────────────────────────────────────────────


def _load_task_yaml(task_dir: Path) -> dict[str, Any] | None:
    """Load a single task's task.yaml, or None on failure."""
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _has_llm_judge(task: dict) -> bool:
    """Check whether any scoring component uses an LLM judge."""
    for comp in task.get("scoring_components", []):
        check = comp.get("check", {})
        if isinstance(check, dict) and check.get("type") == "llm_judge":
            return True
    return False


def _service_count(task: dict) -> int:
    """Count mock services declared in the task."""
    return len(task.get("services", []))


def _is_english(task: dict) -> bool:
    """Check if task is English (non-zh variant)."""
    lang = task.get("prompt", {}).get("language", "en")
    return lang == "en"


def discover_claw_tasks(
    difficulty: str | None = None,
    max_services: int | None = None,
    exclude_llm_judge: bool = True,
    english_only: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Scan claw-eval/tasks/ for tasks matching criteria.

    Returns list of (task_dict, task_dir_name) sorted by task_id.
    """
    if not CLAW_TASKS_DIR.exists():
        return []

    results: list[tuple[str, dict]] = []

    for task_dir in sorted(CLAW_TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = _load_task_yaml(task_dir)
        if task is None:
            continue

        if difficulty and task.get("difficulty") != difficulty:
            continue
        if max_services is not None and _service_count(task) > max_services:
            continue
        if exclude_llm_judge and _has_llm_judge(task):
            continue
        if english_only and not _is_english(task):
            continue

        results.append((task_dir.name, task))

    # Sort by task_id number for stable ordering
    results.sort(key=lambda x: x[0])

    if limit:
        results = results[:limit]

    return [task for _, task in results]


def get_quick_tier_tasks() -> list[str]:
    """Return task directory names for the 5-minute quick agentic smoke test.

    Criteria: easy, ≤2 services, rule-based scoring, English.
    """
    tasks = discover_claw_tasks(
        difficulty="easy",
        max_services=2,
        exclude_llm_judge=True,
        english_only=True,
        limit=QUICK_TASK_COUNT,
    )
    return [t.get("task_id", "") for t in tasks]


def get_full_tier_tasks() -> list[str]:
    """Return task directory names for the 15-minute full agentic eval.

    Criteria: easy OR medium, rule-based scoring, English.
    Picks easy tasks first, then fills with medium.
    """
    easy = discover_claw_tasks(
        difficulty="easy",
        exclude_llm_judge=True,
        english_only=True,
        limit=FULL_TASK_COUNT,
    )

    if len(easy) >= FULL_TASK_COUNT:
        return [t.get("task_id", "") for t in easy[:FULL_TASK_COUNT]]

    # Fill remaining slots with medium tasks
    remaining = FULL_TASK_COUNT - len(easy)
    medium = discover_claw_tasks(
        difficulty="medium",
        exclude_llm_judge=True,
        english_only=True,
        limit=remaining,
    )

    combined = easy + medium
    return [t.get("task_id", "") for t in combined]


# ── Public API ────────────────────────────────────────────────────────────────


def list_agentic_benchmarks(status: str | None = None) -> list[AgenticBenchmarkSpec]:
    """Return benchmark specs sorted by migration priority."""
    specs = sorted(AGENTIC_BENCHMARKS, key=lambda spec: spec.priority)
    if status is None:
        return specs
    return [spec for spec in specs if spec.status == status]


def format_agentic_benchmarks(status: str | None = None) -> str:
    """Format benchmark specs for CLI display."""
    rows = [
        "key\tstatus\tharness\tname",
        *(
            f"{spec.key}\t{spec.status}\t{spec.harness}\t{spec.name}"
            for spec in list_agentic_benchmarks(status=status)
        ),
    ]
    return "\n".join(rows)


def format_claw_tiers() -> str:
    """Pretty-print the quick/full task tiers."""
    quick = get_quick_tier_tasks()
    full = get_full_tier_tasks()

    lines = [
        "=== Claw-Eval Agentic Tiers ===",
        "",
        f"Quick tier ({QUICK_TASK_COUNT} tasks, ~5 min):",
        *(f"  {tid}" for tid in quick),
        "",
        f"Full tier ({FULL_TASK_COUNT} tasks, ~15 min):",
        *(f"  {tid}" for tid in full),
        "",
        "Selection: easy, rule-based scoring, English, ≤2 services (quick) / ≤5 (full).",
    ]
    return "\n".join(lines)
