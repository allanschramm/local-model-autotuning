# bench_config.py
# Bench-project configuration — benchmark selection, task limits, trial budget.
# Imported by the bench project only; NEVER imported by autoresearch.core.

from pathlib import Path
from typing import Any

# Benchmarks to run
INCLUDE_CODING = True  # coding-10 → complete Objective Vector (Pareto acceptance; issue #8)
INCLUDE_AGENTIC_QUICK = True  # smoke validation before the canonical Trial
INCLUDE_AGENTIC_FULL = True  # Claw-Eval full → agentic Objective Vector axis
INCLUDE_AGENTIC_CODING = False  # SWE-lite issue loop; Night selector (ADR 0013); opt-in
INCLUDE_MINI_SWE_AGENT = False  # DM-Code-Agent scoreboard via mini-swe-agent; Night selector (sibling to agentic_coding); opt-in
MINI_SWE_AGENT_TASK_LIMIT = (
    0  # 0 = full suite; >0 = first N tasks (bounds wall-time for smoke runs)
)
MINI_SWE_AGENT_SUITE = "all"  # upstream DM-Code-Agent suite: coding | maintenance | all
CODING_TASK_LIMIT = 10  # tasks per dataset for HE+ / MBPP+
LCB_TASK_LIMIT = 10  # LiveCodeBench v6 sample (contamination-free competitive prog)
BIGCODE_TASK_LIMIT = 10  # BigCodeBench Hard sample (library-call tasks)
AGENTIC_QUICK_TASK_LIMIT = 5  # Claw-Eval quick tier: 5 easy rule-based tasks
AGENTIC_FULL_TASK_LIMIT = 15  # Claw-Eval full tier: 15 easy+medium tasks
EVALPLUS_STRICT = True


def load_config(params: list[str] | None = None) -> dict[str, Any]:
    """Hot-reload bench_config.py and return current values as dict."""
    import importlib
    import sys

    mod = sys.modules[__name__]
    importlib.reload(mod)
    if params is not None:
        return {p: getattr(mod, p, None) for p in params}
    return {k: v for k, v in vars(mod).items() if k.isupper() and not k.startswith("_")}


def write_config(cfg: dict[str, Any], path: str | Path | None = None) -> None:
    """Persist bench config dict back to bench_config.py."""
    from pathlib import Path

    if path is None:
        path = Path(__file__).resolve()
    path = Path(path)

    lines = [
        "# bench_config.py",
        "# Bench-project configuration — benchmark selection, task limits, trial budget.",
        "# Imported by the bench project only; NEVER imported by autoresearch.core.",
        "",
    ]

    bench_params = [
        "INCLUDE_CODING",
        "INCLUDE_AGENTIC_QUICK",
        "INCLUDE_AGENTIC_FULL",
        "INCLUDE_AGENTIC_CODING",
        "INCLUDE_MINI_SWE_AGENT",
        "MINI_SWE_AGENT_TASK_LIMIT",
        "MINI_SWE_AGENT_SUITE",
        "CODING_TASK_LIMIT",
        "LCB_TASK_LIMIT",
        "BIGCODE_TASK_LIMIT",
        "AGENTIC_QUICK_TASK_LIMIT",
        "AGENTIC_FULL_TASK_LIMIT",
        "EVALPLUS_STRICT",
    ]
    for p in bench_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
