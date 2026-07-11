# bench_config.py
# Bench-project configuration — benchmark selection, task limits, trial budget.
# Imported by the bench project only; NEVER imported by autoresearch.core.

from typing import Any
from pathlib import Path

# Benchmarks to run
INCLUDE_CODING = False
INCLUDE_NEXUS = False
INCLUDE_CLAW = False
INCLUDE_AGENTIC_QUICK = True    # smoke validation before the canonical Trial
INCLUDE_AGENTIC_FULL = True     # canonical agentic-coding Val Score
CODING_TASK_LIMIT = 10      # tasks per dataset for HE+ / MBPP+
LCB_TASK_LIMIT = 10         # LiveCodeBench v6 sample (contamination-free competitive prog)
BIGCODE_TASK_LIMIT = 10     # BigCodeBench Hard sample (library-call tasks)
AGENTIC_QUICK_TASK_LIMIT = 5    # Claw-Eval quick tier: 5 easy rule-based tasks
AGENTIC_FULL_TASK_LIMIT = 15    # Claw-Eval full tier: 15 easy+medium tasks
EVALPLUS_STRICT = True
TRIAL_BUDGET = None


def load_config(params: list[str] | None = None) -> dict[str, Any]:
    """Hot-reload bench_config.py and return current values as dict."""
    import importlib
    import sys

    mod = sys.modules[__name__]
    importlib.reload(mod)
    if params is not None:
        return {p: getattr(mod, p, None) for p in params}
    return {k: v for k, v in vars(mod).items() if k.isupper() and not k.startswith('_')}


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
        "INCLUDE_CODING", "INCLUDE_NEXUS", "INCLUDE_CLAW",
        "INCLUDE_AGENTIC_QUICK", "INCLUDE_AGENTIC_FULL",
        "CODING_TASK_LIMIT", "LCB_TASK_LIMIT", "BIGCODE_TASK_LIMIT",
        "AGENTIC_QUICK_TASK_LIMIT", "AGENTIC_FULL_TASK_LIMIT",
        "EVALPLUS_STRICT", "TRIAL_BUDGET",
    ]
    for p in bench_params:
        lines.append(f"{p} = {repr(cfg.get(p))}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
