"""Real-SWE benchmark — DM-Code-Agent scoreboard via mini-swe-agent.

Tasks are loaded directly from `dm_agent.benchmarks.tasks` (the upstream
Python module). No on-disk fixture mirror. See `AGENTS.md` and
`docs/discovery/mini-swe-agent-benchmark.md`.
"""

from autoresearch.benchmarks.mini_swe_agent.dm_code_agent import (
    DEFAULT_SUITE,
    PINNED_DM_CODE_AGENT_COMMIT,
    discover_tasks,
    list_suites,
    load_task,
    suite_signature,
)
from autoresearch.benchmarks.mini_swe_agent.runner import run_mini_swe_agent_eval

__all__ = [
    "DEFAULT_SUITE",
    "PINNED_DM_CODE_AGENT_COMMIT",
    "discover_tasks",
    "list_suites",
    "load_task",
    "suite_signature",
    "run_mini_swe_agent_eval",
]
