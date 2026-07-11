# Agentic Coding Benchmarks

This repo evaluates local models across two tiers:

* **Preflight (direct code gen):** HumanEval+, MBPP+, LiveCodeBench, BigCodeBench.
  Optional single-turn checks. Exactly 10 tasks per dataset when enabled.
* **Agentic (multi-turn, tool use):** Claw-Eval quick/full tiers via `ClawEvalAdapter`
  in `autoresearch/runners/evaluation.py`. Canonical Val Score for Search.

## Tier Structure

| Tier | Tasks | Est. Time | Scoring | CLI Flag |
|------|-------|-----------|---------|----------|
| quick | 5 | ~5 min | Rule-based (tool_called, keywords, categories) | `--agentic-quick` / `--no-agentic-quick` |
| full | 15 | ~15 min | Rule-based (same) | `--agentic-full` / `--no-agentic-full` |

**Task selection policy:**
- English-only tasks (no zh variants)
- Rule-based scoring only (no `llm_judge` — fully local, no API keys)
- Quick tier: `difficulty=easy`, ≤2 mock services — smoke gate, not fair cross-model score
- Full tier: `difficulty=easy` first, then fills with `medium` — canonical Val Score
- Discovered at runtime from `claw-eval/tasks/` submodule

## Current Code Hook

List approved benchmarks:
```bash
python benchmark_search.py --list-agentic-benchmarks
```

List Claw-Eval tier task IDs:
```bash
python benchmark_search.py --list-claw-tiers
```

Run quick agentic smoke test:
```bash
python benchmark_search.py --agentic-quick --desc "agentic smoke test"
```

Run full agentic quality gate:
```bash
python benchmark_search.py --agentic-full --desc "agentic quality gate"
```

Adapter (`ClawEvalAdapter`) starts mock services, runs the agent loop against the
local OpenAI-compatible endpoint, and scores with deterministic `task.yaml` rules.
No Docker, remote judges, or external APIs.

## Scoring Rule

* Preflight coding score = `0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode`
* Agentic score = `passed / total` (simple pass@1, single trial for quick/full)
* Use exactly 10 tasks per dataset for preflight comparisons
* Claw-Eval full is the main quality gate; quick is smoke only
* Never run local benchmark validation below 100k context

## Approved Targets

`claw-eval/claw-eval` — checked out as submodule `claw-eval/`. 300 human-verified
autonomous-agent tasks across general, multimodal, and multi-turn splits.

Quick tier sample (auto-discovered, may vary):
- T002 email triage (1 mock service)
- T004 calendar scheduling (1 mock service)
- T006 email reply draft (1 mock service)
- T008 todo management (1 mock service)
- T010 contact lookup (1 mock service)
