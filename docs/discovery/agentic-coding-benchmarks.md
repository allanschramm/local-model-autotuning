# Agentic Coding Benchmarks

This repo evaluates local models across two tiers:

* **Preflight (direct code gen):** HumanEval+, MBPP+, LiveCodeBench, BigCodeBench.
  Fast single-turn smoke tests. Already integrated.
* **Agentic (multi-turn, tool use):** Claw-Eval quick/full tiers. Long-horizon
  autonomous-agent evaluation. Adapter under construction.

## Tier Structure

| Tier | Tasks | Est. Time | Scoring | CLI Flag |
|------|-------|-----------|---------|----------|
| quick | 5 | ~5 min | Rule-based (tool_called, keywords, categories) | `--agentic-quick` |
| full | 15 | ~15 min | Rule-based (same) | `--agentic-full` |

**Task selection policy:**
- English-only tasks (no zh variants)
- Rule-based scoring only (no `llm_judge` — fully local, no API keys)
- Quick tier: `difficulty=easy`, ≤2 mock services
- Full tier: `difficulty=easy` first, then fills with `medium`
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

Run quick agentic smoke test (adapter placeholder — prints task IDs, scores 0.0):
```bash
python benchmark_search.py --agentic-quick --desc "agentic smoke test"
```

Run full agentic quality gate:
```bash
python benchmark_search.py --agentic-full --desc "agentic quality gate"
```

## What's Missing (Adapter)

The agentic eval block in `evaluation.py` prints selected task IDs but does not
execute them yet. The local-model adapter needs:

1. **Mock service launcher** — start `python mock_services/<name>/server.py` per task
2. **Agent loop** — model generates tool calls, mock services respond, repeat
3. **Rule-based scorer** — apply `scoring_components` (tool_called, keywords_present,
   categories_present, min_length) from each task's `task.yaml`
4. **Result collector** — pass/fail per task, aggregate into `agentic_val`

Until the adapter exists, `agentic_val` defaults to 0.0 and the tier is logged
as `agentic-quick` / `agentic-full` in `results.tsv`.

## Scoring Rule

* Preflight coding score = `0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode`
* Agentic score = `passed / total` (simple pass@1, single trial for quick/full)
* Use exactly 10 tasks per dataset for preflight comparisons
* Agentic benchmarks become the main quality gate once the adapter exists
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
