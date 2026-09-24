# `autoresearch/benchmarks/mini_swe_agent/` — DM-Code-Agent scoreboard via `mini-swe-agent`

## Purpose

Real-SWE eval for the [`hwfengcs/DM-Code-Agent`](https://github.com/hwfengcs/DM-Code-Agent) 30-task scoreboard, driven by [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) against the operator's existing `llama-server`. It is the real-SWE sibling of SWE-lite and is observation-only in v1.

## Ownership

Repository developers. Parent: [`autoresearch/AGENTS.md`](../../AGENTS.md).

## Local Contracts

- **Pinned source:** load tasks from an editable `dm_agent` checkout at commit `e4530be37e6f972d9735381e524cab4b5f732ddd`; reject a missing or different git revision before scoring. The selected task contract is fingerprinted and stored in the trajectory and Trial detail.
- **No on-disk fixtures:** each task materializes into a fresh `tempfile.mkdtemp()` worktree at run time.
- **Hidden tests are post-agent:** stage `hidden_files` at their task-relative paths only after the agent exits, then execute the task's `hidden_test_command` with `{python}` resolved to the harness interpreter. Worktree-relative paths are containment-checked.
- **Official scope rules:** apply `allowed_changed_files` and `required_changed_files` to the pre-verifier workspace diff; a forbidden or missing required file fails the task even when pytest is green.
- **Wire-up is env-only:** set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `MSWEA_MODEL_NAME` for the child. Normalize one `openai/` provider prefix; an explicit model argument wins over a stale environment value.
- **Generation parity:** pass the Trial `GenerationParams` through `model.model_kwargs`; set each task's `agent.step_limit` and the wall-time limit.
- **Tool-call safety:** require one tool call per turn (`parallel_tool_calls=false`, `tool_choice=required`) and allow one format retry per task. These guards prevent MiMo-style tag bursts from consuming 4096 tokens per turn without executing a command.
- **Early-abort gate:** write each task trajectory through mini-swe-agent's `--output`; after two consecutive `RepeatedFormatError` exits with no workspace changes, abort the suite as incomplete instead of spending the remaining task wall-time budgets.
- **Plan-first skill:** `skills/plan-first.md` is inlined for unconstrained tasks. Constrained tasks keep the plan in the response and must not create `TODO.md`, because the official changed-file grader treats it as an out-of-scope edit.
- **No post-hoc EvalGuard gate:** EvalGuard's CLI is an execution wrapper (`audit run -t … -c …`), not a worktree auditor. This harness does not invoke it; a future integration must wrap the agent command through a supported API.
- **Process lifecycle:** `uvx` and verifier subprocesses use `autoresearch.core.process_guard.ProcessGuard`; Windows console flags are preserved.
- **Standalone selection:** `mini_swe_agent=True` disables coding, Claw, and SWE-lite evaluation for that Trial and records category `mini-swe-agent`.
- **Trajectory sidecars:** one JSONL line per task lands in `autoresearch/runners/logs/mini-swe-agent-*.jsonl`, rotated with the repository log policy.
- **No live `gh` or external network calls during a Trial:** task discovery is local to the pinned checkout; the only child network target is the configured local server.
- **Results store:** `results.db` receives `mini_swe_agent` and `mini_swe_agent_detail`; missing infrastructure leaves the score `NULL`, never a measured zero.

## Work Guidance

- Mirror `dm_code_agent.py` when adding a benchmark pack: preserve setup files, hidden files, both verifier commands, step limits, tags, and changed-file constraints.
- Keep the loader field-preserving. The suite signature is the reproducibility boundary; update the pinned commit and tests together when upstream intentionally changes.
- Keep the runner thin around `uvx mini-swe-agent`; do not reimplement the agent loop.
- Keep the skill concise. The constrained-task branch must remain compatible with official scope scoring.

## Verification

- `.\venv\Scripts\python.exe -m pytest tests/test_mini_swe_agent.py`
- Run the full suite with `.\venv\Scripts\python.exe -m pytest`.
- Operator smoke: start the configured `llama-server`, then run `python -m autoresearch.runners.run --mini-swe-agent --desc "msa-smoke"` and inspect the `mini_swe_agent` row in `results.db`.

## Child DOX Index

None.
