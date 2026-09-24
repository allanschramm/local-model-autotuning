# `mini-swe-agent` Benchmark (DM-Code-Agent scoreboard)

Daily-SWE eval: an existing OpenAI-compatible `llama-server` is queried by the [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) harness against the [DM-Code-Agent scoreboard](https://github.com/hwfengcs/DM-Code-Agent) (15 maintenance + 15 coding hidden-test tasks). This is the "real SWE" sibling of the SWE-lite `--agentic-coding` Night selector (ADR 0013); both layers live in `results.db` (the `mini_swe_agent` column is observation-only in v1 — `scripts/rank_results.py` does not render it yet).

> **Status:** v1 (operator default OFF; opt-in via `--mini-swe-agent`). Not a Pareto axis in v1 — ADR 0018 territory.
>
> **Pinned source (v1):** Install the editable DM-Code-Agent checkout at commit `e4530be37e6f972d9735381e524cab4b5f732ddd`. The loader refuses a missing or different revision and records a task-contract signature with the result. No fetch script or fixture mirror is used.

## Why this exists

The benchmark addresses the gap between a small frozen SWE-lite pack and a daily software-engineering workflow. Published comparisons of `mini-swe-agent` report material harness-level gains on the same underlying model, so this scoreboard measures the harness path without changing the model search.

## Quickstart

```powershell
# 0. (once) Install dm_agent into the harness venv.
.\venv\Scripts\python.exe -m pip install -e <path-to-cloned-dm-code-agent>

# 1. (each session) Bring up the configured model server.
.\venv\Scripts\python.exe scripts\model_up.py <model-basename> serve

# 2. (each session) Verify the server is up. Expect a non-empty `data` array.
curl http://127.0.0.1:18080/v1/models

# 3. (once) Install `uvx` (ships with the `uv` package — `pip install uv` or `pipx install uv`).
uvx --version

# 4. Run a one-task wiring smoke first.
.\venv\Scripts\python.exe -m autoresearch.runners.run `
    --mini-swe-agent `
    --mini-swe-agent-task-limit 1 `
    --desc "msa-smoke-<model-basename>"

# Inspect the newest autoresearch/runners/logs/mini-swe-agent-*.jsonl line.
# Require a non-empty changed_files list and mini_exit_status other than
# RepeatedFormatError before continuing.

# 5. Run the full benchmark only after the smoke is wired correctly.
.\venv\Scripts\python.exe -m autoresearch.runners.run `
    --mini-swe-agent `
    --desc "msa-baseline-<model-basename>"
```

`mini-swe-agent` is ephemeral — `uvx` runs it in an isolated venv so its dependencies do not pollute the harness venv. The plan-first skill (`autoresearch/benchmarks/mini_swe_agent/skills/plan-first.md`) is adapted from `esagduyu/qwen-pi-plan-vs-haiku-benchmark`; unconstrained tasks inline it into the prompt, while constrained tasks keep the plan in the response. The full-suite command also aborts after two consecutive no-progress format failures.

## What the harness does

1. **Per-Trial artifact.** One JSONL sidecar per Trial under `autoresearch/runners/logs/mini-swe-agent-<UTC>-<stem>.jsonl` (gitignored; rotated; keep last 20 per `autoresearch/AGENTS.md:26`). Each line includes the pinned upstream commit, suite signature, changed files, and verifier tail.
2. **Per-task worktree.** Each task (`task_id`, `prompt`, `setup_files`, `hidden_files`, verifier commands, step limit, and changed-file rules) is materialized into a fresh `tempfile.mkdtemp()`. Only visible starter files are present before the agent runs. Hidden files are written to their original task-relative paths only after the agent exits.
3. **mini-swe-agent invocation.** Per task: `uvx mini-swe-agent --task <prompt+skill> --model openai/<model> -c <config> --output <task-trajectory>` with `MSWEA_CONFIGURED=1` and `MSWEA_COST_TRACKING=ignore_errors`. The config carries the task's `step_limit`, wall-time limit, Trial sampling/reasoning parameters, `parallel_tool_calls=false`, `tool_choice=required`, and one format retry. Windows console flash is suppressed.
4. **No-progress abort.** The per-task trajectory exposes `RepeatedFormatError`. Two consecutive format failures with no workspace changes abort the suite as incomplete; the remaining task budgets are not spent after a systemic tool-call failure.
5. **Official scoring.** The workspace diff is checked against `allowed_changed_files` and `required_changed_files`, then the task's `hidden_test_command` runs with `{python}` resolved. Pytest and scope checks must both pass.
6. **Process cleanup.** `uvx` and verifier commands run through the repository `ProcessGuard`, which binds the complete process tree on Windows and POSIX and tears it down on timeout.
7. **No post-hoc EvalGuard gate.** EvalGuard's current CLI audits an agent command (`evalguard audit run -t … -c …`) and cannot audit a completed worktree with `--worktree`; this harness does not invoke it. A future integration must wrap execution through its supported API.
8. **Score write-back.** The Trial row writes `mini_swe_agent_val` (passed/total) and `mini_swe_agent_detail` (pinned commit, suite signature, and per-task `<id>=<verdict>`) into `results.db`. Existing rows read `NULL` for the new columns.
9. **Status recompute.** `scripts/recompute_status.py` runs automatically after every write; the observation-only column does not alter the four-axis Pareto nucleus.

## Acceptance gate

A Trial is "v1 ready" when **all four** pass:

1. `mini_swe_agent_val ≥ 0.60` for the selected suite.
2. Every task finishes within the configured per-task wall-time budget and records one trajectory line.
3. The workspace passes the task's changed-file constraints and verifier command.
4. Hidden tests are never in the prompt and never on disk while the agent runs. The post-agent verifier then stages `hidden_files` at their original task-relative paths and runs the task's `hidden_test_command`.

If any one fails, the wire-up is broken; see **Troubleshooting** below.

## Smoke (Layer 1, env-only)

The cheapest validation is the env-only loop:

```powershell
$Env:OPENAI_BASE_URL  = "http://127.0.0.1:18080/v1"
$Env:OPENAI_API_KEY   = "sk-no-auth-required"
$Env:MSWEA_MODEL_NAME = "openai/<served-model-id>"
$Env:MSWEA_CONFIGURED = "1"
$Env:MSWEA_COST_TRACKING = "ignore_errors"
uvx mini-swe-agent --task "Read .\TODO.md; print its contents."
```

A 200 OK reply to the inner HTTP call is the wire-up green-light; the harness smoke (Trial run) is the next step.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uvx mini-swe-agent` not found | `uvx` not on PATH | `pip install uv` or `pipx install uv` |
| `prompt_toolkit.output.win32.NoConsoleScreenBufferError: Found xterm-256color, while expecting a Windows console` | TTY emulator instead of native Windows console | Patch `prompt_user.py` in the uv cache to lazy-init PromptSession. See **Known Windows-TTY workaround** below. |
| "plan-first skill missing" | skill file missing | restore from upstream per **Plan-first skill** below |
| `dm_agent is not importable` or revision mismatch | pinned upstream checkout is missing or stale | install the editable checkout at commit `e4530be37e6f972d9735381e524cab4b5f732ddd` |
| All tasks fail with `tests_red` | `pytest` not on PATH in the harness venv | `.\venv\Scripts\python.exe -m pip install pytest` |
| Tasks end with `mini_repeated_format_error`, `changed_files=[]` | model emitted tag bursts or another non-executable tool format | inspect the task trajectory; verify the smoke changed a file before any full run |
| `mini_timeout_600s` on every task | `llama-server` not up or wrong port | re-run `model_up.py ... serve`; check `curl http://127.0.0.1:18080/v1/models` |
| `RuntimeError: Error calculating cost for model openai/<served-model-id>` | litellm cost-tracker doesn't know your local GGUF | set `MSWEA_COST_TRACKING=ignore_errors` (the harness does this automatically) |
| `ValidationError: 2 validation errors for AgentConfig (system_template / instance_template)` | mini-swe-agent 2.x quirk: any `-c` flag replaces the default `mini.yaml`, dropping the templates | the harness's per-task config embeds the templates inline |
| Windows console flash per task | `creationflags` not set | already handled in `runner.py`; raise an issue if regressed |

## Known Windows-TTY workaround

`mini-swe-agent 2.x` instantiates `PromptSession(history=...)` at module import time. On PowerShell-over-SSH / Git Bash / any TTY emulator that reports `TERM=xterm-256color` instead of a native Windows console, this fails hard with `NoConsoleScreenBufferError`. The harness selects the autonomous `default` agent in its per-task config, but the module-import crash can still occur before the config is read.

Workaround (one-line patch in the uv cache, idempotent):

```python
# File: <uv-cache>/archive-v0/<hash>/Lib/site-packages/minisweagent/agents/utils/prompt_user.py
# Wrap the PromptSession instantiation in a lazy try/except so the module loads
# even when the runtime can't construct a Win32Output.
```

The harness documentation tracks this as a Windows-on-non-native-console gap; long-term the fix belongs upstream in `SWE-agent/mini-swe-agent`.

## Plan-first skill

`autoresearch/benchmarks/mini_swe_agent/skills/plan-first.md` is inlined into the prompt body by the harness for tasks without changed-file constraints. Constrained tasks keep the plan in the response and do not create `TODO.md`, because the official grader treats out-of-scope files as failures. Adapted from `esagduyu/qwen-pi-plan-vs-haiku-benchmark`; the upstream URL is pinned in a header comment so re-pulls detect drift.

Re-pull when the upstream skill materially changes: read the file directly from GitHub and overwrite the local copy (the harness doesn't auto-sync).

## Schema reference

`results.db` migration (`autoresearch/core/results_db.py:_MIGRATED_COLUMNS`):

```sql
ALTER TABLE trials ADD COLUMN mini_swe_agent REAL;
ALTER TABLE trials ADD COLUMN mini_swe_agent_detail TEXT;
```

The new column is **observation-only** in v1 — it does not feed into the Pareto nucleus (ADR 0006) or Day/Night membership (ADR 0017). The four-axis Pareto Set stays unchanged.

## Out of scope for v1

- Fifth Pareto axis (ADR 0018).
- SWE-lite retirement (ADR 0019 candidate).
- SWE-rebench V2 runner (Path B — second move; v2 effort).
- Cross-model DM-Code-Agent leaderboard table (requires ≥ 3 models measured).

## Source links

- Upstream harness: <https://github.com/SWE-agent/mini-swe-agent>
- Benchmark dataset: <https://github.com/hwfengcs/DM-Code-Agent>
- Plan-first skill source: <https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark>
- Lit baseline: <https://github.com/mrguo6221/swe-bench-88-90>
- Operator research: `docs/discovery/2026-09-22-external-findings.md`
- Companion: `docs/discovery/mini-swe-agent-bridge.md`
