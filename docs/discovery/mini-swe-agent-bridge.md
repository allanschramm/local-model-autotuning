# mini-swe-agent → local llama-server bridge

This guide connects the tracked DM-Code-Agent benchmark wrapper to an existing OpenAI-compatible `llama-server`. For the maintained benchmark workflow, use [`mini-swe-agent-benchmark.md`](./mini-swe-agent-benchmark.md).

## Prerequisites

1. Start the configured local server and verify its model list:

   ```powershell
   curl http://127.0.0.1:18080/v1/models
   ```

   Copy the returned model identifier. The benchmark wrapper adds exactly one `openai/` provider prefix when needed.

2. Install `uvx`:

   ```powershell
   uvx --version
   ```

3. Install the pinned DM-Code-Agent editable checkout:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -e <path-to-dm-code-agent-checkout>
   ```

   The checkout must be at commit `e4530be37e6f972d9735381e524cab4b5f732ddd`; the wrapper refuses a missing or different revision.

## Run the maintained benchmark

```powershell
$Env:OPENAI_BASE_URL = "http://127.0.0.1:18080/v1"
$Env:OPENAI_API_KEY = "sk-no-auth-required"
$Env:MSWEA_MODEL_NAME = "openai/<served-model-id>"

.\venv\Scripts\python.exe -m autoresearch.runners.run `
    --mini-swe-agent `
    --mini-swe-agent-task-limit 1 `
    --desc "msa-bridge-smoke"
```

Inspect the newest `autoresearch/runners/logs/mini-swe-agent-*.jsonl` line before starting the full suite. The wiring smoke passes only when `changed_files` is non-empty and `mini_exit_status` is not `RepeatedFormatError`.

The wrapper creates a fresh temporary worktree per task, passes the Trial sampling parameters to `model.model_kwargs`, requires one tool call per turn, writes a task-local trajectory through `--output`, enforces each task's `step_limit`, stages hidden files only after the agent exits, and runs the upstream verifier command. Two consecutive no-progress format failures abort the suite as incomplete. It records the suite signature and pinned commit in the trajectory and `results.db` detail.

## Direct smoke

For a wiring-only check outside the Trial harness:

```powershell
$Env:OPENAI_BASE_URL = "http://127.0.0.1:18080/v1"
$Env:OPENAI_API_KEY = "sk-no-auth-required"
$Env:MSWEA_MODEL_NAME = "openai/<served-model-id>"
$Env:MSWEA_CONFIGURED = "1"
$Env:MSWEA_COST_TRACKING = "ignore_errors"

uvx mini-swe-agent --task "Read README.md and print its first heading."
```

A successful local HTTP response confirms endpoint and model routing. It does not replace the maintained benchmark's hidden verifier and changed-file checks.

## Harness boundaries

- The agent is allowed to work only inside its temporary worktree.
- Hidden tests are absent while the agent runs and are written at their original task-relative paths before verification.
- `uvx` and pytest are launched through `autoresearch.core.process_guard.ProcessGuard`; Windows descendants are bound to a Job Object and POSIX descendants are handled through the process-group lifecycle.
- The plan-first skill is used for unconstrained tasks. Constrained tasks keep the plan in the response so an out-of-scope `TODO.md` cannot invalidate the task.
- EvalGuard is not invoked post hoc. Its CLI is an execution wrapper and requires a task command; a future integration must wrap the agent through its supported API.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uvx` is unavailable | Install `uv` or run `uvx` through an installed `pipx` environment. |
| `dm_agent` cannot be imported | Install the pinned editable checkout. |
| Revision mismatch | Check out the pinned commit or update the loader pin and its regression test together. |
| Model not found | Use the exact identifier returned by `/v1/models`; keep one `openai/` prefix. |
| `mini_timeout_*` | Verify the server, port, and model identifier before increasing the wall-time limit. |
| `tests_red` | Inspect the trajectory tail and verifier output; the agent result alone is not the score. |
| `mini_repeated_format_error` with no changed files | Stop before the full suite and inspect the tag-style tool-call output. |
| Scope failure | Check `changed_files`; the task's allowed/required file contract is enforced independently of pytest. |

## Sources

- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [DM-Code-Agent](https://github.com/hwfengcs/DM-Code-Agent)
- [plan-first skill source](https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark)
