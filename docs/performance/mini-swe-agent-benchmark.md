# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/mini-swe-agent-benchmark.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `prompt_toolkit.output.win32.NoConsoleScreenBufferError: Found xterm-256color, while expecting a Windows console` | TTY emulator instead of native Windows console | Patch `prompt_user.py` in the uv cache to lazy-init PromptSession. See **Known Windows-TTY workaround** below. |
| "plan-first skill missing" | skill file missing | restore from upstream per **Plan-first skill** below |
| `dm_agent is not importable` or revision mismatch | pinned upstream checkout is missing or stale | install the editable checkout at commit `e4530be37e6f972d9735381e524cab4b5f732ddd` |
| All tasks fail with `tests_red` | `pytest` not on PATH in the harness venv | `.\venv\Scripts\python.exe -m pip install pytest` |
| Tasks end with `mini_repeated_format_error`, `changed_files=[]` | model emitted tag bursts or another non-executable tool format | inspect the task trajectory; verify the smoke changed a file before any full run |
| `mini_timeout_600s` on every task | `llama-server` not up or wrong port | re-run `model_up.py ... serve`; check `curl http://127.0.0.1:18080/v1/models` |
| `RuntimeError: Error calculating cost for model openai/<served-model-id>` | litellm cost-tracker doesn't know your local GGUF | set `MSWEA_COST_TRACKING=ignore_errors` (the harness does this automatically) |
| `ValidationError: 2 validation errors for AgentConfig (system_template / instance_template)` | mini-swe-agent 2.x quirk: any `-c` flag replaces the default `mini.yaml`, dropping the templates | the harness's per-task config embeds the templates inline |
| Windows console flash per task | `creationflags` not set | already handled in `runner.py`; raise an issue if regressed |
