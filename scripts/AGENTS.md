# scripts — auto-tuning operator tools

## Purpose
Provision, inspect and serve local runtimes; verify bounded tuning and installation. Quality evaluator and leaderboard tools are archived.

## Ownership
Repository developers and operators.

## Local Contracts
- Use the repo venv and core runtime resolution/safeguards.
- setup-check.sh imports all root runtime requirements through the project venv; missing mandatory packages fail. Tool probes do not replace model validation.
- Prefer compatible prebuilt releases; engine compilation requires an explicit operator request.
- `model_up.py` (global `model-up`) must work from any cwd: resolve `--model` and the Fingerprint `SPEC_DRAFT_MODEL` to absolute paths, and spawn `llama-server` with `cwd=REPO_ROOT`.
- `models/aliases/model-up.cmd` (local Windows launcher, **never committed**) must invoke the **repo venv** python, falling back to bare `python` only when absent. A bare system `python` has no `gguf`, and `model_up.py` would then drop `--n-cpu-moe` on a MoE (`GGUFMetadataError` now makes that fail closed rather than spill).
- `model_up.py` serves the Fingerprint file (issue #52, ADR 0014 phase 2): alias gives identity + local bind (`model`, `alias`, `host`, `port`, `llama_cpp_root`); engine flags and optional sampler defaults come from `fingerprints/<stem>.json` via the same subset/order as the trial runner (incl. SPEC auto-MTP + `--n-cpu-moe` auto). Missing/invalid file fails closed (exit 1, never alias-flag soup); stale alias `flags:` are ignored with a NOTE. Single-load gate unchanged. Engine flags pass through `load_mode_flag` (`NO_MMAP`/`MLOCK` -> `--load-mode`) and `build_command` runs `assert_flags_supported`, so an engine bump that dropped a flag fails before spawn instead of at startup.
- OpenVINO is optional. bench_openvino.py measures runtime throughput, not model quality.

## Work Guidance
- Run finite JSON plans through python -m autoresearch.tuning. Explicit Fingerprint application uses apply_fingerprint.py.
- Do not revive retired quality CLI or TSV ranking paths.

## Verification
- run_validate.py: shared Ruff/pytest checks.
- test_tuning_e2e.py: native lifetime, HTTP, fixed PPL/context/frontier and SQLite.
- test_tuning_consumers_e2e.py: retired surfaces absent, dashboard and historical SQLite rows unchanged.
- test_tuning_install_e2e.py: minimal fresh venv, readiness positive/negative controls and both SQLite E2Es.
- watch_validate.py must exit 0 after each push; Windows tests do not replace Linux CI.

## Child DOX Index
- [apply_fingerprint.py](apply_fingerprint.py) — operator tool.
- [bench_openvino.py](bench_openvino.py) — operator tool.
- [build-llamacpp.py](build-llamacpp.py) — operator tool.
- [check_hardware.py](check_hardware.py) — operator tool.
- [discover-models.sh](discover-models.sh) — operator tool.
- [llama-monitor.sh](llama-monitor.sh) — operator tool.
- [measure_vram_peak.py](measure_vram_peak.py) — operator tool.
- [model_info.py](model_info.py) — operator tool.
- [model_up.py](model_up.py) — operator tool.
- [run_pytest_hook.py](run_pytest_hook.py) — operator tool.
- [run_validate.py](run_validate.py) — operator tool.
- [serve-config.py](serve-config.py) — operator tool.
- [setup-check.sh](setup-check.sh) — operator tool.
- [test_tuning_consumers_e2e.py](test_tuning_consumers_e2e.py) — operator tool.
- [test_tuning_e2e.py](test_tuning_e2e.py) — operator tool.
- [test_tuning_install_e2e.py](test_tuning_install_e2e.py) — operator tool.
- [verify_setup.py](verify_setup.py) — operator tool.
- [watch_validate.py](watch_validate.py) — operator tool.
