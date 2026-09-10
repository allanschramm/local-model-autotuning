# `scripts/` — Utilities and Runner Scripts

## Purpose
Operator scripts for running setup health checks, monitoring GPU metrics, discovering candidate models, and managing the server daemon.

## Ownership
Repository operators and developers.

## Local Contracts
- Scripts must be runnable from the repository root.
- OpenVINO GenAI remains optional. `bench_openvino.py` imports it only while running and exits nonzero with an actionable install message when unavailable.
- `model_up.py` (global `model-up`) must work from any cwd: resolve `--model` and the Fingerprint `SPEC_DRAFT_MODEL` to absolute paths, and spawn `llama-server` with `cwd=REPO_ROOT`.
- `model_up.py` serves the Fingerprint file (issue #52, ADR 0014 phase 2): alias gives identity + local bind (`model`, `alias`, `host`, `port`, `llama_cpp_root`); engine flags come from `fingerprints/<stem>.json` via the same subset/order as the trial runner (incl. SPEC auto-MTP + `--n-cpu-moe` auto). Missing/invalid file fails closed (exit 1, never alias-flag soup); stale alias `flags:` are ignored with a NOTE. Single-load gate unchanged.
- `setup-check.sh` is the canonical readiness verification script (supports GPU acceleration and CPU-only builds). It must import-check every package listed in root `requirements.txt` (including `gguf`).

## Work Guidance
- Use `serve-config.py` as the preferred CLI helper to start/stop the llama-server daemon based on the mutable Baseline in `config.py`. It runs VRAM + host-memory preflight before spawn and exits 2 on reject.
- Use `build-llamacpp.py` (`python scripts/build-llamacpp.py --cpu` or `--cuda`) to build runtime binaries for local inference.
- Use `check_hardware.py` to diagnose hardware on **Windows, macOS, and Linux**: RAM, NVIDIA VRAM (`discrete_gpu`) or unified memory / no discrete NVIDIA (`unified_memory`, including Apple Silicon Metal). Report memory class + model pool; give conservative GGUF/context guidance. Also reports physical vs logical cores, best-effort SIMD hints (line omitted when undetectable — no crash), and CPU-only recommendations (`-ngl 0`, `-t` = physical cores, NUMA). Dense must fit the detected pool (VRAM or unified RAM with OS headroom); Metal uses `-ngl 99`, true CPU-only uses `-ngl 0`; never suggest partial dense offload. Agents treat this script as local fit authority over whichllm/llmfit rankings.
- Use `verify_setup.py` to validate local API server health and benchmark real-time TPS. Its default port matches `serve-config.py` (18080).
- Use `lcb_only.py` to re-measure LiveCodeBench (10 tasks) against current Baseline — gambiarra when coding HE/MBPP/BC already logged but LCB cache failed.
- Use `recompute_status.py` to refresh Trial statuses after the fact (issue #5; ADR 0017). Write paths (`run.py`, `autoloop.py`) already recompute after every Trial; the script covers retroactive refreshes. Default scope is per-model — same-basename competition only, no model ever demotes another; `--scope bucket` is retained as an accepted argument form with the same competition. Operates on the store (canonical `results.db` first, legacy `results.tsv` fallback) and writes refreshed statuses to both. Idempotent, no GPU. `--relabel-watchdog` runs the one-shot issue #72 migration first (legacy NVML device-wide policy kills `MODEL_REJECTED`/`VRAM_LIMIT_EXCEEDED` → `WATCHDOG_KILL`; `run.relabel_watchdog_kills`, idempotent).
- `rank_results.py` to print Day / Night / claw / coding / agentic-coding rankings from the results store — canonical `results.db` (SQLite) first, legacy `results.tsv` fallback. ADR 0017: the rank is a leaderboard of every complete model (quality-first, ±0.05 near-tie band; quality dominance enforced; Day tie-break TPS, Night tie-break ctx); each row = the model's single best Trial; the `pareto_front` alias passes complete points through. Agents must use this CLI for model rankings — no ad-hoc temp filter scripts.
- Use `model_info.py` to query GGUF ground truth read-only: arch (MoE/dense), block count, KV sizing, `--n-cpu-moe` resolution, and (with `--tensors`) the full tensor inventory. Agents must use this for model details / GGUF forensics — never parse the raw `.gguf` or dump tensors with ad-hoc scripts.
- Use `rebuild_results_db.py` for store maintenance: default seeds the canonical `results.db` from the legacy TSV when missing/unseeded and parity-checks both (`trial_id` set + duplicate detection); `--force` rebuilds the DB from the TSV; `--rebuild-tsv` rewrites the legacy TSV from the canonical DB. Exit 0 on parity, 1 on drift. Both stores stay in sync automatically after every Trial write; this script covers manual repairs.

## Verification
- Test script changes locally by executing them.
- `rank_results.py`: `.\venv\Scripts\python.exe -m pytest tests/test_rank_results.py`
- `check_leaderboard_docs.py`: pytest `tests/test_leaderboard_doc_check.py` + run the script against a seeded store for doc/table parity.
- Ensure `bash scripts/setup-check.sh` passes before declaring environment readiness.
- Shared validate: `python scripts/run_validate.py` (same as `.github/workflows/validate.yml`).
- After every push: `python scripts/watch_validate.py` waits for GitHub Actions `validate.yml` on HEAD (`gh run watch`). Local Windows pytest does not cover POSIX `fcntl` branches.
- Git pre-commit (Ruff/pytest) is repo-root owned — see root `AGENTS.md` Verification + `.pre-commit-config.yaml`.

## Child DOX Index
- [bench_openvino.py](bench_openvino.py) — optional OpenVINO GenAI CPU/iGPU benchmark with separate prefill/decode TPS output.
- [run_pytest_hook.py](run_pytest_hook.py) — pre-commit local entry for venv pytest.
- [run_validate.py](run_validate.py) — shared ruff+pytest validate (CI + local agents).
- [watch_validate.py](watch_validate.py) — wait for GitHub Actions `validate.yml` on HEAD after push.
- [lcb_only.py](lcb_only.py) — LCB-only remeasure helper (`scripts/lcb_only.py`).
- [recompute_status.py](recompute_status.py) — store-wide status recompute over the results store (issue #5; ADR 0017; `autoresearch/core/recompute.py` owns the logic).
- [rank_results.py](rank_results.py) — Day / Night / claw / coding / agentic-coding leaderboard over the results store, `results.db` first with legacy TSV fallback (ADR 0012/0017; ADR 0009/0013 floors demoted to history).
- [check_leaderboard_docs.py](check_leaderboard_docs.py) — leaderboard doc integrity (issue #69): no duplicate model names within a table section, Day/Night membership identical, doc tables byte-equal to `rank_results.py --mode pareto` output (parity leg skipped when store unseeded).
- [model_info.py](model_info.py) — read-only GGUF metadata query (arch, block count, KV sizing, `--tensors` inventory) via `autoresearch.core.model_arch`.
- [rebuild_results_db.py](rebuild_results_db.py) — store maintenance: seed/repair canonical `results.db`, rewrite legacy TSV (`--rebuild-tsv`), parity check (`autoresearch/core/results_db.py` owns the logic).
- [backfill_2026_08_08_missing_claw.py](backfill_2026_08_08_missing_claw.py) — one-shot restore of session-documented claw/coding rows that never landed in TSV (idempotent).
- [apply_fingerprint.py](apply_fingerprint.py) — copy a Fingerprint file into the mutable Baseline (issue #50, ADR 0014 phase 1): engine always, sampler only when present; benches keep reading Baseline.
- [measure_vram_peak.py](measure_vram_peak.py) — real peak-VRAM measurement for the Baseline model, gate bypass (estimator calibration, issue #10).
- Git pre-commit: [../.pre-commit-config.yaml](../.pre-commit-config.yaml) + [../pyproject.toml](../pyproject.toml).
