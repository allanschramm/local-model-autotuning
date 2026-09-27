# Golden Rules & Learnings for Auto-Tuning

## 1. Performance-Impacting Flags

*   **KV Cache Quantization**: Quantizing K/V caches (e.g., `q8_0`, `turbo3`, `q4_0`) reduces memory bandwidth requirements and VRAM footprint, yielding higher TPS at a minor retrieval cost.
*   **Flash Attention (`-fa`)**: Must always be `on` to utilize optimized GPU kernels. Disabling drops TPS by 3x+.
*   **Speculative Decoding (MTP)**:
    *   Upstream `llama.cpp` accepts `--spec-type draft-mtp`.
    *   Turboquant and similar forks typically accept `--spec-type mtp` (NOT `draft-mtp`). The autoloop's `llama_runner.py` probes `--help` at runtime and picks whichever value the build supports.
    *   Speculative draft tokens (`--spec-draft-n-max` between `1` and `4`) accelerate generation if accepted. Speedup is **1.15–1.25× for MoE**, **1.4–2.0× for dense**.
    *   **Architectural caveat**: speculative decoding with a separate draft model (e.g., Qwen-3.5-800M as drafter) does **not** help MoE+SSM models — verification becomes PCIe-bound. MTP (draft heads built into the model) is a different mechanism and does help.
*   **Offloading (`-ngl`)**: Default to maximum (`99` or `999`) for full GPU. Dense models must stay fully on GPU — never partial layer offload to shared memory. MoE may use `--n-cpu-moe` (VITRIOL) when experts do not fit. Baseline `N_CPU_MOE=None` auto-uses GGUF `block_count`; set `0` only when the MoE fits physical VRAM.
*   **Batching (`-b` / `-ub`)**: Micro-batch (`-ub`) and batch (`-b`) sizes balance GPU Tensor Core utilization during prefill against VRAM overhead.
*   **Threading (`-t`)**: CPU threads must match physical CPU core boundaries to avoid thrashing and context-switch latency.

## 2. VRAM Safety & Hardware Failsafes

*   **Detect hardware first**: Before recommending or downloading a GGUF, run `scripts/check_hardware.py` (Win/macOS/Linux). Read `memory_class`: `discrete_gpu` (NVIDIA VRAM) vs `unified_memory` (Apple Silicon / no discrete NVIDIA — one RAM pool shared with the OS). Explain and confirm with the user. Do not download blind if detection is incomplete — guide manual checks (`nvidia-smi`, About This Mac / `sysctl`, Task Manager).
*   **whichllm / llmfit ≠ fit authority**: Treat them as candidate lists. On unified memory they may over-rank large models. Discard picks that would leave too little OS/IDE headroom (e.g. ~12 GB GGUF on 16 GB unified) — do not treat total RAM as a fill target.
*   **Host-memory preflight (hard gate)**: Before `llama-server` / validation / autoloop Trial, `estimate_host_memory_mb` (full GGUF + draft + KV + overhead, **no** MoE shrink) must fit `RAM − headroom`. Unified headroom = `max(6144, 0.20×RAM)` MiB; discrete = `max(4096, 0.15×RAM)`. Override via `HOST_MEMORY_HEADROOM_MB` / `AUTORESEARCH_HOST_HEADROOM_MB`. Fail closed on unified if RAM unknown. Rejects as `HOST_MEMORY_PREFLIGHT` / `MODEL_REJECTED` — even if an agent ignores docs.
*   **Dense = physical pool only**: Never partially offload dense GGUFs (layers → CPU / Windows shared GPU memory). That path freezes the PC. Dense must fit **physical VRAM** (discrete) or the **unified RAM pool with OS headroom** (Mac/UMA). Cut `CTX_SIZE` / KV quant / drop draft or reject. Only MoE may use `--n-cpu-moe` / VITRIOL.
*   **Pre-flight Estimation**: `estimate_vram_mb` (weights + optional draft file + KV + overhead) runs before `llama-cli` / `llama-server`. Skip/reject any config with estimate `> VRAM_LIMIT_MB` (default 7900 on 8GB). Override via `ENGINE_DEFAULTS['VRAM_LIMIT_MB']` or `AUTORESEARCH_VRAM_LIMIT_MB`.
*   **Physical keepout**: `resolve_vram_limit_mb` clamps to ``physical − PHYSICAL_VRAM_KEEPOUT_MB`` (default 512). Kill dedicated overshoot (dense **and** MoE).
*   **Shared GPU kill (absolute)**: Kill if process Task Manager Shared GPU `> SHARED_VRAM_LIMIT_MB` (default 2048) even when dedicated is only ~4–5 GB — MoE+`NO_MMAP` WDDM/PCI-e host maps → pagefile freeze. Also `GGML_CUDA_NO_PINNED=1` on cli/server.
*   **TPS llama-cli guard**: Caps `-c` at `BENCH_CTX_CAP` (4096); watches dedicated + Shared.
*   **TPS Floor**: User-set in Baseline `ENGINE_DEFAULTS['TPS_FLOOR']` (default **20.0**). Below this, `val_score` is zeroed / Trial rejects. MoE on 8GB often needs **15–18** — lower the floor per model; do not hardcode in harness.
*   **Shared Memory Mitigation**: Shared GPU bucket (not “normal RAM”) freezes via pagefile/SSD. Dedicated-only keepout is not enough — Shared absolute kill + `GGML_CUDA_NO_PINNED=1` are the MoE guards.
*   **Loop Resilience**: All model server startup failures, bad configurations, or exceptions are caught at the Trial level. They log a `FAIL` status to `results.db` (with the `results.tsv` legacy mirror) and proceed to the next candidate configuration instead of crashing the search loop.
*   **NVML Failsafe**: If NVML query fails mid-run, set `nvml = None` in the exception block immediately to avoid repetitive CDLL calling overhead.
*   **Testing CDLL**: When writing unit tests for VRAM sampling, always mock `ctypes.CDLL` to raise an exception. This forces fallback to the mocked `nvidia-smi` parser and avoids testing against host GPU status.

## 3. llama-server binary resolution

*   **Resolution order** (checked by `llama_runner.py`): `AUTORESEARCH_LLAMA_CPP_ROOT`, repo-local `./llama.cpp`, parent/sibling `llama.cpp`, then `PATH`. The resolver supports POSIX `llama-server` / `llama-bench` and Windows `llama-server.exe` / `llama-bench.exe`, including CMake `bin/Release/` and `bin/Debug/` layouts.
*   **Upstream `ggml-org/llama.cpp` and forks are not interchangeable** for advanced features. TurboQuant, MTP, QAT, and diffusion support require specific forks. If a flag (`--spec-type`, `--cache-type-k`, `--n-cpu-moe`) is silently rejected, the build lacks that feature — try a different fork.
*   **Default install path is `./llama.cpp/` in the repo root.** Forks or custom builds must be cloned with the literal name `llama.cpp` to be auto-discovered, OR exported via `AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/llama.cpp`. On Windows, the same env var can point to a native Windows checkout/build root.
*   **Directory model paths use SGLang**: when `MODEL` resolves to a directory under `models/`, the harness uses `autoresearch/core/sglang_runner.py` and `venv-sglang/`. Do not launch SGLang directly for evaluation.
*   **`models/` store — real directory:** never run recursive/forced deletes on the `models/` root (`Remove-Item -Recurse`, `rm -rf`, `rmdir /s`). Per-file deletes inside `models/<publisher>/` are normal maintenance.
*   **`scripts/setup-check.sh` validates** that the build supports the expected flags (probes `--help`). Run it before the autoloop.

## 4. Loop Agent Constraints

*   **Mutable Baseline**: `autoresearch/core/config.py` (`ENGINE_DEFAULTS` = performance, `SAMPLER_DEFAULTS` = quality). The Search loop writes the Baseline here via `write_baseline`.
*   **Visited memory**: Ignored `.autoresearch_state.json` tracks visited Neighbors only — never Baseline.
*   **Fixed protocol**: `program.md` and `autoresearch/benchmarks/*` stay fixed unless the user explicitly requests a change.
*   **No Code Edits**: The looping agent is strictly forbidden from editing codebase source code (e.g., `run.py`, benchmarks, tests) under any circumstances. If any error, bug, or exception occurs during the Search, the agent MUST NOT attempt to edit code to fix it. Instead, the agent MUST immediately stop execution, print the full traceback/error, and warn the user.
*   **Unified Evaluation**: Every round runs the active agentic gate (Claw-Eval full Val Score; quick as smoke). Optional Coding preflight uses exactly 10 tasks per dataset when enabled.
*   **Canonical Results Store**: All runs log to the single canonical SQLite database `results.db` (typed columns, indexed, gitignored). An append-only mirror `results.tsv` is kept as a legacy export. Production readers must use `autoresearch.core.results_db.load_rows()` (SQLite-first, TSV fallback). No ad-hoc results CSV/TSV/log files should be committed.
*   **Offline Results**: Benchmark results and search tweak branches must be kept offline and local-only. Never push result/tweak branches or local benchmark scores to the remote public repository to avoid polluting the public history or messing up other users' results.
*   **Hardware-Aware Path Resolution**: Path constants in `config.py` (e.g., `MODEL`) must use portable references — never absolute system paths (`/home/user/...`). Use `models/` (relative) or environment variables.

## 5. Validation Protocol

Every Trial runs hard gates that prove the rig can load the model, then smoke validation:

0. **Arch + VRAM + host (before any bench):** classify GGUF dense vs MoE, resolve `--n-cpu-moe` (`None` → auto `block_count` for MoE), VRAM preflight, then host-memory preflight (full GGUF, no MoE shrink). MoE `N_CPU_MOE=0` over physical VRAM → `MODEL_REJECTED` (set `None` for auto offload). Host over budget → `HOST_MEMORY_PREFLIGHT` / `MODEL_REJECTED`.
1. **llama-bench speed check** (`prompt=512`, `gen=128`, 3 repeats). If `tg_tps < TPS Floor` (Baseline `TPS_FLOOR`, default 20.0), Trial FAILs immediately — no server spin-up, no agentic eval.
2. **Claw-Eval quick smoke**. Reports local tool-use score under the config — not just fast garbage. **No score floor**: low smoke scores are recorded, not rejected. Only the TPS Floor rejects.

**Validation mode** (`python3 benchmark_search.py --validation`): runs gates 0–2 and exits. No extended eval, no keep/discard. For quick config sanity checks.

**Default speed path** (good enough, fast): smoke `--validation` → `autoloop.py --mode tps` → champion Claw full only after TPS is acceptable. See `docs/discovery/good-enough-tuning.md`. Do not use Claw full / coding-10 inside the speed search loop.

**Short-circuit**: Gate 0 or step 1 failure → logged as `FAIL`. The loop never wastes time on unloadable or unusably slow configs. Smoke score never short-circuits.

See `autoresearch/runners/evaluation.py` → `run_llama_bench_validation()` + `run_trial()` for implementation.

### How to Validate a Single Model (step-by-step)

When asked to "validate a model", follow this exact procedure:

1. **Set MODEL** — Put the target GGUF filename in `autoresearch/core/config.py` Baseline. Leave other constants alone unless the task says otherwise.

2. **Run validation** — Execute directly, no wrapper scripts:
   ```
   python3 benchmark_search.py --validation --desc "validate <model-filename>"
   ```
   This runs the unified harness (not raw binaries). The harness:
   - Resolves the model path from `config.py` Baseline
   - Translates config flags to llama-server CLI args
   - Manages server lifecycle (start, health-check, teardown)
   - Monitors VRAM via NVML sampling
   - Logs results to `results.db` (canonical) with `results.tsv` as a legacy export mirror

3. **What the --validation flag does** — Gates 0–2, always:
   - **Gate 0 (arch + VRAM):** dense vs MoE from GGUF, resolve `N_CPU_MOE`, VRAM preflight (MoE full-GPU over limit → reject).
   - **Step 1 (speed check):** `llama-bench` with `prompt=512`, `gen=128`, 3 repeats. If `tg_tps < TPS_FLOOR` (config.py; default 20.0), FAILs immediately — no agentic eval runs.
   - **Step 2 (agentic smoke):** Claw-Eval quick scores local tool use with deterministic rule-based grading (no pass/fail cut on that score). Validation never runs coding-10 — coding is canonical-Trial work (complete Objective Vector); `--validation` is smoke-only (issue #9 user rule; `INCLUDE_CODING=True` default does not leak into validation).

4. **One model at a time** — Never run multiple validations in parallel. All models share the same GPU (CUDA device 0) and default port 18080. Each validation must finish (PASS or FAIL) before the next starts.

5. **Result in `results.db`** — Written with category `validation` and Trial Status `incomplete` (smoke-only; missing agentic/coding axes never join the front) — or `rejected` when a hard gate fails. Read the latest entry per model for comparison via `rank_results.py` (SQLite-first).

**RULES**:
- Do NOT run `llama-server` or `llama-bench` directly. The harness handles everything.
- Do NOT write wrapper scripts or bash loops. Change `config.py` Baseline and invoke benchmark_search.py / `run.py` directly.
- Do NOT batch models into a single command chain. One validation per invocation.

## 6. Use the Harness, Not Raw Binaries

*   **Do NOT run `llama-server` or `llama-bench` directly** for evaluation. The harness (`benchmark_search.py`, `autoloop.py`) resolves paths, translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results. Bypassing it produces unlogged, unreproducible trials.
*   **Do NOT override flags via raw `llama-server` CLI arguments**. All tuning goes through `config.py` Baseline. The harness generates the correct `llama-server` command.
*   **Mutable Search surface is `autoresearch/core/config.py`**. Visited memory is `.autoresearch_state.json`. Run `python3 autoloop.py` or `python3 benchmark_search.py --desc "what you changed"`.

## 7. Codebase Architecture

*   **Simplicity First**: Never overengineer. Keep the architecture simple. Less is more.
*   **Minimalism**: Try to reduce lines of code, not increase. Simplify instead of complicate.
*   **Portable Documentation**: Docs and configs must use relative paths, env vars, or placeholders — never `/home/<user>/...` or `/mnt/<host>/...` in committed files.
