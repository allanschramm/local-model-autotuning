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

*   **Dense = physical VRAM only**: Never partially offload dense GGUFs (layers → CPU / Windows shared GPU memory). That path freezes the PC. Dense must fit physical VRAM (cut `CTX_SIZE` / KV quant / drop draft) or the Trial is rejected. Only MoE may use `--n-cpu-moe` / VITRIOL.
*   **Pre-flight Estimation**: `estimate_vram_mb` (weights + optional draft file + KV + overhead) runs before `llama-cli` / `llama-server`. Skip/reject any config with estimate `> VRAM_LIMIT_MB` (default 7900 on 8GB). Override via `ENGINE_DEFAULTS['VRAM_LIMIT_MB']` or `AUTORESEARCH_VRAM_LIMIT_MB`.
*   **Runtime NVML kill (dense)**: While `llama-server` runs, the VRAM sampler kills the process if `used_mb > VRAM_LIMIT_MB` and records `VRAM_LIMIT_EXCEEDED` — belt-and-suspenders when the estimate undercounts.
*   **TPS Floor**: User-set in Baseline `ENGINE_DEFAULTS['TPS_FLOOR']` (default **20.0**). Below this, `val_score` is zeroed / Trial rejects. MoE on 8GB often needs **15–18** — lower the floor per model; do not hardcode in harness.
*   **Shared Memory Mitigation**: Driver fallback to shared system memory freezes dense workloads. Do not rely on the TPS floor alone — preflight + NVML kill are the primary guards.
*   **Loop Resilience**: All model server startup failures, bad configurations, or exceptions are caught at the Trial level. They log a `FAIL` status to `results.tsv` and proceed to the next candidate configuration instead of crashing the search loop.
*   **NVML Failsafe**: If NVML query fails mid-run, set `nvml = None` in the exception block immediately to avoid repetitive CDLL calling overhead.
*   **Testing CDLL**: When writing unit tests for VRAM sampling, always mock `ctypes.CDLL` to raise an exception. This forces fallback to the mocked `nvidia-smi` parser and avoids testing against host GPU status.

## 3. llama-server binary resolution

*   **Resolution order** (checked by `llama_runner.py`): `AUTORESEARCH_LLAMA_CPP_ROOT`, repo-local `./llama.cpp`, parent/sibling `llama.cpp`, then `PATH`. The resolver supports POSIX `llama-server` / `llama-bench` and Windows `llama-server.exe` / `llama-bench.exe`, including CMake `bin/Release/` and `bin/Debug/` layouts.
*   **Upstream `ggml-org/llama.cpp` and forks are not interchangeable** for advanced features. TurboQuant, MTP, QAT, and diffusion support require specific forks. If a flag (`--spec-type`, `--cache-type-k`, `--n-cpu-moe`) is silently rejected, the build lacks that feature — try a different fork.
*   **Default install path is `./llama.cpp/` in the repo root.** Forks or custom builds must be cloned with the literal name `llama.cpp` to be auto-discovered, OR exported via `AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/llama.cpp`. On Windows, the same env var can point to a native Windows checkout/build root.
*   **Directory model paths use SGLang**: when `MODEL` resolves to a directory under `models/`, the harness uses `autoresearch/core/sglang_runner.py` and `venv-sglang/`. Do not launch SGLang directly for evaluation.
*   **`scripts/setup-check.sh` validates** that the build supports the expected flags (probes `--help`). Run it before the autoloop.

## 4. Loop Agent Constraints

*   **Mutable Baseline**: `autoresearch/core/config.py` (`ENGINE_DEFAULTS` = performance, `SAMPLER_DEFAULTS` = quality). The Search loop writes keeps here via `write_baseline`.
*   **Visited memory**: Ignored `.autoresearch_state.json` tracks visited Neighbors only — never Baseline.
*   **Fixed protocol**: `program.md` and `autoresearch/benchmarks/*` stay fixed unless the user explicitly requests a change.
*   **No Code Edits**: The looping agent is strictly forbidden from editing codebase source code (e.g., `run.py`, benchmarks, tests) under any circumstances. If any error, bug, or exception occurs during the Search, the agent MUST NOT attempt to edit code to fix it. Instead, the agent MUST immediately stop execution, print the full traceback/error, and warn the user.
*   **Unified Evaluation**: Every round runs the active agentic gate (Claw-Eval full Val Score; quick as smoke). Optional Coding preflight uses exactly 10 tasks per dataset when enabled.
*   **Canonical Results File**: All runs must log results exclusively to the single canonical tab-separated file `results.tsv`. No other results CSV, TSV, or log files should be committed or left in the workspace.
*   **Offline Results**: Benchmark results and search tweak branches must be kept offline and local-only. Never push result/tweak branches or local benchmark scores to the remote public repository to avoid polluting the public history or messing up other users' results.
*   **Hardware-Aware Path Resolution**: Path constants in `config.py` (e.g., `MODEL`) must use portable references — never absolute system paths (`/home/user/...`). Use `models/` (relative) or environment variables.

## 5. Validation Protocol

Every Trial runs a **2-step validation** before the full eval:

1. **llama-bench speed check** (`prompt=512`, `gen=128`, 3 repeats). If `tg_tps < TPS Floor` (Baseline `TPS_FLOOR`, default 20.0), Trial FAILs immediately — no server spin-up, no agentic eval.
2. **Claw-Eval quick smoke**. Reports local tool-use score under the config — not just fast garbage. **No score floor**: low smoke scores are recorded, not rejected. Only the TPS Floor rejects.

**Validation mode** (`python3 benchmark_search.py --validation`): runs steps 1-2 and exits. No extended eval, no keep/discard. For quick config sanity checks.

**Short-circuit**: Step 1 failure → logged as `FAIL` with bench TPS. The loop never wastes time on unusably slow configs. Smoke score never short-circuits.

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
   - Logs results to results.tsv

3. **What the --validation flag does** — Two steps, always both:
   - **Step 1 (speed check)**: `llama-bench` with `prompt=512`, `gen=128`, 3 repeats. If `tg_tps < TPS_FLOOR` (config.py; default 20.0), FAILs immediately — no agentic eval runs.
   - **Step 2 (agentic smoke)**: Claw-Eval quick scores local tool use with deterministic rule-based grading (no pass/fail cut on that score). Optional direct-coding preflight always uses exactly 10 tasks per dataset.

4. **One model at a time** — Never run multiple validations in parallel. All models share the same GPU (CUDA device 0) and default port 18080. Each validation must finish (PASS or FAIL) before the next starts.

5. **Result in results.tsv** — Written with category `validation` and status `discard` (validation runs never "keep"). Read the latest entry per model for comparison.

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
