# Golden Rules & Learnings for Auto-Tuning

## 1. Performance-Impacting Flags

*   **KV Cache Quantization**: Quantizing K/V caches (e.g., `q8_0`, `turbo3`, `q4_0`) reduces memory bandwidth requirements and VRAM footprint, yielding higher TPS at a minor retrieval cost.
*   **Flash Attention (`-fa`)**: Must always be `on` to utilize optimized GPU kernels. Disabling drops TPS by 3x+.
*   **Speculative Decoding (MTP)**:
    *   Upstream `llama.cpp` accepts `--spec-type draft-mtp`.
    *   Turboquant and similar forks typically accept `--spec-type mtp` (NOT `draft-mtp`). The autoloop's `llama_runner.py` probes `--help` at runtime and picks whichever value the build supports.
    *   Speculative draft tokens (`--spec-draft-n-max` between `1` and `4`) accelerate generation if accepted. Speedup is **1.15–1.25× for MoE**, **1.4–2.0× for dense**.
    *   **Architectural caveat**: speculative decoding with a separate draft model (e.g., Qwen-3.5-800M as drafter) does **not** help MoE+SSM models — verification becomes PCIe-bound. MTP (draft heads built into the model) is a different mechanism and does help.
*   **Offloading (`-ngl`)**: Default to maximum (`99` or `999`) for full GPU. Partial CPU offload is acceptable if throughput stays above 20 TPS — trading speed for accuracy is a valid strategy.
*   **Batching (`-b` / `-ub`)**: Micro-batch (`-ub`) and batch (`-b`) sizes balance GPU Tensor Core utilization during prefill against VRAM overhead.
*   **Threading (`-t`)**: CPU threads must match physical CPU core boundaries to avoid thrashing and context-switch latency.

## 2. VRAM Safety & Hardware Failsafes

*   **Pre-flight Estimation**: Check `estimate_vram_mb` before starting `llama-server`. Skip any configuration predicted to exceed hardware limit (e.g., 7.9 GB on a 8GB GPU) to prevent system hangs or hard OOMs. Partial CPU offload is preferred over skipping entirely.
*   **TPS Floor**: Hard floor is 20 TPS. Below this, `val_score` is zeroed.
*   **Shared Memory Mitigation**: Driver fallback to shared system memory drops processing speeds. The 20 TPS floor serves as the primary guardrail to automatically discard configurations relying on shared RAM.
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

*   **Mutable Search state**: Baseline + visited live in ignored `.autoresearch_state.json`. `config.py` holds immutable defaults only — the Search loop must never rewrite it.
*   **No Code Edits**: The looping agent is strictly forbidden from editing codebase source code (e.g., `run.py`, benchmarks, tests) under any circumstances. If any error, bug, or exception occurs during the Search, the agent MUST NOT attempt to edit code to fix it. Instead, the agent MUST immediately stop execution, print the full traceback/error, and warn the user.
*   **Unified Evaluation**: Every round runs the active agentic gate (Claw-Eval full Val Score; quick as smoke). Optional Coding preflight uses exactly 10 tasks per dataset when enabled.
*   **Canonical Results File**: All runs must log results exclusively to the single canonical tab-separated file `results.tsv`. No other results CSV, TSV, or log files should be committed or left in the workspace.
*   **Offline Results**: Benchmark results and search tweak branches must be kept offline and local-only. Never push result/tweak branches or local benchmark scores to the remote public repository to avoid polluting the public history or messing up other users' results.
*   **Hardware-Aware Path Resolution**: Path constants in `config.py` (e.g., `MODEL`) must use portable references — never absolute system paths (`/home/user/...`). Use `models/` (relative) or environment variables.

## 5. Validation Protocol

Every Trial runs a **2-step validation** before the full eval:

1. **llama-bench speed check** (`prompt=512`, `gen=128`, 3 repeats, ctx >= 100k). If `tg_tps < TPS Floor` (20.0), Trial FAILs immediately — no server spin-up, no agentic eval.
2. **Claw-Eval quick smoke**. Reports local tool-use score under the config — not just fast garbage. **No score floor**: low smoke scores are recorded, not rejected. Only the TPS Floor rejects.

**Validation mode** (`python3 benchmark_search.py --validation`): runs steps 1-2 and exits. No extended eval, no keep/discard. For quick config sanity checks.

**Short-circuit**: Step 1 failure → logged as `FAIL` with bench TPS. The loop never wastes time on unusably slow configs. Smoke score never short-circuits.

See `autoresearch/runners/evaluation.py` → `run_llama_bench_validation()` + `run_trial()` for implementation.

### How to Validate a Single Model (step-by-step)

When asked to "validate a model", follow this exact procedure:

1. **Set MODEL** — Put the target GGUF filename in `config.py` defaults (or the Baseline in `.autoresearch_state.json`). Leave other constants alone unless the task says otherwise.

2. **Run validation** — Execute directly, no wrapper scripts:
   ```
   python3 benchmark_search.py --validation --desc "validate <model-filename>"
   ```
   This runs the unified harness (not raw binaries). The harness:
   - Resolves the model path from Baseline / config defaults
   - Translates config flags to llama-server CLI args
   - Manages server lifecycle (start, health-check, teardown)
   - Monitors VRAM via NVML sampling
   - Logs results to results.tsv

3. **What the --validation flag does** — Two steps, always both:
   - **Step 1 (speed check)**: `llama-bench` with `prompt=512`, `gen=128`, 3 repeats. If `tg_tps < 20.0`, FAILs immediately — no agentic eval runs.
   - **Step 2 (agentic smoke)**: Claw-Eval quick scores local tool use with deterministic rule-based grading (no pass/fail cut on that score). Optional direct-coding preflight always uses exactly 10 tasks per dataset.

4. **One model at a time** — Never run multiple validations in parallel. All models share the same GPU (CUDA device 0) and default port 18080. Each validation must finish (PASS or FAIL) before the next starts.

5. **Result in results.tsv** — Written with category `validation` and status `discard` (validation runs never "keep"). Read the latest entry per model for comparison.

**RULES**:
- Do NOT run `llama-server` or `llama-bench` directly. The harness handles everything.
- Do NOT write wrapper scripts or bash loops. Change Baseline/state and invoke benchmark_search.py directly.
- Do NOT batch models into a single command chain. One validation per invocation.

## 6. Use the Harness, Not Raw Binaries

*   **Do NOT run `llama-server` or `llama-bench` directly** for evaluation. The harness (`benchmark_search.py`, `autoloop.py`) resolves paths, translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results. Bypassing it produces unlogged, unreproducible trials.
*   **Do NOT override flags via raw `llama-server` CLI arguments**. All tuning goes through Baseline state / config defaults. The harness generates the correct `llama-server` command.
*   **Mutable Search surface is `.autoresearch_state.json`**. `config.py` is immutable defaults. Run `python3 autoloop.py` or `python3 benchmark_search.py --desc "what you changed"`.

## 7. Codebase Architecture

*   **Simplicity First**: Never overengineer. Keep the architecture simple. Less is more.
*   **Minimalism**: Try to reduce lines of code, not increase. Simplify instead of complicate.
*   **Portable Documentation**: Docs and configs must use relative paths, env vars, or placeholders — never `/home/<user>/...` or `/mnt/<host>/...` in committed files.
