# Golden Rules & Learnings for Auto-Tuning

## 1. Performance-Impacting Flags

*   **KV Cache Quantization**: Quantizing K/V caches (e.g., `q8_0`, `turbo3`, `q4_0`) reduces memory bandwidth requirements and VRAM footprint, yielding higher TPS at a minor retrieval cost.
*   **Flash Attention (`-fa`)**: Must always be `on` to utilize optimized GPU kernels. Disabling drops TPS by 3x+.
*   **Speculative Decoding (MTP)**:
    *   Nexus-System `llama.cpp` uses `--spec-type draft-mtp`.
    *   Turboquant `llama.cpp` uses `--spec-type mtp`.
    *   Speculative draft tokens (`--spec-draft-n-max` between `1` and `4`) accelerate generation if accepted.
*   **Offloading (`-ngl`)**: Default to maximum (`99` or `999`) for full GPU. Partial CPU offload is acceptable if throughput stays above 20 TPS — trading speed for accuracy is a valid strategy.
*   **Batching (`-b` / `-ub`)**: Micro-batch (`-ub`) and batch (`-b`) sizes balance GPU Tensor Core utilization during prefill against VRAM overhead.
*   **Threading (`-t`)**: CPU threads must match physical CPU core boundaries to avoid thrashing and context-switch latency.

## 2. VRAM Safety & Hardware Failsafes

*   **Pre-flight Estimation**: Check `estimate_vram_mb` before starting `llama-server`. Skip any configuration predicted to exceed hardware limit (e.g., 7.9 GB on a 8GB GPU) to prevent system hangs or hard OOMs. Partial CPU offload is preferred over skipping entirely.
*   **TPS Floor**: Hard floor is 20 TPS. Below this, `val_score` is zeroed. Between 20-40 TPS, a soft penalty curve applies via `speed_factor`.
*   **NVML Failsafe**: If NVML query fails mid-run, set `nvml = None` in the exception block immediately to avoid repetitive CDLL calling overhead.
*   **Testing CDLL**: When writing unit tests for VRAM sampling, always mock `ctypes.CDLL` to raise an exception. This forces fallback to the mocked `nvidia-smi` parser and avoids testing against host GPU status.

## 3. Loop Agent Constraints

*   **Single Changeable Surface**: The looping agent must only modify constants in [autoresearch/core/config.py](file:///home/shark/workspace/autoresearch-public/config.py).
*   **Unified Evaluation**: Every round must execute all active benchmarks (Nexus Retrieval + Claw Agency + optionally Coding) rather than testing a single domain.
*   **Canonical Results File**: All runs must log results exclusively to the single canonical tab-separated file `results.tsv`. No other results CSV, TSV, or log files should be committed or left in the workspace.

## 4. Codebase Architecture

*   **Simplicity First**: Never overengineer. Keep the architecture simple. Less is more.
*   **Minimalism**: Try to reduce lines of code, not increase. Simplify instead of complicate.
