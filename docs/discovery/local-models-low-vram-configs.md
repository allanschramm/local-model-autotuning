# Low VRAM Optimization Guide: Local & LM Studio Models

This document provides the optimal `llama.cpp` configuration parameters to run all target models in our local store (`models/local/` and `models/lmstudio-community/`) on consumer hardware with **8 GB of VRAM** (specifically optimized for an RTX 4060 rig with 16–24 GB of System RAM).

---

## 1. Optimal Parameter Decision Matrix

For a GPU with 8 GB VRAM, models are categorized into three operational classes:

| Class | Model Examples | GPU Layers (`-ngl`) | expert split (`--n-cpu-moe`) | KV Cache Quant (`-ctk`/`-ctv`) | Speculative Decoding (`--spec-type`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A: Small Dense / small MoE (<10B)** | Qwen3.5-9B, Qwythos-9B, LFM2.5-1.2B, **LFM2.5-8B-A1B** | `99` (Full) | N/A dense; **`0` for LFM 8B** (fits) | `q4_0` (1.2B: `f16`) | `draft-mtp` if embedded, else `none` |
| **B: Ternary / Packed (27B)** | Bonsai-27B Q1_0 | `99` (Full) | N/A | `q4_0` | `none` (prevents speed inversion) |
| **C: Mixture-of-Experts** | Qwen3.6-35B, Ornith-1.0-35B, **Laguna-XS-2.1** | `99` (Hybrid) | **`block_count`** (40 on these GGUFs) | `q4_0` | `none` (prevents CPU sync bottleneck) |

---

## 2. Detailed Configurations per Model

### Group A: Small Dense Models (<10B) — *Fits 100% on GPU*

These models run entirely in VRAM. The primary low-VRAM concern is preventing context cache growth from overflowing the 8 GB boundary (which causes driver-level paging and drops speed to ~2 t/s).

#### 1. Qwen3.5-9B-UD-Q4_K_XL.gguf
*   **Path:** [Qwen3.5-9B-UD-Q4_K_XL.gguf](file:///D:/Dev/Nexus-System/local-model-autotuning/models/lmstudio-community/qwen3.5-9b-gguf/Qwen3.5-9B-UD-Q4_K_XL.gguf)
*   **VRAM Strategy:** Fits comfortably on GPU. Uses embedded MTP tensors for hardware-level acceleration.
*   **Optimal `llama-server` / `llama-cli` Flags:**
    ```bash
    -ngl 99 -c 131072 -fa on -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 4
    ```
*   **Why this works:**
    *   `-ctk q4_0 -ctv q4_0` compresses KV Cache keys and values to 4-bit, saving ~75% of VRAM context overhead, allowing context to scale up to 131k without paging.
    *   `--spec-type draft-mtp --spec-draft-n-max 4` enables speculative decoding using the model's native embedded prediction heads. **No external draft model file is needed.**
    *   **Performance:** Boosts generation speed from **38.7 t/s** (base) to **57.3 t/s** (+48% gain).

#### 2. Qwythos-9B-Claude-Mythos-5-1M (Base / MTP / v2)
*   **Path:** [Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf](file:///D:/Dev/Nexus-System/local-model-autotuning/models/local/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf) / [v2-Q4_K_M.gguf](file:///D:/Dev/Nexus-System/local-model-autotuning/models/local/Qwythos-9B-v2-Q4_K_M/Qwythos-9B-v2-Q4_K_M.gguf)
*   **VRAM Strategy:** Fits on GPU. Context is designed for 100k+; KV Cache compression is mandatory.
*   **Optimal Flags:**
    ```bash
    -ngl 99 -c 131072 -fa on -ctk q4_0 -ctv q4_0 -b 1024 -ub 256 --spec-type none
    ```
*   **Why this works:**
    *   `-b 1024 -ub 256` represents the evaluated batch size sweet spot on RTX 4060, balancing prefill latency and text generation throughput.
    *   `--spec-type none` is selected because the Mythos MTP draft heads yield negligible performance gain (~+1% in short generation tests) while adding memory overhead that risks driver paging at extreme contexts.
    *   **Performance:** Delivers **51.2 t/s** with 7.5 GB peak VRAM usage under full 131k context.

#### 3. LFM2.5-1.2B-Instruct-Q8_0.gguf
*   **Path:** `models/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf`
*   **VRAM Strategy:** Extremely small footprint (~1.16 GB). Alias `lfm2.5-1.2b`. GGUF max ctx 128k.
*   **Optimal Flags (alias — best TPS):**
    ```bash
    -ngl 99 -c 65536 -fa on -ctk f16 -ctv f16 --no-mmap
    ```
*   **Measured (2026-07-24 claw-quick matrix):**
    *   **Preferred:** ctx **65k** KV **f16** — **0.80**, **180.6 t/s**, peak 3.3 GB
    *   128k f16 — preflight FAIL (est 11.5 GB)
    *   128k q4_0 — 0.60, 178.6 t/s, peak 3.4 GB
    *   128k q8_0 — 0.60, 177.6 t/s, peak 3.7 GB

#### 4. LFM2.5-8B-A1B-Q4_K_M.gguf
*   **Path:** `models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf` (~5.16 GB)
*   **VRAM Strategy:** Hybrid MoE (`lfm2moe`, 32 experts / 4 active). **Fits full GPU** — use `--n-cpu-moe 0`. Alias `lfm2.5-8b-a1b`.
*   **Optimal Flags:**
    ```bash
    -ngl 99 -c 65536 -fa on -ctk q4_0 -ctv q4_0 --n-cpu-moe 0 --jinja --cont-batching
    # sampler (Liquid): --temp 0.2 --top-k 80 --repeat-penalty 1.05
    ```
*   **Why this works:** Active ~1.5B → ~174 t/s. Harness `N_CPU_MOE=None` auto-dumps experts to CPU (peak ~2.3 GB) and slows agentic; `0` keeps experts on GPU (peak **6.5 GB**, agentic ~3× faster). Bench tg unchanged (~174) because it already ran full-GPU.
*   **Measured (2026-07-23):** KEEP; claw-quick **0.20** (tool-call format suspect). Card: [lfm2.5-8b-a1b.md](../models/lfm2.5-8b-a1b.md).

---

### Group B: Ternary & Packed Models (27B) — *Fits 100% on GPU*

Ternary (1-bit / 2-bit) quantizations compress 27B parameter models down to under 7 GB, enabling them to run entirely on a consumer GPU.

#### 1. Bonsai-27B-Q1_0.gguf
*   **Path:** `models/local/Bonsai-27B-Q1_0/Bonsai-27B-Q1_0.gguf`
*   **VRAM Strategy:** Ultra-low weight size (3.53 GiB) due to 1-bit quantization. Run with upstream `llama.cpp` (stock CUDA is faster than specialized forks for this format).
*   **Optimal Flags (TPS / short-gen):**
    ```bash
    -ngl 99 -c 131072 -fa on -ctk q4_0 -ctv q4_0 --spec-type none --no-mmap
    ```
*   **Agentic / claw-full Flags (required on 8 GB):**
    ```bash
    -ngl 99 -c 65536 -fa on -ctk q4_0 -ctv q4_0 --spec-type none --no-mmap
    ```
*   **Why this works:**
    *   `--spec-type none` is critical. Enabling DSpark speculative decoding triggers **Quantization-Speed Inversion** (−48% TPS).
    *   `--no-mmap` avoids memory paging bottlenecks during model load.
    *   **131k agentic VRAM-kill (2026-07-24):** mid-full `used=7906MB > limit=7900MB` — alias `bonsai` locked to **65k** for Claw-Eval. Short-gen @ 131k still ~41 t/s.
    *   **Measured claw-full @ 65k:** Val Score **0.4667**, bench_tg **40.2**, peak **6.5 GB**. Card: [bonsai-27b.md](../models/bonsai-27b.md).

#### 2. Ternary-Bonsai-27B-Q2_0.gguf — **deleted locally (2026-07-23)**
*   **Status:** Rejected. PrismML CUDA loads the quant, but **bench_tg ≈ 10.6 t/s** @ ctx 32k on RTX 4060 8 GB — below `TPS_FLOOR` 15. Prefer [Bonsai-27B-Q1_0](#1-bonsai-27b-q1_0gguf) (~41 t/s). Full note: [ternary-bonsai-27b.md](../models/ternary-bonsai-27b.md).
*   **If re-acquiring:** needs `llama.cpp-prismml`; start with `-c 32768` (65k fails VRAM preflight ~8.5 GB est).

---

### Group C: Mixture-of-Experts (35B) — *Requires Hybrid Offloading*

MoE models exceed the 8 GB physical VRAM limit. However, because they only activate a subset of experts per token (~3B parameters), we can offload the inactive experts to CPU/System RAM.

#### 1. Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
*   **Path:** [Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf](file:///D:/Dev/Nexus-System/local-model-autotuning/models/lmstudio-community/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf)
*   **VRAM Strategy:** Offloads 100% of routed experts to CPU, placing only the attention mechanism, shared experts, and routing layers on GPU (**VITRIOL technique**).
*   **Optimal Flags:**
    ```bash
    -ngl 99 --n-cpu-moe 40 -c 65536 -fa on -ctk q4_0 -ctv q4_0 --spec-type none
    ```
*   **Why this works:**
    *   `--n-cpu-moe 40` keeps MoE experts for all 40 layers in CPU System RAM. The RTX 4060 only computes the active attention/routing paths.
    *   `--spec-type none` is **mandatory**. Speculative decoding forces sequential CPU-to-GPU memory synchronization on every proposed draft token, collapsing MoE offloading throughput by **60%** (dropping from ~20 t/s to 8 t/s).
    *   **Performance:** Expected throughput of **~22–25 t/s** on modern host CPUs (e.g. 12th+ Gen Intel/AMD) with DDR5 RAM.

#### 2. Ornith-1.0-35B-UD-Q4_K_XL.gguf
*   **Path:** `models/local/Ornith-1.0-35B-UD-Q4_K_XL/Ornith-1.0-35B-UD-Q4_K_XL.gguf` (or nested publisher layout)
*   **VRAM Strategy:** MoE VITRIOL — offload experts for all layers (`block_count=40`).
*   **Optimal Flags:**
    ```bash
    -ngl 99 --n-cpu-moe 40 -c 65536 -fa on -ctk q4_0 -ctv q4_0 --spec-type none -b 1024 -ub 256
    ```
*   **Why this works:**
    *   A/B (2026-07-24): `n-cpu-moe=40` (= `block_count`) beat prior **32** — **28.9 vs 27.0 t/s**, peak VRAM **4.3 vs 7.8 GB**, claw-quick 0.80 both.
    *   Prefer **65k** for agentic (131k preflight / headroom risk on 8 GB).
    *   **Measured claw-full:** Val Score **0.6000**, bench_tg **25.7**, peak **4.9 GB**. Alias `ornith-35b`. Card: [ornith-1.0-35b.md](../models/ornith-1.0-35b.md).

#### 3. Laguna-XS-2.1-Q3_K_XL.gguf
*   **Path:** `models/bartowski/laguna-xs-2.1-gguf/Laguna-XS-2.1-Q3_K_XL.gguf` (~16.3 GB)
*   **VRAM Strategy:** MoE (`laguna`, 256 experts / 8 active). Full expert offload `n-cpu-moe 40`.
*   **Optimal Flags:**
    ```bash
    -ngl 99 --n-cpu-moe 40 -c 65536 -fa on -ctk q4_0 -ctv q4_0 --jinja --cont-batching --no-mmap
    ```
*   **Why this works:** Attention/routing on GPU; experts on CPU. Lowest VRAM among top Val Score models.
*   **Measured (2026-07-24):** claw-quick **1.00**; claw-full **0.6667** (best Val Score on this rig); bench_tg **37.2**; peak **3.6 GB**. Alias `laguna-xs`. Card: [laguna-xs-2.1.md](../models/laguna-xs-2.1.md). Leaderboard: [claw-eval-leaderboard.md](claw-eval-leaderboard.md).

---

## 3. Advanced Diagnostics & System-Level Tuning

For extreme VRAM constraints, system-level configurations complement model-specific command parameters:

### A. Memory Allocator Optimization (`tcmalloc` / `jemalloc`)
Standard host operating system allocators introduce multi-threaded lock contention when thread execution scale increases under deep context. Replacing the default allocation library with a lock-free thread-local cache structure reduces CPU latency overhead.
*   **When to use:** When running MoE models with high CPU experts offload (`--n-cpu-moe` active) or processing massive prompt contexts.
*   **Linux Execution:** Preload the library before launching `llama-server`:
    ```bash
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./llama-server ...
    # Or for tcmalloc:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./llama-server ...
    ```

### B. PrismML Ternary Q2_0 Format Details
Ternary (1.58-bit) representation constrains model weights to $\{-s, 0, +s\}$, saving significant VRAM memory footprint.
*   **Layout mechanics:** Quantized group scaling maps blocks of `128` weights to share a single FP16 scale factor. 2-bit codes represent ternary configurations (~1.58 bpw effective size).
*   **Execution constraint:** Upstream standard `llama.cpp` does not support group-wise `g128` ternary layouts. The external [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) custom build is required to run the `Q2_0` binary formats on CUDA; it is not a repository submodule.

### C. Unified Memory System Fallback
*   **Environment Variable:** On Linux setups running NVIDIA GPUs, set:
    ```bash
    export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
    ```
    This enables CUDA Unified Memory allocation. If allocation requests overflow physical VRAM, memory slides to system RAM without raising an out-of-memory crash.
*   **Windows equivalent:** Windows drivers enable unified memory fallback automatically. If memory paging occurs, generation speed drops to ~2 t/s. Monitor VRAM headroom using `nvidia-smi` to ensure active VRAM usage does not exceed **7.5 GB** of physical capacity.
