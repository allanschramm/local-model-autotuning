# Advanced Inference Optimizations Guide

This guide documents advanced, high-performance optimization techniques for local LLM inference engines (such as `llama.cpp` and `vLLM`), drawing lessons from the official **Fast Gemma Challenge** (Google DeepMind & Hugging Face, 2026) and our own empirical benchmarks.

---

## 1. CUDA Graph Capture (Static Execution Graph)

For small models (e.g. Gemma-4 E4B, Qwen3.5-9B), GPU kernels execute so quickly that the **CPU overhead of launching kernels** (CUDA API call latency) becomes the primary performance bottleneck.

*   **Technique:** CUDA Graphs allow the engine to record the execution sequence of GPU kernels once during initialization and replay the entire sequence with a single launch command.
*   **Performance Impact:** Eliminates CPU-to-GPU launch latency. Can boost TPS by **15% to 40%** on small models.
*   **In `llama.cpp`:** there is **no `--cuda-graph` CLI flag** — CUDA graphs are compiled into the CUDA backend (`USE_CUDA_GRAPH`) and **enabled automatically** at runtime. Each graph is checked for compatibility (graph capture is skipped when the node mix needs it); disable via the `GGML_CUDA_DISABLE_GRAPHS` environment variable, not a flag. b10488 relaxed one over-conservative disable for `mul_mat_id` ([#26802](https://github.com/ggml-org/llama.cpp/pull/26802)). Capturing requires fixed input shapes; effective with a fixed batch size and context window during serving.

---

## 2. Memory Allocators (`tcmalloc` / `jemalloc`)

Standard Linux/Windows memory allocators (`glibc malloc` / `msvcrt`) are designed for general-purpose applications and suffer from thread lock contention when multiple CPU threads allocate memory concurrently (such as during KV cache allocation and prefill).

*   **Technique:** Replace the default system allocator with **`tcmalloc`** (Google) or **`jemalloc`** (used by FreeBSD/Rust) which use thread-local cache structures to allocate memory lock-free.
*   **Performance Impact:** Accelerates prefill speed and reduces latency spikes by **10% to 25%** under high concurrency or deep contexts.
*   **How to apply:**
    - On Linux, preload the allocator: `LD_PRELOAD=/usr/lib/libtcmalloc.so.4 ./venv/bin/python -m autoresearch.tuning --plan <plan.json>`
    - On Windows, link the binary against `tcmalloc.lib` or `jemalloc.lib` during compilation.

---

## 3. KV Cache Quantization & Centroid Top-K

As context size grows (e.g., our 65k target context), memory bandwidth becomes the main bottleneck. Loading uncompressed 16-bit keys and values (FP16) from VRAM on every step throttles generation.

*   **Technique:** Quantize the KV Cache to **8-bit (`q8_0`)** or **4-bit (`q4_0`)** formats, or use **KV Centroid Top-K** to load only the most relevant clusters of keys/values instead of the entire history.
*   **Performance Impact:** Saves **50% to 75%** of KV Cache VRAM footprint, directly translating to:
    1.  Lower VRAM memory bandwidth requirements (boosting generation speed at deep contexts).
    2.  Massive headroom to load deeper contexts (e.g., moving from 32k to 65k+ depth).
*   **In `llama.cpp`:**
    - Pass `-ctk q4_0 -ctv q4_0` to compress both keys and values.

*   **Sizing math (measured 2026-08-20, b10375, 8 GB-class):** `gguf_kv_f16_mb`
    (`autoresearch/core/model_arch.py`) returns the f16 MiB; scale by bits/16
    for the actual cache type: `q8_0 ×0.5`, `q4_0/tq4 ×0.25`, `tq2_0 ×0.125`,
    `tq1_0 ×0.0625`. Validated against Ornith-1.5-9B: f16 @65k = 4096 MiB →
    q4_0 = 1024 MiB; measured total ≈ weights 5358 MiB +
    KV 1024 MiB + residual ~1042 MiB (CUDA ctx + compute + allocator slack).
    Per-token q4 cost ≈ 16 KiB (9B), ≈10 KiB (Ornith-1.5-35B A3B, f16 @65k
    2624 MiB). Caveat: the residual is not constant — prefill scratch grows
    with ctx, so 96k on the 9B (~278 MiB margin) is spike-risky.

---

## 4. Embedding Folding (Per-Layer Embeddings)

Embedding tables are typically very large. In models with tied embeddings (where inputs and outputs share the same weight table) or large vocabulary sizes, loading the embedding layer for every token evaluation is expensive.

*   **Technique:** Fold the embedding matrix directly into the initial attention/transformer layers or compress the representation using Per-Layer Embedding (PLE) folding.
*   **Performance Impact:** Streamlines the graph, reducing memory lookup overhead during prefill.

---

## 5. CPU/GPU Offloading Bottlenecks (MoE Routing)

As discovered in our Qwen3.6-35B-A3B and Bonsai-27B speculative decoding benchmarks, neural networks (like Eagle-3 or DSpark) assume that the draft model is running on the same device and is significantly faster than the target model.

*   **The MoE / Offloading Bottleneck:**
    - When active experts are offloaded to CPU to fit in VRAM, speculative decoding forces the engine to run sequential CPU-GPU synchronization calls for every single draft token proposed.
    - This overhead completely destroys throughput (**-60% slowdown**).
*   **Actionable Rule:** Disable neural speculation (`--spec-type none`) for Mixture-of-Experts (MoE) models if experts are running on the CPU. Only use speculative decoding if **both** target and draft models fit entirely in VRAM.

---

## 6. Optimization Decision Matrix for Local Rigs

| Hardware/Model Scenario | Recommended Engine | Essential Flags | Memory Settings |
| :--- | :--- | :--- | :--- |
| **Small Models (<10B) on GPU** | `llama.cpp` / `vLLM` | `-fa on` (CUDA graphs are automatic in llama.cpp; `GGML_CUDA_DISABLE_GRAPHS` to disable) | tcmalloc, MTP active |
| **Large Models (>20B) fully on GPU** | `llama.cpp` | `-ctk q4_0 -ctv q4_0`, `-fa on` | MTP active |
| **MoE Models with CPU expert offloading** | `llama.cpp` | `--n-cpu-moe <N>`, `--spec-type none` | Disable speculative decoding |
| **Ultra low-bit Quantization (e.g. Q1_0)** | `llama.cpp` | `--spec-type none`, `--dry-multiplier 0.8` | Disable speculative decoding, DRY active |

---

## 7. DRY Sampling Guardrails for Ultra-Low-Bit Quants

Aggressively quantized reasoning models (e.g. 1-bit to 2.5-bit quants like Q1Q/Q1Z/Q2) are prone to degenerate repetition and infinite thinking loops under default samplers.

*   **Technique:** Use Don't Repeat Yourself (DRY) sampling to apply an exponential penalty to repeated token sequences dynamically.
*   **Key Flags (`llama.cpp`):**
    - `--dry-multiplier N`: Penalty multiplier strength (default `0.0`, recommended `0.8`).
    - `--dry-base N`: Base for exponential penalty curve (default `1.75`).
    - `--dry-allowed-length N`: Allowed unpenalized repeat length (default `2`).
    - `--dry-penalty-last-n N`: History window scanned for matching prefixes (default `64`).
    - `--dry-sequence-breaker STRING`: Sequence breakers interrupting repetition matching (defaults: `\n`, `:`, `"`, `*`).

---

## 8. Reasoning Budget & Multi-Turn Preservation

When serving reasoning-capable models (e.g. Qwen3.5/3.8, DeepSeek R1-derived architectures) in automated coding or agentic loops, unbounded reasoning tokens can cause latency explosions.

*   **Technique:** Bound reasoning step length and control reasoning trace persistence across multi-turn sessions.
*   **Key Flags (`llama.cpp`):**
    - `--reasoning [on|off|auto]`: Control reasoning template activation (`-rea`).
    - `--reasoning-budget N`: Enforce a hard ceiling of $N$ thinking tokens per generation turn.
    - `--reasoning-budget-message STRING`: Custom transition injected when budget is exhausted before concluding thoughts.
    - `--reasoning-preserve` / `--no-reasoning-preserve`: Control whether thinking traces are kept in long multi-turn context or stripped to conserve KV budget.
    - `--reasoning-format [deepseek|deepseek-legacy|none]`: Structure thought output into `message.reasoning_content`.

---

## 9. Unified Dynamic KV Buffer Allocation (`-kvu`)

*   **Technique:** Allocate a single unified KV buffer shared across all sequences and slots rather than pre-partitioning fixed slot memory slices.
*   **Impact:** Minimizes VRAM memory fragmentation on multi-turn agentic workloads with variable query lengths.
*   **Flag (`llama.cpp`):** `-kvu` / `--kv-unified` (default enabled with `-np auto`).
