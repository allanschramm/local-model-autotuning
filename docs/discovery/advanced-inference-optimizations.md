# Advanced Inference Optimizations Guide

This guide documents advanced, high-performance optimization techniques for local LLM inference engines (such as `llama.cpp` and `vLLM`), drawing lessons from the official **Fast Gemma Challenge** (Google DeepMind & Hugging Face, 2026) and our own empirical benchmarks.

---

## 1. CUDA Graph Capture (Static Execution Graph)

For small models (e.g. Gemma-4 E4B, Qwen3.5-9B), GPU kernels execute so quickly that the **CPU overhead of launching kernels** (CUDA API call latency) becomes the primary performance bottleneck.

*   **Technique:** CUDA Graphs allow the engine to record the execution sequence of GPU kernels once during initialization and replay the entire sequence with a single launch command.
*   **Performance Impact:** Eliminates CPU-to-GPU launch latency. Can boost TPS by **15% to 40%** on small models.
*   **In `llama.cpp`:**
    - Enable using the `--cuda-graph` or `-cgraph` flags.
    - Capturing CUDA graphs requires fixed input shapes. It is most effective when combined with a fixed batch size and context window during serving.

---

## 2. Memory Allocators (`tcmalloc` / `jemalloc`)

Standard Linux/Windows memory allocators (`glibc malloc` / `msvcrt`) are designed for general-purpose applications and suffer from thread lock contention when multiple CPU threads allocate memory concurrently (such as during KV cache allocation and prefill).

*   **Technique:** Replace the default system allocator with **`tcmalloc`** (Google) or **`jemalloc`** (used by FreeBSD/Rust) which use thread-local cache structures to allocate memory lock-free.
*   **Performance Impact:** Accelerates prefill speed and reduces latency spikes by **10% to 25%** under high concurrency or deep contexts.
*   **How to apply:**
    - On Linux, preload the allocator: `LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python3 autoloop.py ...`
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
| **Small Models (<10B) on GPU** | `llama.cpp` / `vLLM` | `--cuda-graph`, `-fa on` | tcmalloc, MTP active |
| **Large Models (>20B) fully on GPU** | `llama.cpp` | `-ctk q4_0 -ctv q4_0`, `-fa on` | MTP active |
| **MoE Models with CPU expert offloading** | `llama.cpp` | `--n-cpu-moe <N>`, `--spec-type none` | Disable speculative decoding |
| **Ultra low-bit Quantization (e.g. Q1_0)** | `llama.cpp` | `--spec-type none` | Disable speculative decoding |
