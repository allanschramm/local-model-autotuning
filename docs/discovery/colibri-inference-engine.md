# Colibrì Inference Engine — Streaming MoE Runtime & Architecture

## Overview

**Colibrì** (`JustVugg/colibri`) is an open-source, zero-dependency inference runtime written in pure C (`c/glm.c`) designed to execute massive Mixture-of-Experts (MoE) models — specifically **GLM-5.2 (744B parameters)** — on consumer hardware with as little as 25 GB of system RAM.

Unlike conventional LLM runtimes (e.g. `llama.cpp`, `vLLM`) that attempt to fit the entire model into VRAM or system memory, Colibrì exploits MoE parameter sparsity by treating VRAM, RAM, and NVMe storage as a single managed memory hierarchy.

---

## Core Philosophy: Placement Over Fitting

A 744B parameter MoE model activates only **~40B parameters per token**, of which only **~11 GB change from token to token** (the routed experts).

- **The Dense Core** (attention layers, shared experts, embeddings — ~17B parameters) remains **permanently resident in RAM** at INT4 (~9.9 GB).
- **The Routed Experts** (19,456 experts across 75 layers + MTP head, ~19 MB each at INT4) reside on **NVMe SSD storage** (~370 GB total) and are **streamed on demand**.

> **Quality Gate Principle**: Insufficient fast memory reduces generation speed, but Colibrì **never silently degrades precision, prunes experts, or alters router semantics**. Memory tiering affects execution latency only, not output correctness.

---

## Memory Hierarchy & Execution Pipeline

```
+-----------------------------------------------------------------------+
|                       Colibrì Memory Hierarchy                         |
+-----------------------------------------------------------------------+
| Tier 1: VRAM (Optional) -> Resident Core & Pinned Hot Experts          |
| Tier 2: RAM             -> Dense Core (~9.9 GB) + LRU Expert Cache    |
| Tier 3: NVMe SSD        -> Full Expert Pool (~370 GB / 19,456 Experts) |
+-----------------------------------------------------------------------+
```

### The 5-Step Per-Token Execution Pipeline

Every layer for every token executes through five deterministic steps:

1. **Route**: Top-K gating evaluation selects active experts for the token.
2. **Union**: Batch-union deduplication ensures each unique expert is fetched and executed only once per batch.
3. **Place**: Dynamically resolves weight location across VRAM $\rightarrow$ RAM LRU Cache $\rightarrow$ NVMe Storage.
4. **Overlap**: Asynchronous I/O pool (`PIPE=1`) streams missing experts while resident experts execute. A lookahead thread (`PILOT=1`) evaluates router predictions 1 layer ahead (**71.6% lookahead hit rate**).
5. **Learn**: The engine writes routing frequency to an adaptive usage ledger (`.coli_usage`), automatically promoting the most frequently used experts to RAM/VRAM over time.

---

## Technical Features & Performance Mechanics

### 1. Compressed KV Cache via MLA
- Implements Multi-head Latent Attention (MLA) storing **576 floats/token** (vs standard 32,768), representing a **57× KV memory reduction**.
- **Persistent State (`.coli_kv`)**: KV cache states persist across restarts, enabling warm session resume without prompt re-prefill.

### 2. DeepSeek Sparse Attention (DSA)
- Native implementation of GLM-5.2's lightning indexer.
- Validated against full-key dense attention to guarantee zero numerical drift.

### 3. Multi-Token Prediction (MTP) & Speculative Decoding
- Leverages GLM-5.2 native MTP draft heads for speculative decoding (2.2–2.8 tokens per forward pass).
- **INT8 MTP Requirement**: INT4 draft heads collapse to 0–4% acceptance rate due to quantization noise. INT8 MTP heads are mandatory.
- **Kernel Alignment (`SPEC_PIN=1`)**: Draft and verification loops must execute under identical floating-point kernel implementations to avoid spurious rejection.
- **Grammar-Forced Drafts (`GRAMMAR=file.gbnf`)**: Constrains draft token selection to GBNF rules for structured JSON generation.

---

## Hardware Performance Matrix

| Hardware Tier | Memory Configuration | Decode Throughput | Latency / TTFT |
| :--- | :--- | :--- | :--- |
| **6× NVIDIA RTX 5090** | Full VRAM Residency (`CUDA_EXPERT_GB=auto`) | **5.8 – 6.8 tok/s** | ~13s TTFT |
| **128 GB System RAM** | Full RAM Residency / Cached | **~1.8 tok/s** | Warm cache |
| **1× RTX 5070 Ti Laptop** | GPU-Resident Pipeline | **1.07 tok/s** | Hybrid CPU-GPU |
| **25 GB RAM Consumer PC** | NVMe Disk Streaming Baseline | **0.05 – 0.1 tok/s** | Cold NVMe read floor |

---

## Takeaways for Local Model Autotuning

1. **MoE Streaming Bottlenecks**: Bandwidth between NVMe $\rightarrow$ RAM is the primary throughput bottleneck for un-cached experts. Maximizing OS page cache efficiency and pre-fetching is critical.
2. **Speculative Quantization Sensitivity**: MTP / Draft heads are far more sensitive to quantization than base MoE weights (INT8 minimum for MTP vs INT4 for base experts).
3. **Usage-Based Caching**: Workload-adaptive expert pinning (`.coli_usage`) significantly improves steady-state performance over generic static LRU caching.
