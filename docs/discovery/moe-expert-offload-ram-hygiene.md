# MoE Expert Offload and RAM Hygiene: Sizing 35B Architectures on 8GB VRAM / 32GB RAM Rigs

## Purpose

Methodology guide for diagnosing and balancing Mixture-of-Experts (MoE) models (such as `Qwen3.6-35B-A3B`, `Qwen3.8-35B-A3B`, `Ornith-1.5-35B`, `Tiel-Coder-35B`) on commodity desktop rigs with 8 GB VRAM discrete GPUs and 32 GB system RAM.

---

## 1. The MoE Lazy-Loading Illusion on Windows

### A. Private Working Set vs. File-Backed Mmap
When running GGUF inference with `NO_MMAP: False` (the default performance path), the weights file is mapped using Win32 `MapViewOfFile`. 
- **Windows Task Manager "Processes" tab:** Only displays the **Private Working Set** (memory exclusively committed by the process, such as heaps and runtime buffers). The `llama-server.exe` process will typically show only 1 to 2 GB of memory usage.
- **True Physical Footprint:** File-backed mapped pages belong to the **Shareable Working Set**. As weights are accessed, physical RAM pages are committed, but this remains invisible in the per-process memory column.

### B. High Sparsity and Delayed Burst Paging
In architectures like Qwen 35B-A3B:
- 256 routed experts exist per MoE block, but only **8 experts are active per token (3.125% activation)**.
- When serving begins, only the experts relevant to the initial prompts are paged into physical RAM.
- **Domain Shift Risk:** When an evaluation shifts from conversational prompts (e.g. email triage) to domain-specific prompts (e.g. coding, math, tool calling), hundreds of previously untouched experts are referenced simultaneously. This produces a sudden surge of page faults within 1–2 seconds, rapidly consuming several gigabytes of physical RAM and potentially triggering the RAM Circuit Breaker.

---

## 2. The Offload Balancing Math (`N_CPU_MOE`)

In 35B-A3B MoE models, weights divide into two categories:
1. **Shared / Dense Weights:** Attention projections, norms, routers, shared experts (~1.6 GB in Q4_K_M).
2. **MoE Expert Blocks:** Routed feed-forward experts across all layers (~19.1 GB in Q4_K_M across 41 blocks $\approx 465.8\text{ MiB}$ per block).

### The Balancing Trade-off:
- **Full CPU Offload (`N_CPU_MOE = 41`):**
  - VRAM used: $\approx 3.5\text{ to }4.0\text{ GB}$ (leaving $\approx 4\text{ GB}$ of GPU VRAM idle).
  - Host RAM demanded: $\approx 19.1\text{ GB (experts)} + 8.0\text{ GB (OS)} + 2.0\text{ GB (buffers)} \approx 29.1\text{ GB}$.
  - Result on 32 GB system: Physical RAM starvation and Circuit Breaker kills.
- **Targeted Split Offload (`N_CPU_MOE = 31`):**
  - Keeps **10 expert layers on GPU VRAM**: $10 \times 465.8\text{ MiB} \approx 4.65\text{ GB}$.
  - Peak VRAM: $\approx 7.7\text{ GB}$ (fits under the 7,932 MB keepout clamp on 8 GB cards).
  - Host RAM relieved: $\mathbf{-4.65\text{ GB}}$ from system RAM.
  - Result: Ample free RAM (~6 to 7 GB free) and +20% faster token generation (`tps`).

---

## 3. The Context Inviolability Invariant

When encountering memory pressure or circuit breaker trips with MoE models:
> **NEVER reduce `CTX_SIZE` as a workaround.**

### Why Context Reduction is the Wrong Lever:
With modern Grouped-Query Attention (`head_count_kv = 2`):
- A 65,536 token context in `q4_0` consumes only **~735 MiB** of memory.
- Halving context to 32,768 saves less than **370 MiB**, while destroying the benchmark comparability and Pareto frontier position.
- In contrast, shifting just **one expert layer** to GPU VRAM via `N_CPU_MOE` frees **~466 MiB** of host RAM while keeping the full 65k context intact.

---

## 4. Operational Checklist for 35B MoE on 8GB / 32GB Rigs

1. **Calculate per-block expert weight:**
   $$\text{Expert Size per Block} = \frac{\text{Total Expert Weight}}{\text{Block Count}}$$
2. **Determine GPU capacity:**
   $$\text{GPU Layers} = \left\lfloor \frac{\text{VRAM Clamp} - \text{Shared Weights} - \text{KV Cache} - \text{CUDA Overhead}}{\text{Expert Size per Block}} \right\rfloor$$
   For 8 GB cards (~7.9 GB limit) with Q4_K_M: 9 to 10 layers fit comfortably on GPU (`N_CPU_MOE = 31` to `32`).
3. **Calibrate Circuit Breaker Floor:**
   Set `FREE_RAM_FLOOR_MB = 128` to tolerate transient memory spikes during concurrent benchmark test runners.
4. **Host Memory Hygiene:**
   Close heavy desktop applications (browser instances, games, communication apps) to ensure at least 22–24 GB of available physical RAM before launching multi-turn evaluations.
