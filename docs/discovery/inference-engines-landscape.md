# High-Performance LLM Inference Engines — Architecture & Taxonomy Guide

## Executive Summary

The LLM inference engine ecosystem has matured into distinct specialized tiers based on deployment constraints (datacenter server high-concurrency vs local consumer hardware) and execution strategies (dynamic runtime vs compiled engine graph).

This guide surveys the primary inference engines, analyzing their memory managers, attention mechanisms, KV cache strategies, and hardware targets.

---

## Technical Comparison Matrix

| Engine | Primary Memory Innovation | KV Cache Sharing Mechanism | Primary Target Hardware | Ideal Workload Pattern | Status / Lifecycle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vLLM** | PagedAttention (Virtual Memory Paging) | Block-level Prefix Caching | Datacenter GPUs (NVIDIA/AMD/Intel) | General Production Serving (Multi-tenant API) | Active (Production Standard) |
| **SGLang** | RadixAttention (Trie/Radix Tree) | Token-level Radix Tree Sharing | Datacenter / High-End Workstation GPUs | RAG, Agentic Multi-turn, Structured JSON | Active (Frontier Speed) |
| **TensorRT-LLM** | TensorRT Compiled Engine Graph | Static / Custom Memory Pools | NVIDIA Enterprise GPUs (H100/H200/B200) | Static High-Traffic Monolithic Models | Active (Peak Throughput) |
| **LMDeploy** | TurboMind Engine / Memory Pools | Page-based KV Allocation | NVIDIA Datacenter GPUs | High-throughput server deployment | Active |
| **llama.cpp** | Mmap / Direct CPU/GPU Offload | Static / Slot-based KV | Consumer CPUs, Apple Silicon, Consumer GPUs | Single-user local execution, Edge devices | Active (Local Standard) |
| **Colibrì** | 3-Tier Hierarchy (VRAM $\rightarrow$ RAM $\rightarrow$ NVMe) | LRU Cache + Usage Ledger (`.coli_usage`) | Consumer PCs (~25 GB RAM) | Ultra-large MoE (GLM-5.2 744B) on low RAM | Active (Streaming MoE) |
| **TGI** | Paged KV / FlashAttention | Chunked Prefix Caching | Datacenter GPUs | Legacy HuggingFace serving | Maintenance Mode |

---

## Detailed Engine Breakdown

### 1. vLLM (PagedAttention Architecture)
- **Core Mechanism**: Introduced **PagedAttention**, dividing KV cache into fixed-size physical memory blocks, eliminating external memory fragmentation and reducing internal fragmentation to under 4%.
- **Scheduling**: Continuous Batching (iteration-level scheduling) ensuring decode steps execute without waiting for full sequence completion.
- **Hardware & Architecture**: Broadest support across NVIDIA, AMD (ROCm), Intel Gaudi, and Google TPU. Supports over 400+ model architectures.
- **Strengths**: Enterprise standard, robust OpenAI API server compatibility, seamless multi-GPU Tensor Parallelism (TP) and Pipeline Parallelism (PP).

### 2. SGLang (RadixAttention & Structured Decoding)
- **Core Mechanism**: Uses **RadixAttention**, organizing the KV cache as a dynamic radix tree (trie). Matches prefix tokens hierarchically rather than via discrete block chunks.
- **Optimization Strategy**: Token-level prefix reuse drastically lowers **Time to First Token (TTFT)** when multiple queries share system prompts, RAG context blocks, or multi-turn agent histories.
- **Structured Output**: Native integration with compressed finite-state machine (FSM) grammar execution for accelerated JSON/schema decoding.
- **Strengths**: Best-in-class performance for agentic coding workflows, long-context RAG pipelines, and complex prompt DAGs.

### 3. TensorRT-LLM (Compiled Graph & Custom Kernels)
- **Core Mechanism**: Converts PyTorch model definitions into optimized C++ **TensorRT engine graphs**, performing ahead-of-time (AOT) kernel fusion, fp8/fp4 precision selection, and custom GEMM tuning.
- **In-Flight Batching**: Highly customized C++ execution runtime with custom CUDA kernels (e.g. FlashAttention-3, XQA kernels).
- **Trade-Offs**: Requires compilation per GPU architecture and configuration. High operational setup complexity; less flexible for fast model iteration.
- **Strengths**: Industry-peak raw token throughput for fixed production models on NVIDIA hardware.

### 4. LMDeploy (TurboMind Backend)
- **Core Mechanism**: Developed by OpenMMLab, utilizing the TurboMind C++ inference engine for aggressive memory allocation and fused matrix multiplications.
- **Quantization Support**: Native support for AWQ, INT4, and INT8 weight-only and KV-cache quantization.
- **Strengths**: Extremely low latency and high concurrency serving on NVIDIA H100/A100 clusters with minimal memory overhead.

### 5. llama.cpp (Cross-Platform & Edge Execution)
- **Core Mechanism**: Pure C/C++ engine utilizing custom GGUF quantization formats (Q4_K_M, IQ3_XS, etc.).
- **Hardware Agnostic**: Runs across x86 CPU (AVX2/AVX-512/AMX), ARM (NEON), Apple Silicon (Metal), NVIDIA (CUDA), AMD (HIP), and Vulkan.
- **Local Autotuning Target**: The core backend engine integrated into `local-model-autotuning` via `llama-server` and `llama-cli`.

### 6. Colibrì (Streaming MoE Runtime)
- **Core Mechanism**: Specialized zero-dependency single-file C runtime (`c/glm.c`) designed for 700B+ MoE models (GLM-5.2).
- **Storage Tiering**: Holds dense core in RAM (~9.9 GB int4) while streaming 19,456 routed experts (~370 GB int4) from NVMe SSD on demand.
- **Strengths**: Enables frontier-scale MoE execution on ~25 GB RAM consumer PCs without quality degradation.

---

## Architectural Decision Framework

```
                          [Inference Requirement]
                                     |
           +-------------------------+-------------------------+
           |                                                   |
   [Local / Edge / Consumer]                          [Datacenter / Server]
           |                                                   |
     +-----+-----+                                       +-----+-----+
     |           |                                       |           |
[MoE Stream] [General Local]                       [Prefix Heavy] [Fixed High-Traffic]
     |           |                                       |           |
 [Colibrì]   [llama.cpp]                             [SGLang]   [TensorRT-LLM]
                                                         |
                                                 [General Production]
                                                         |
                                                       [vLLM]
```

1. **Choose vLLM**: For standard enterprise deployment needing broad model support, reliable scaling, and OpenAI-compatible API endpoints.
2. **Choose SGLang**: For long-context RAG, multi-turn AI agents, or heavy prompt sharing where RadixAttention token reuse delivers 20–40% TTFT reduction.
3. **Choose TensorRT-LLM**: For maximum static hardware throughput on dedicated NVIDIA infrastructure where engine compilation cost is acceptable.
4. **Choose llama.cpp / Ollama**: For local development, Apple Silicon, CPU-only execution, or GGUF quantized model deployment.
5. **Choose Colibrì**: For streaming 700B+ MoE models on RAM-constrained consumer hardware.
