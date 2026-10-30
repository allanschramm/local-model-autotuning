# High-Performance LLM Inference Engines — Architecture & Taxonomy Guide

## Executive Summary

The LLM inference engine ecosystem has matured into distinct specialized tiers based on deployment constraints (datacenter server high-concurrency vs local consumer hardware) and execution strategies (dynamic runtime vs compiled engine graph).

This guide surveys the primary inference engines, analyzing their memory managers, attention mechanisms, KV cache strategies, and hardware targets.

> Fastest-TPS on the operator host (discrete 8 GB-class NVIDIA / Windows): see [fastest-tps-inference-engine.md](./fastest-tps-inference-engine.md) — llama.cpp CUDA stays baseline; ExLlamaV3/EXL3 is the only Windows-capable raw-TPS challenger.

### Comparing engines without confusing formats

| Level | Weights | What you may claim |
| :-- | :-- | :-- |
| **Same-bytes** | Identical checkpoint on every engine that can load it (e.g. GGUF on llama.cpp and on vLLM/SGLang NVIDIA GGUF paths) | Engine difference |
| **Native-stack** | Best practical pack per engine (GGUF vs AWQ vs NVFP4, …) | Compound engine+format — label it |

vLLM/SGLang are Linux-first (WSL2/Docker on Windows). Size every pack against Baseline `VRAM_LIMIT_MB` / no-spill rules; HF multimodal 4-bit packs are often much larger than text GGUF Q4 of the same family ([quantization-cascade.md](./quantization-cascade.md) § HF vs GGUF).

---

## Quantization equivalence — the "Q4_K_M of each engine"

Q4_K_M is llama.cpp's classic 4-bit "good enough" pack: mixed 4/6-bit (attention + first/last blocks at 6-bit), ~0.1–0.15 PPL of FP16 on WikiText, 1–3% MMLU loss, ~95% of BF16 quality on 8B (**external / unverified on this rig**). GGUF-side quality cascade (UD-Q4_K_XL / UD-Q4_K_M > Q4_K_M > IQ4_XS for compression): [quantization-cascade.md](./quantization-cascade.md). Every other engine has a format filling the same niche (≈4-bit, ~1–3% quality-loss band):

| Engine | Q4_K_M-equivalent | Hardware gate | Same-bytes path? |
| :-- | :-- | :-- | :-- |
| **llama.cpp** | Q4_K_M (classic) · UD-Q4_K_XL/UD-Q4_K_M (best 4-bit) · IQ4_XS (smallest) | any (CPU→GPU) | — (native) |
| **vLLM** | AWQ W4A16 gs128 (best 4-bit quality) · GPTQ W4A16 (4/8-bit flexible) · NVFP4 W4A16 (ModelOpt) | AWQ/GPTQ SM75+ (Marlin; Conch SM80+, Machete SM90+) · NVFP4 W4A4 SM100+ / W4A16-Marlin SM75+ | GGUF via `vllm-gguf-plugin` (OOT, experimental, GPU-only) |
| **SGLang** | AWQ/GPTQ (Triton/vLLM kernels; `awq_marlin`/`gptq_marlin` CUDA-only) · torchao int4wo-128 · NVFP4 (SM100+) | matrix: [sglang-inference-engine.md](./sglang-inference-engine.md) §5.1 | GGUF (`--load-format gguf`, NVIDIA-only) |
| **TensorRT-LLM** | FP8 · NVFP4 · MXFP4 (ModelOpt) · INT4 AWQ (legacy) | compiled per-GPU; W4A4 needs Blackwell | ❌ |
| **ExLlamaV3** | EXL3 / QTIP (~4 bpw) | consumer NVIDIA, source build on Windows | ❌ (EXL3-only format) |
| **LMDeploy** | AWQ / INT4 / INT8 | NVIDIA datacenter | ❌ |
| **TGI** | GPTQ / AWQ / FP8 | datacenter | ❌ |

Best-per-engine on this repo's hardware class (discrete 8 GB-class NVIDIA, SM89):

- **llama.cpp**: UD-Q4_K_XL — repo-measured (Qwen3.5-9B-UD, gemma-4-E4B rows in `results.tsv`; cascade doc).
- **vLLM / SGLang**: AWQ (Marlin W4A16, SM75+ — runs on this GPU class) — or GGUF when a same-bytes claim is wanted (vLLM plugin experimental/under-optimized; SGLang GGUF NVIDIA-only). **NVFP4 also loads on this class via Marlin W4A16 fallback** (memory win, W4A16 speed — not native W4A4) — see [nvfp4-quantization.md](./nvfp4-quantization.md). See [vllm-quant-deep-dive.md](./vllm-quant-deep-dive.md) §5 for the SM75/SM80-usable list (GPTQ/AWQ/INT8/FP8-W8A16-Marlin/GGUF/NVFP4-Marlin; bnb = slow dequant path).
- **Native Blackwell tier** (Blackwell rig, for speed): NVFP4 **W4A4** or MXFP4 — native FP4 tensor cores, ~2.3× INT4 throughput (**external**); W4A16 is the same checkpoint without the activation-quant speed edge. FP8 W8A8 is the 8-bit tier (Q8_0-class quality), **not** a Q4 equivalent.

Quality ordering (**external / unverified on this rig**): UD-Q4_K_XL ≈ AWQ > GPTQ ≈ Q4_K_M > Q4_0. NVFP4 W4A16 sits in the AWQ band (2–4× better KLD than W4A4 from the same weights — config switch, not re-quant; NVIDIA forums 2026); NVFP4 W4A4 slightly behind INT4/AWQ on some tasks; AWQ beats GPTQ in PPL/MMLU and is more calibration-robust (AWQ paper arXiv 2306.00978; gingerlabs 2024; Microsoft Data Science guide). Caveat: one vLLM-GGUF-plugin benchmark showed GGUF worst-PPL-but-best-HumanEval among quants — GGUF path was unoptimized there; not a llama.cpp verdict (r/LocalLLaMA, 2026-05).

**TBDs (research sharpened, still open):**

- **TBD:** AWQ vs GGUF Q4_K_M quality parity **on this rig** — external evidence only; no local measurement. Falsifiable: same model, same tasks, llama.cpp Q4_K_M vs SGLang/vLLM AWQ. The blocker from issue #59 (SGLang harness path) is **resolved 2026-08-18**: `SGLangServerRunner` + `venv-sglang` (WSL2) exist and produced a `backend=sglang` row; the parity run is now executable, still unmeasured. Next missing piece: a Qwen3.8-class AWQ pack (no AWQ/GPTQ pack exists on HF for Qwen3.8-9B as of 2026-08-18 — the 2B fp16 fit is the only same-family SGLang comparison so far).
- **TBD:** SGLang `--load-format gguf` same-bytes run — support still documented (server args; NVIDIA-only) and on the SGLang 2H2026 quantization refactor roadmap (GGUF/Autoround standardization); never executed on this hardware. `venv-sglang` now exists (WSL2), so the run is executable — still unmeasured.
- **TBD:** NVFP4 Marlin fallback (W4A16) vs GGUF Q4_K_M on the 8 GB-class rig — NVFP4 *loads* (SM89 Marlin), but TPS/quality gap unmeasured. Falsifiable: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` via `vllm --quantization modelopt_fp4` (Marlin) vs `Ornith-1.5-35B-A3B-GGUF Q4_K_M` via `llama-server`, same tasks. Native W4A4 delta still needs Blackwell.
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
- **Quantization Support**: AWQ, GPTQ, bitsandbytes, GGUF (GPU-only), FP8/INT8/INT4 via LLM Compressor, NVFP4 via NVIDIA Model Optimizer. Full matrix: [vllm-quantization.md](./vllm-quantization.md).
- **Strengths**: Enterprise standard, robust OpenAI API server compatibility, seamless multi-GPU Tensor Parallelism (TP) and Pipeline Parallelism (PP).

### 2. SGLang (RadixAttention & Structured Decoding)
- **Core Mechanism**: Uses **RadixAttention**, organizing the KV cache as a dynamic radix tree (trie). Matches prefix tokens hierarchically rather than via discrete block chunks.
- **Optimization Strategy**: Token-level prefix reuse drastically lowers **Time to First Token (TTFT)** when multiple queries share system prompts, RAG context blocks, or multi-turn agent histories.
- **Structured Output**: Native integration with compressed finite-state machine (FSM) grammar execution for accelerated JSON/schema decoding.
- **Strengths**: Best-in-class performance for agentic coding workflows, long-context RAG pipelines, and complex prompt DAGs.
- **Deep dive**: [sglang-inference-engine.md](./sglang-inference-engine.md) — RadixAttention internals, XGrammar structured outputs, speculative decoding (EAGLE/MTP/DFlash), quantization matrix, hardware support, GGUF/Windows limits.

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
- **Local Autotuning Target**: The core backend engine integrated into `local-model-autotuning` via `llama-server` and `llama-cli`. Default Search path is **upstream** `llama.cpp` (MoE: `--n-cpu-moe`). Arch forks only when required by the GGUF.
- **Related fork (not default):** [Randozart/VITRIOL](https://github.com/Randozart/VITRIOL) — page-locked host experts + GPU PCIe DMA (+ optional Chimera CUDA/Vulkan). Study notes: [vitriol-technique.md](vitriol-technique.md).

### 5.1 The `llama.cpp` Fork & Distribution Ecosystem

The `llama.cpp` ecosystem includes distinct classes of forks and distributions, ranging from user-facing GUI wrappers to deep architectural and kernel experiments. The matrix below benchmarks four prominent projects against the requirements of autonomous tuning on consumer 8 GB-class hardware:

| Project | Primary Focus | Architecture & Backend | KV Cache & Speculative Tech | 8 GB Hardware Relevance | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp)** | Turn-key consumer app / WebUI | Standard `llama.cpp` CUDA/Vulkan/Metal + Python server wrapper | Context Shift (sliding KV cache recycling), Smart Context | Identical kernel TPS ceiling; adds process wrapper overhead | **Wrapper — No TPS gain** |
| **[antimatter15/alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)** | Early historical prototype (March 2023) | Pre-GGUF / Pre-GGJT `.bin`, CPU-only AVX2 | None (hardcoded Alpaca instruction prompt template) | Obsolete; lacks CUDA, GGUF, modern quants, RoPE, MoE | **Dead end (Historical artifact)** |
| **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** | Low-bit KV cache compression | `llama.cpp` fork + custom WHT / Polar Quant kernels | `turbo2`, `turbo3`, `turbo4` (2–4 bit KV), Asymmetric KV (`q8_0`/`turbo3`) | Fits 65k–131k ctx on tight VRAM; ~5–10% decode TPS penalty | **Specialized niche (High ctx only)** |
| **[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** | Experimental performance & speculative decoding | Aggressive `llama.cpp` fork + DFlash drafter / TCQ | KVarN, KV precision tail, DFlash, CopySpec, TCQ (`turbo3_tcq`) | Measured: -20% baseline TPS, DFlash collapses (3.3 TPS), server crashes | **Measured dead end on 8 GB** |

#### Detailed Comparative Breakdown

1. **LostRuins/koboldcpp** ([GitHub](https://github.com/LostRuins/koboldcpp), release `v1.120` published 2026-08-29; active commits 2026-09-02; 11.6k stars):
   - *Mechanism*: Packages `llama.cpp` and `ggml` C/C++ backend, embedded Python HTTP server, and the Kobold Lite web UI into a single self-contained PyInstaller executable (`koboldcpp.exe` on Windows).
   - *Memory & Inference*: Features mature **Context Shifting** (sliding KV cache without reprocessing prompts, cutting multi-turn prompt ingestion latency by up to ~85% in chat/roleplay), Smart Context, DirectIO model loading (`--usedirectio`), combined mlock+mmap, and multimodal extensions (Stable Diffusion, Whisper, OuteTTS). Supports new architectures like Qwen3.8-Flash-Next and Ling-3.0-flash.
   - *Assessment on 8 GB rig*: Relies on stock `llama.cpp` compute kernels; generation TPS ceiling is identical to upstream `llama.server`. For automated benchmarking and search loops (Claw-Eval, LiveCodeBench), the bundled web server adds process overhead and API latency without compute gains.

2. **antimatter15/alpaca.cpp** ([GitHub](https://github.com/antimatter15/alpaca.cpp), created 2023-03-16; release `81bd894` on 2023-03-21; dormant since 2023-04-19; 10.1k stars):
   - *Mechanism*: Created in mid-March 2023 immediately following Stanford Alpaca 7B, this was the first viral fork enabling local instruction-following on personal computers.
   - *Assessment on 8 GB rig*: Entirely obsolete. Built on pre-GGUF/pre-GGJT legacy `.bin` formats (`ggml-alpaca-7b-q4.bin`), CPU-only (no GPU acceleration), hardcoded prompt formats (`### Instruction: ... ### Response:`), and primitive 4-bit quantizers. Completely superseded by upstream `llama.cpp`.

3. **TheTom/llama-cpp-turboquant** ([GitHub](https://github.com/TheTom/llama-cpp-turboquant), created 2026-03-25; release `tqp-v0.3.0` published 2026-07-12; active commits 2026-09-02; 2.3k stars; companion `TheTom/turboquant_plus`):
   - *Mechanism*: Implements Walsh-Hadamard Transform (WHT) rotation and polar quantization based on TurboQuant (ICLR 2026) to compress Key-Value caches down to 2–4 bits (`turbo2`, `turbo3`, `turbo4`). Release `tqp-v0.3.0` added DFlash speculative decoding (#201), server slot save/restore across restarts via `.ckpt` sidecars (#206; ~720x delta prefill on 100k sessions), and fused-MMA decode for `head_dim 128` (up to +69% at 131k ctx).
   - *Assessment on 8 GB rig*: Niche lever for long-context survival ([`2026-08-01-turboquant-release-research.md`](../sessions/2026-08-01-turboquant-release-research.md)). Documented best practice is asymmetric KV (`--cache-type-k q8_0 --cache-type-v turbo3`). Tradeoffs: incurs ~5–10% decode TPS overhead due to dequantization math; on GQA 8:1 models `turbo4` K-cache auto-upgrades to `q8_0` yielding negligible savings over standard `q4_0` ([`CONTEXT.md`](../../CONTEXT.md)); tested Ornith 9B @ 100k still exceeded the 7900 MB physical limit with MTP ([`2026-08-01-ornith-turboquant-100k.md`](../performance/2026-08-01-ornith-turboquant-100k.md)). Treat as an isolated external binary toolchain (`llama.cpp-releases/turboquant/tqp-v0.3.0`), not the default engine.

4. **Anbeeld/beellama.cpp** ([GitHub](https://github.com/Anbeeld/beellama.cpp), created 2026-05-05; release `v0.4.4` published 2026-08-29; active commits 2026-09-02; 1.0k stars):
   - *Mechanism*: Performance testbed tracking `llama.cpp` (base `6fdd0ac89`, ggml `0.22.0`) and implementing DFlash/DFlash2 (cross-attention feature-level speculative decoding), CopySpec (context suffix-matching speculation), KVarN (variance-normalized KV cache quantization), KV precision tail (keeping recent N tokens in FP16), TCQ, MTP multi-ubatch synchronization, and pathological reasoning-loop force-close detection.
   - *Assessment on 8 GB rig*: Measured dead end ([`2026-06-29-beellama-tcq-copyspec-dflash-iq3.md`](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-06-29-beellama-tcq-copyspec-dflash-iq3.md)). Pure inference baseline ran 20% slower than stock fork (Ornith 9B: 41.7 vs 52.2 TPS; 35B MoE: 26.2 vs 31.5 TPS). Speculative decoding collapsed under hardware constraints: DFlash on GPU dropped throughput to 3.3 TPS (draft model competed for saturated compute); DFlash CPU offload OOMed system RAM; CopySpec slowed MoE decode; TCQ (`turbo3_tcq`) crashed with HTTP 500 at 131k ctx on hybrid architectures; server suffered connection drops between benchmark rounds. Suitable primarily for 24 GB+ GPUs with compute surplus, not 8 GB budget tiers.


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
