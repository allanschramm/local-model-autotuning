# K2-Horizon-7B — Model Card (Local)

**Source repo:** https://huggingface.co/IFM/K2-Horizon-7B  
**GGUF repo:** https://huggingface.co/abenzerps/K2-Horizon-7B-GGUF  
**License:** Apache-2.0 (https://huggingface.co/IFM/K2-Horizon-7B/blob/main/LICENSE)  
**Local file:** `models/abenzerps/K2-Horizon-7B-GGUF/K2-Horizon-7B-Q4_K_M.gguf` (~5.33 GB)  
**Family:** K2-Horizon (IFM; custom `k2_horizon` architecture in `llama.cpp-ifm-k2horizon`)  
**Quantization:** Q4_K_M  
**MTP repo:** none (dense transformer backbone; no MTP speculative head)  

---

## Architecture (verified from `IFM/K2-Horizon-7B/config.json` & `llama.cpp-ifm-k2horizon/src/models/k2-horizon.cpp`)

- **Architecture key:** `k2_horizon` (dense transformer backbone)
- **Total physical parameters:** **8,999,178,240 (~8.999 Billion)**
  - *Transformer backbone (36 layers):* $6,946,066,432$ parameters (~6.95B)
  - *Untied 250k vocabulary:* $2 \times (250,624 \times 4096) = 2,053,111,808$ parameters (~2.05B)
  - *Note on nominal naming:* Although labeled "7B", the model physically carries ~9.0B weights due to its untied 250k embedding and output projection matrices.
- **Layers (`block_count`):** 36
- **Hidden dimension (`hidden_size`):** 4,096
- **FFN intermediate dimension (`intermediate_size`):** 12,288 (SwiGLU)
- **Attention mechanism:** **100% Full Dense Quadratic Attention across all 36 layers**
  - Query heads (`num_attention_heads`): 32
  - KV heads (`num_key_value_heads`): 8 (GQA 4:1)
  - Head dimension (`head_dim`): 128
  - KV elements per token per layer: $8 \times (128 + 128) = 2,048$
  - **Total KV elements per token across model:** $36 \times 2048 = \mathbf{73,728\text{ elements/token}}$
- **KV cache bytes per token (`q4_0` quant):**
  $$73,728 \times 0.5625\text{ bytes} = \mathbf{41,472\text{ bytes/token (40.50 KiB/token)}}$$
- **KV Cache size at 65,536 context (`q4_0`):** **2,592.0 MiB (2.531 GiB / 2.718 GB)**

---

## Hardware Requirements & Memory Analysis

| Quantization | Resident Model Weights | KV Cache @ 32k (`q4_0`) | KV Cache @ 65k (`q4_0`) | Total Peak VRAM @ 65k | Fits 8 GB Card? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q4_K_M** | **~5.33 GB** | **~1.30 GB** | **~2.59 GB** | **~8.37 GB** | **NO (VRAM Limit Exceeded / WDDM Paging)** |
| **Q3_K_M** | ~4.25 GB | ~1.30 GB | ~2.59 GB | ~7.29 GB | Fits with tight headroom |
| **Q4_K_M (at 32k ctx)**| **~5.33 GB** | **~1.30 GB** | — | **~7.08 GB** | **YES (Safe baseline on 8 GB cards)** |

### Why K2-Horizon-7B Exceeds 8 GB VRAM at 65k Context
On an 8 GB-class GPU running on Windows, Desktop Window Manager (DWM) and OS compositing consume 300–500 MB of VRAM, leaving ~7.6–7.8 GB for the inference engine:
1. Model weights at Q4_K_M require **5.33 GB**.
2. CUDA runtime contexts, workspace buffers, and scratch compute require **~0.45 GB**.
3. Free VRAM remaining for the KV cache is approximately:
   $$7.70\text{ GB} - (5.33\text{ GB} + 0.45\text{ GB}) \approx \mathbf{1.92\text{ GB}}$$
4. At 65k context, K2-Horizon's 100% full dense attention demands **2.59 GB** for KV cache alone.
5. The total allocation reaches **8.37 GB**, causing Windows WDDM to page activation memory over PCIe to host RAM.

### Comparison: Why Nominal 9B and 12B Models Fit in Less Memory

| Metric | K2-Horizon-7B | Ornith-1.5-9B | Qwen3.8-9B | Gemma-4-12B |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Dense MHA (36 layers) | Hybrid SSM + Attn (`qwen35`) | Hybrid SSM + Attn (`qwen35`) | Sliding Window Attn (`gemma4`) |
| **Total Physical Params** | **8.999 B** | **~8.98 B** | **~8.2 B** | **~12.0 B** |
| **Full Attention Layers** | **36 of 36 (100%)** | **8 of 32 (25%)** | **8 of 32 (25%)** | **8 of 48 (16.7%)** |
| **Recurrent / SSM Layers** | 0 | 24 ($O(1)$ ~51 MB state) | 24 ($O(1)$ ~51 MB state) | 0 |
| **Sliding Window Layers** | 0 | 0 | 0 | 40 (`sliding_window=1024`) |
| **KV Elements / Token** | **73,728** | **16,384** | **16,384** | **17,664** (effective) |
| **KV Cache @ 65k (`q4_0`)**| **2,592 MiB (2.53 GiB)** | **576 MiB (0.56 GiB)** | **576 MiB (0.56 GiB)** | **621 MiB (0.61 GiB)** |
| **KV Cache Multiplier** | **1.00× (Baseline)** | **0.22× (4.50× smaller)** | **0.22× (4.50× smaller)** | **0.24× (4.17× smaller)** |
| **Total VRAM @ 65k (`q4_0`)**| **8.37 GB (Spills)** | **7.01 GB (Fits)** | **6.43 GB (Fits)** | **7.01 GB (Fits @ Q3)** |

---

## Recommended Settings

- **Server build:** `llama.cpp-ifm-k2horizon` (required for `k2_horizon` architecture and `k2-horizon.jinja` chat template).
- **Thinking tags:** Emits `<ifm|think>` and `</ifm|think>` (also supports `<ifm|think_fast>` and `<ifm|think_faster>`).
- **Stop tokens:** Must explicitly pass `<|ifm|im_end|>` (do not rely on default `</s>`).
- **Sampling profiles:**
  - **General reasoning / Agentic:** TEMP 1.0, TOP_P 0.95, TOP_K 40, MIN_P 0.0
  - **Precise coding:** TEMP 0.6, TOP_P 0.95, TOP_K 20, MIN_P 0.0
- **Token budget:** Output tokens should be configured with `max_tokens >= 4096` (or `8192`) because `reasoning_effort` defaults to `high` and short budgets cause mid-thought truncation.

---

## Reasoning Control

- Embedded chat template: [`models/templates/k2-horizon.jinja`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/llama.cpp-ifm-k2horizon/models/templates/k2-horizon.jinja#L816-L880).
- Supports `reasoning_effort` template variables:
  - `high` (default): Emits full deliberate chain-of-thought enclosed in `<ifm|think> ... </ifm|think>`.
  - `medium` / `fast`: Emits `<ifm|think_fast> ... </ifm|think_fast>`.
  - `low` / `faster`: Emits `<ifm|think_faster> ... </ifm|think_faster>`.
- Client requests can pass `chat_template_kwargs: {"reasoning_effort": "low"}` to curb token generation explosion when working under tight token budgets.

---

## VITRIOL / Split Strategy

- Dense architecture — no MoE routed experts (`num_experts = 0`).
- Offload full model to GPU: `-ngl 99`, `N_CPU_MOE=None`.

---

## Local Evaluation Post-Mortem (Trial 2026-09-05)

In local Pareto benchmarking ([`results.db`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/results.db), trial `07f0498f`), K2-Horizon-7B scored:
- **Agentic (Claw-full):** **0.6000** (9/15)
- **Coding-10:** **0.4800**
- **Throughput:** 46.0 t/s

This underperformance was investigated and identified as a compound failure of our evaluation harness:

1. **WDDM Memory Paging Collapse (168× slowdown on long tasks):**
   - On short tasks (T002–T018, <2k tokens), K2-Horizon scored **9/9 (100%)** with an average score of **0.805**.
   - On long research tasks (T044–T054), context reached 4k–7k tokens. Squeezing into an 8.0 GB footprint caused WDDM to page tensors across PCIe.
   - Prompt processing speed collapsed from **>2,000 t/s down to 12.29 t/s** ([`llama-server-20260905-041815-K2-Horizon-7B-Q4_K_M.log#L1550`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/autoresearch/runners/logs/llama-server-20260905-041815-K2-Horizon-7B-Q4_K_M.log#L1550)).
   - Single turns took over 420 seconds, triggering HTTP client timeouts on **all 6 research tasks**.
   - *Cross-family proof:* Siblings with safe VRAM headroom experienced zero timeouts: **K2-Horizon-0.9B** scored **0.8000** (612s) and **K2-Horizon-3.7B** scored **0.7333** (2412s).
2. **Coding Harness Tag Mismatch:**
   - [`benchmark_coding.py`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/autoresearch/benchmarks/benchmark_coding.py#L48-L81) only stripped `<think>.*?</think>`.
   - K2-Horizon's `<ifm|think>` monologue was not stripped. As a result, `_strip_code` extracted draft exploratory code from inside the scratchpad, or submitted raw XML tags resulting in Python `SyntaxError`.
3. **Generation Token Starvation (`max_tokens = 2048`):**
   - In `benchmark_coding.py`, `_MAX_TOKENS = 2048` truncated 13 of 40 coding tasks mid-thought.
4. **Missing Stop Tokens:**
   - Harness passed `stop: ["</s>"]` instead of `<|ifm|im_end|>`.

---

## Sources / Verification

- Primary model card: https://huggingface.co/IFM/K2-Horizon-7B (extracted 2026-09-15)
- Published benchmarks: 70.6% SWE-bench Verified, 39.1% Terminal-Bench 2.1, 73.3% HMMT Feb 2026, 25.8% tau3-Banking, 59.0% BrowseComp
- Architecture source code: [`llama.cpp-ifm-k2horizon/src/models/k2-horizon.cpp`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/llama.cpp-ifm-k2horizon/src/models/k2-horizon.cpp)
- Jinja template: [`llama.cpp-ifm-k2horizon/models/templates/k2-horizon.jinja`](file:///D:/Dev/ailocal-nexus-system/ailocal-model-autotuning/llama.cpp-ifm-k2horizon/models/templates/k2-horizon.jinja)
- Local run logs: `autoresearch/runners/logs/llama-server-20260905-041815-K2-Horizon-7B-Q4_K_M.log` and `agentic-20260905-061333-K2-Horizon-7B-Q4_K_M.json`

---

## Open Questions

- **TurboQuant / 2-bit KV Cache**: Can K2-Horizon-7B run with `tq2_0` or `turbo2` KV cache to reduce KV memory from 2,592 MiB to ~648 MiB at 65k context without degrading reasoning quality?
- **Host Offload of Untied Output Matrix**: Can `output.weight` (842 MB) be offloaded to pinned host RAM while executing the 36 attention layers on GPU to create ~800 MB of VRAM headroom on 8 GB cards?
