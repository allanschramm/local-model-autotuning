# Ternary-Bonsai-2-27B-PTQ1_0 — Model Card

**Source:** https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf  
**Base:** Qwen3.8-27B (hybrid attention causal language model: ~75% linear SSM / ~25% full attention; 64 blocks, dense)  
**License:** Apache 2.0  
**Family:** Bonsai 2 (PrismML-Eng) — dense ternary packing (`PTQ1_0`, 1.75 bits/weight, ~5.95 GB)  
**Runtime:** requires PrismML CUDA release `llama.cpp-releases/prismml/prism-b10669-6f1c690` (`AUTORESEARCH_LLAMA_CPP_ROOT`). Stock upstream `llama.cpp` does not support custom GGML ternary type 143 (`PTQ1_0`).

## Measured Trial (RTX 4060 8 GB)

Measured via `benchmark_search.py --agentic-full` on 2026-09-18 (`trial_id: 90615cd1-2578-4e2d-85a8-4942e3881688`):

| Metric | Measured Value | Threshold / Target | Status |
|---|---|---|---|
| **Pareto Status** | **`on_front`** | Frontier entrant | **ON_FRONT** |
| **Agentic (Claw-15)** | **0.8000 (12/15)** | $\ge 0.6000$ | **PASS** |
| **Coding (Coding-10)** | **0.4650** | LCB 0.40 / HE+ 0.50 / MBPP+ 0.80 / BC 0.00 | **PASS** |
| **Generation Throughput (`TPS`)** | **36.2 t/s** | $\ge 20.0\text{ t/s}$ (`TPS_FLOOR`) | **PASS** |
| **Throughput Probe (`bench_tg`)** | **29.6 t/s** | $\ge 20.0\text{ t/s}$ (`TPS_FLOOR`) | **PASS** |
| **Peak VRAM** | **7.7 GB** | Fits physical 8 GB VRAM | **PASS** |
| **Context Length (`CTX_SIZE`)** | 32768 (32k) | Floor 2048 | Configured |
| **KV Cache Type** | `q4_0` | Memory safety | Configured |

Smoke validation record (`trial_id: 25d1a4f4-5d9b-4084-b0c7-32998d791b7d`): `bench_tg 26.2 t/s`, quick smoke 5/5 (1.0000), peak VRAM 7.9 GB.

## Recommended Settings (Model Card & Baseline)

```yaml
engine:
  model: Ternary-Bonsai-2-27B-PTQ1_0.gguf
  ctx_size: 32768
  n_gpu_layers: 99
  kv_cache: q4_0
  kv_cache_k: q4_0
  kv_cache_v: q4_0
  batch_size: 512
  ubatch_size: 128
  threads: 8
  threads_batch: 8
  flash_attn: on
  n_cpu_moe: null
  spec_type: null

sampler:
  temp: 1.0
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  repeat_penalty: 1.0
  presence_penalty: 0.0
```

## Notes & Comparison to Bonsai 1

- **Bonsai 1 (historical):** `Ternary-Bonsai-27B-Q2_0` was rejected on this host (~10.6 t/s).
- **Bonsai 2 (`PTQ1_0`):** Enters the Pareto frontier (`on_front`) on the 8 GB operator rig, delivering **36.2 t/s** generation throughput, **0.8000** on Claw-15, and **0.4650** on Coding-10 while remaining within **7.7 GB** peak VRAM at 32k context.
