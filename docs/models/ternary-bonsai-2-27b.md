# Ternary-Bonsai-2-27B-PTQ1_0 — Model Card

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/ternary-bonsai-2-27b.md). Historical performance retains its original protocol.

**Source:** https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
**Base:** Qwen3.8-27B (hybrid attention causal language model: ~75% linear SSM / ~25% full attention; 64 blocks, dense)
**License:** Apache 2.0
**Family:** Bonsai 2 (PrismML-Eng) — dense ternary packing (`PTQ1_0`, 1.75 bits/weight, ~5.95 GB)
**Runtime:** requires PrismML CUDA release `llama.cpp-releases/prismml/prism-b10669-6f1c690` (`AUTORESEARCH_LLAMA_CPP_ROOT`). Stock upstream `llama.cpp` does not support custom GGML ternary type 143 (`PTQ1_0`).

## Measured Trial (8 GB-class)


| Metric | Measured Value | Threshold / Target | Status |
| --- | --- | --- | --- |
| **Pareto Status** | **`on_front`** | Frontier entrant | **ON_FRONT** |
| **Generation Throughput (`TPS`)** | **36.2 t/s** | $\ge 20.0\text{ t/s}$ (`TPS_FLOOR`) | **PASS** |
| **Throughput Probe (`bench_tg`)** | **29.6 t/s** | $\ge 20.0\text{ t/s}$ (`TPS_FLOOR`) | **PASS** |
| **Peak VRAM** | **7.7 GB** | Fits physical 8 GB VRAM | **PASS** |
| **Context Length (`CTX_SIZE`)** | 32768 (32k) | Floor 2048 | Configured |
| **KV Cache Type** | `q4_0` | Memory safety | Configured |


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
- **Bonsai 2 (`PTQ1_0`):** Historical runtime on the 8 GB-class configuration: **36.2 t/s** generation throughput; and while remaining within **7.7 GB** peak VRAM at 32k context.
