# LFM2.5-8B-A1B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/lfm2.5-8b-a1b.md). Historical performance retains its original protocol.

**Source:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
**GGUF:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF
**Base:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base
**License:** LFM 1.0 (`general.license.name=lfm1.0`)
**Local file:** `models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf` (~5.16 GB)
**Family:** Liquid LFM2.5
**Architecture type:** Hybrid MoE (`lfm2moe`) — 8.3B total / ~1.5B active
**Quantization:** `Q4_K_M` (`general.file_type=15`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-23):

| Key | Value |
| --- | --- |
| `general.architecture` | `lfm2moe` |
| `general.size_label` | `32x959M` |
| `lfm2moe.block_count` | 24 |
| `lfm2moe.context_length` | 128000 |
| `lfm2moe.embedding_length` | 2048 |
| `lfm2moe.feed_forward_length` | 7168 |
| `lfm2moe.expert_feed_forward_length` | 1792 |
| `lfm2moe.expert_count` | 32 |
| `lfm2moe.expert_used_count` | 4 |
| `lfm2moe.leading_dense_block_count` | 2 |
| `lfm2moe.attention.head_count` | 32 |
| `lfm2moe.attention.head_count_kv` | sparse GQA (8 on attn layers; 0 on conv layers) |
| `lfm2moe.vocab_size` | 128000 |
| `lfm2moe.rope.freq_base` | 5_000_000 |
| `lfm2moe.shortconv.l_cache` | 3 |
| MTP / `nextn` keys | **none** |

HF card: 18 double-gated LIV conv + 6 GQA layers (matches 24 blocks). Hybrid — not a pure transformer MoE.

## Hardware requirements (discrete 8 GB-class NVIDIA)

| Quant | Size | Fit notes |
| --- | --- | --- |
| Q4_K_M | ~5.16 GB | Full GPU with `--n-cpu-moe 0`. Peak VRAM **6.5 GB** @ ctx 65k / KV q4_0. |

- Context claim 128k; on 8 GB prefer **65k** + KV `q4_0` (validated).
- Do **not** leave `N_CPU_MOE=None` in harness Baseline for this size — see VITRIOL section.

## Recommended settings

From Liquid HF card (2026-08-29):

| Param | Value |
| --- | --- |
| temperature | 0.2 |
| top_k | 80 |
| repetition_penalty | 1.05 |

Local Baseline (validated):

| Param | Value | Notes |
| --- | --- | --- |
| CTX_SIZE | 65536 | Fits 8 GB with q4_0 KV |
| KV_CACHE_K/V | q4_0 |  |
| BATCH / UBATCH | 512 / 256 |  |
| FLASH_ATTN | on | Required |
| CONT_BATCHING | True |  |
| N_CPU_MOE | **0** | Full VRAM (see below) |
| TPS_FLOOR | 15.0 | MoE floor on the operator host |
| JINJA | true | ChatML-like template |

## MTP

No MTP / `nextn` tensors in this GGUF. Speculative decoding not applicable from embedded heads.

## VITRIOL split

Model fits physical VRAM — prefer **full GPU**:

```text
--n-gpu-layers 99 --n-cpu-moe 0
```

Harness gotcha (2026-07-23): for MoE filenames matching `A1B` / etc., if Baseline `N_CPU_MOE` is `None`, `LlamaServerRunner` injects `--override-tensor .*exps.*=CPU` (all experts on CPU). That drops peak VRAM to ~2.3 GB and slows agentic wall time; **bench TPS is unchanged** because `llama-cli` bench did not apply the same offload in the first trial.

Set `N_CPU_MOE: 0` to pass `--n-cpu-moe 0` and keep experts on GPU.

Large MoE that does **not** fit → use positive `--n-cpu-moe N` (Codacus VITRIOL). See [vitriol-technique.md](../discovery/vitriol-technique.md).

## Our config baseline (2026-07-23)

```python
ENGINE_DEFAULTS = {
    'MODEL': 'LFM2.5-8B-A1B-Q4_K_M.gguf',
    'CTX_SIZE': 65536,
    'KV_CACHE': 'q4_0',
    'KV_CACHE_K': 'q4_0',
    'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512,
    'UBATCH_SIZE': 256,
    'THREADS': 8,
    'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on',
    'SPEC_TYPE': None,
    'SPEC_DRAFT_N_MAX': 0,
    'SPEC_DRAFT_MODEL': None,
    'NO_MMAP': False,
    'JINJA': True,
    'CONT_BATCHING': True,
    'N_CPU_MOE': 0,
    'VRAM_LIMIT_MB': 7900,
    'TPS_FLOOR': 15.0,
}
SAMPLER_DEFAULTS = {
    'TEMP': 0.2,
    'TOP_P': 0.95,
    'TOP_K': 80,
    'MIN_P': 0.0,
    'REPEAT_PENALTY': 1.05,
    'PRESENCE_PENALTY': 0.0,
    'FREQUENCY_PENALTY': None,
}
```

## Measured (2026-07-23)


| Mode | Peak VRAM |
| --- | --- |
| Server `exps→CPU` (`N_CPU_MOE=None`) | 2.3 GB |
| Full VRAM (`N_CPU_MOE=0`) | **6.5 GB** |

TPS identical because Combined TPS comes from `llama-cli` bench (already GPU); VRAM/wall-time change is server-side expert placement.

## Historical runtime evidence (Measured (validation + coding-10, 2026-08-03))

`N_CPU_MOE=0`, ctx 65k, KV q4_0 (upstream CUDA), coding-10 on:

| Run | peak VRAM |
| --- | --- |
| A | 6.8 GB |

Combined TPS 186.9 (≥15.0). Coding axis now measured; agentic axis still needs Claw full for a complete Objective Vector.

## Real peak VRAM (2026-08, gate bypass)

Measured with `scripts/measure_vram_peak.py` (harness `LlamaServerRunner`, NVML 20 ms sampler, preflight bypassed):

| Stage | used | free |
| --- | --- | --- |
| pre | 1622 MB (desktop) | 6335 MB |
| post-load | 7021 MB (+5399) | 936 MB |
| post-gen (534 tok, 171.3 t/s) | 7015 MB | 942 MB |
| **peak** | **7260 MB** (total; llama's own ≈ 5638) | — |

Model fits physical VRAM with ~700 MB spare in a dirty-GPU state — no shared-memory spill. This exposed the KV estimator over-charge (flat 80 KB/token): this arch has `head_count_kv = 0` (per-layer sparse GQA array, KV only on 6 attention layers) so real KV @ 65k q4_0 ≈ 480–720 MB, not 1434. Estimator now derives KV bytes/token from GGUF metadata (`n_layer × n_head_kv × (k_len+v_len)`, per-layer array supported) — est 5324 MB vs real 5399/5638 MB (2026-08-08).

## Historical runtime evidence (Measured (claw-full, 2026-07-24))

`N_CPU_MOE=0`, ctx 65k, KV q4_0 (upstream CUDA):

| Run | peak VRAM |
| --- | --- |
| A | 7.3 GB |
| B (re-run) | 7.8 GB |


Related tiny dense sibling: alias `lfm2.5-1.2b` — **166–180 t/s** @ ctx **65k** f16. Card: [lfm2.5-1.2b.md](lfm2.5-1.2b.md).

## Sources / Verification

- HF model card LiquidAI/LFM2.5-8B-A1B — benchmarks, architecture, generation params — extracted **2026-08-29**.
- HF model card LiquidAI/LFM2.5-8B-A1B-GGUF — GGUF quant variants — extracted **2026-08-29**.
- HF model card LiquidAI/LFM2.5-8B-A1B-DSpark — speculative-decoding drafter (328M) — extracted 2026-08-29.
- Local GGUF metadata via `gguf.GGUFReader` — 2026-07-23.
- Session: [2026-07-23-lfm2.5-8b-a1b-validation.md](../performance/2026-07-23-lfm2.5-8b-a1b-validation.md).
- Claw-full: [2026-07-24-claw-full-top-tps.md](../performance/2026-07-24-claw-full-top-tps.md).

## Reasoning control

The `lfm2` family chat template contains **no reasoning/thinking control variables** (`reasoning_effort`, `enable_thinking`, `thinking_budget`). Reasoning flags (`--reasoning`, `--reasoning-budget`, `--reasoning-effort`) are **not applicable** to this model.

Publisher note (HF README 2026-08-29): "Because LFM2.5-8B-A1B is a reasoning model, assistant turns contain an explicit chain of thought before the final answer." — this is a built-in model behavior, not user-controllable via template flags.

## Open questions

- 131k ctx on 8 GB with heavier KV compression — not tried (65k validated).
- DSpark (328M speculative-draft companion) — SGLang-only per HF card 2026-08-29; not applicable to our llama.cpp stack.

## Preserved historical runtime tables

| Mode | Bench tg | Peak VRAM |
| --- | --- | --- |
| Server `exps→CPU` (`N_CPU_MOE=None`) | 174.0 t/s | 2.3 GB |
| Full VRAM (`N_CPU_MOE=0`) | 174.1 t/s | **6.5 GB** |

| Run | bench_tg | peak VRAM |
| --- | --- | --- |
| A | 171.2 | 6.8 GB |

| Run | bench_tg | peak VRAM |
| --- | --- | --- |
| A | 184.3 | 7.3 GB |
| B (re-run) | 178.5 | 7.8 GB |
