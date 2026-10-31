# Laguna-XS-2.1 — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/laguna-xs-2.1.md). Historical performance retains its original protocol.

**GGUF publisher path:** `models/bartowski/laguna-xs-2.1-gguf/Laguna-XS-2.1-Q3_K_XL.gguf` (~16.3 GB)
**License:** `openmdw-1.1` (`general.license`)
**Family:** Laguna XS
**Architecture type:** MoE (`laguna`) — `256x2.2B`, 8 experts active
**Quantization:** `Q3_K_XL` (`general.file_type=13`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-24):

| Key | Value |
| --- | --- |
| `general.architecture` | `laguna` |
| `general.name` | Laguna XS 2.1 |
| `general.size_label` | `256x2.2B` |
| `laguna.block_count` | 40 |
| `laguna.context_length` | 262144 |
| `laguna.embedding_length` | 2048 |
| `laguna.feed_forward_length` | 8192 |
| `laguna.expert_count` | 256 |
| `laguna.expert_used_count` | 8 |
| `laguna.expert_feed_forward_length` | 512 |
| `laguna.expert_shared_feed_forward_length` | 512 |
| `laguna.attention.head_count_kv` | 8 |
| `laguna.vocab_size` | 100352 |
| MTP / `nextn` keys | **none** |

Harness `is_moe_model` → **True** (expert_count > 1). Dense offload rules do not apply.

## Hardware requirements (discrete 8 GB-class NVIDIA)

| Quant | Size | Fit notes |
| --- | --- | --- |
| Q3_K_XL | ~16.3 GB on disk | MoE VITRIOL: `n-cpu-moe 40` (= `block_count`). Peak VRAM **3.6 GB** @ ctx 65k / KV q4_0 |

- Claimed ctx 262k; on 8 GB prefer **65k** + KV `q4_0` (validated agentic + TPS).
- Leave experts on CPU via `--n-cpu-moe 40` (or Baseline `N_CPU_MOE=None` → auto `block_count`).

## Recommended settings

GGUF embeds sampling defaults (`temp=1.0`, `top_p=1.0`, `min_p=0.0`). Local harness Baseline uses search defaults unless overridden.

| Param | Value | Notes |
| --- | --- | --- |
| CTX_SIZE | 65536 | Validated claw-full |
| KV_CACHE_K/V | q4_0 |  |
| BATCH / UBATCH | 512 / 128 |  |
| THREADS / THREADS_BATCH | 8 / 8 |  |
| FLASH_ATTN | on | Required |
| CONT_BATCHING | True |  |
| NO_MMAP | True |  |
| N_CPU_MOE | **40** | = `block_count` |
| TPS_FLOOR | 15.0 | MoE floor on the operator host |
| JINJA | true |  |

## MTP

No MTP / `nextn` tensors. Speculative decoding not applicable from embedded heads.

## Reasoning control

Publisher documents `enable_thinking` via GGUF chat template (local verified 2026-08-29). `--reasoning-effort` is a silent NO-OP on Laguna templates; working levers: `--reasoning on/off` (Baseline REASONING), `--reasoning-budget N` (REASONING_BUDGET), REASONING_PRESERVE (where `/props` `supports_preserve_reasoning` — verified true for Laguna GGUFs).

## VITRIOL split

Experts do not fit physical VRAM — full expert offload:

```text
--n-gpu-layers 99 --n-cpu-moe 40
```

See [vitriol-technique.md](../discovery/vitriol-technique.md). Explicit `40` matches auto-`None` resolve on this GGUF.

## Our config baseline (validated 2026-07-24)

```python
MODEL = 'Laguna-XS-2.1-Q3_K_XL.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
N_CPU_MOE = 40
NO_MMAP = True
CONT_BATCHING = True
JINJA = True
TPS_FLOOR = 15.0
```

### Measured

| Gate | Notes |
| --- | --- |
| bench_tg | cli-bench gen=512 (claw-full row) |
| peak VRAM | Agentic ~3.6; coding ~4.5 |


Evidence: [claw-full](../performance/2026-07-24-claw-full-smoke-high.md), [coding-10](../performance/2026-07-24-coding-10-claw-leaders.md), [lcb-patch](../performance/2026-07-24-lcb-patch-gambiarra.md), [claw leaderboard](../performance/claw-eval-leaderboard.md), [coding leaderboard](../performance/coding-leaderboard.md).

## Sources / Verification

- Local GGUF metadata via `gguf.GGUFReader` — 2026-07-24
- Harness runs logged in `results.tsv` (`agentic-full` / `10-task`) — 2026-07-24
- HF repo `bartowski/laguna-xs-2.1-gguf`: API metadata + README — extraction date 2026-08-29

## Open questions

- None blocking local agentic use. Optional later: ctx >65k headroom test if VRAM_LIMIT raised or KV further compressed.
- Coding remains weak vs Ornith/Mythos — do not use Laguna as coding Baseline.
