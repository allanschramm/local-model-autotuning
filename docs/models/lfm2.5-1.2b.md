# LFM2.5-1.2B-Instruct — Model Card (Local)

**GGUF path:** `models/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf`  
**Family:** Liquid LFM2.5  
**Architecture type:** Dense hybrid (`lfm2`) — not MoE  
**Quantization:** `Q8_0`  
**Alias:** `lfm2.5-1.2b` (`model-up`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` (2026-07-24):

| Key | Value |
|---|---|
| `general.architecture` | `lfm2` |
| `general.name` | LiquidAI_LFM2.5 1.2B Instruct |
| `general.size_label` | 1.2B |
| `lfm2.context_length` | **128000** |
| `lfm2.block_count` | 16 (harness arch log) |
| MTP / `nextn` | **none** |

Harness: dense → `N_CPU_MOE` must stay `None` (no expert offload).

## Hardware requirements (RTX 4060 8GB)

Tiny weights (~1.2B Q8). Context + KV dominate VRAM at ceiling.

| ctx | KV | Fits? | Notes |
|---|---|---|---|
| 128k | f16 | **No** | preflight est ~11.5 GB > 7900 |
| 65k | f16 | **Yes** | peak **3.3 GB** (preferred) |
| 128k | q4_0 / q8_0 | **Yes** | peak 3.4–3.7 GB; lower smoke/TPS |

## Recommended settings / config baseline

```python
MODEL = 'LFM2.5-1.2B-Instruct-Q8_0.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'f16'
KV_CACHE_V = 'f16'
BATCH_SIZE = 512
UBATCH_SIZE = 256
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
NO_MMAP = True
JINJA = True
N_CPU_MOE = None
TPS_FLOOR = 15.0
```

## MTP

No embedded MTP. Speculative N/A from this GGUF.

## VITRIOL

N/A (dense).

## Measured (claw-quick, 2026-07-24)

| Setup | Score | TPS | peak VRAM |
|---|---|---|---|
| **65k f16 (alias)** | **0.80** | **180.6** | 3.3 GB |
| 128k q4_0 | 0.60 | 178.6 | 3.4 GB |
| 128k q8_0 | 0.60 | 177.6 | 3.7 GB |
| 128k f16 | FAIL preflight | — | est 11.5 GB |

Evidence: [session 2026-07-24](../sessions/2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md). Sibling MoE: [lfm2.5-8b-a1b.md](lfm2.5-8b-a1b.md).

## Sources / Verification

- Local GGUF `lfm2.context_length` via `GGUFReader` — 2026-07-24
- Harness `--validation` rows in `results.tsv` — 2026-07-24

## Open questions

- Claw-Eval full not run (tiny model; optional later).
- Whether 128k q4 smoke 0.60 vs 65k f16 0.80 is KV quality or n=5 noise — full would clarify.
