# LFM2.5-1.2B-Instruct — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/lfm2.5-1.2b.md). Historical performance retains its original protocol.

**Source repo**: [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)  ·  Base: [LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base)  ·  GGUF: [lmstudio-community/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF)  ·  Extracted 2026-08-29
**License:** `other` (name: `lfm1.0`)  ·  HF `lastModified`: 2026-08-24
**GGUF path:** `models/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf`  ·  **Family:** Liquid LFM2.5 (`lfm2`) — dense hybrid, **not MoE**  ·  **Quant:** `Q8_0`  ·  **Alias:** `lfm2.5-1.2b`

## Architecture (from GGUF metadata)


Reasoning control: `lfm2` family has NO thinking variables in the embedded chat template
(verified 2026-08-29 on the 2.6B sibling; same `lfm2` template lineage). No reasoning-related
template fields are read → `--reasoning`, `--reasoning-effort`, `--reasoning-budget`, and
`--reasoning-preserve` are inert on these GGUFs. Do not seed reasoning flags in Baseline for
this alias. (Reasoning models in this family ship as a separate `LFM2.5-1.2B-Thinking` checkpoint
with its own template — not the Instruct GGUF we measured.)

| Key | Value |
| --- | --- |
| `general.architecture` | `lfm2` |
| `general.name` | LiquidAI_LFM2.5 1.2B Instruct |
| `general.size_label` | 1.2B |
| `lfm2.context_length` | **128000** |
| `lfm2.block_count` | 16 (harness arch log) |
| Publisher block layout (HF, 2026-08-29) | 16 total = 10 double-gated convolution + 6 GQA |
| MTP / `nextn` | **none** |
Harness: dense → `N_CPU_MOE` must stay `None` (no expert offload).

## Hardware requirements (discrete 8 GB-class NVIDIA)

Tiny weights (~1.2B Q8). Context + KV dominate VRAM at ceiling.

| ctx | KV | Fits? | Notes |
| --- | --- | --- | --- |
| 128k | f16 | **No** | preflight est ~11.5 GB > 7900 |
| 65k | f16 | **Yes** | peak **3.3 GB** (preferred) |

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

### HF publisher recommended settings (2026-08-29)

```python
# From HF model card / README (LiquidAI/LFM2.5-1.2B-Instruct)
TEMPERATURE = 0.1        # default generation
TOP_K = 50               # default generation
REPETITION_PENALTY = 1.05
# Context (HF native): 32,768 tokens; GGUF extended: 128,000 via `lfm2.context_length`
# Not recommended for knowledge-intensive tasks or programming (per HF card text).
# Recommended for: agentic tasks, data extraction, RAG.
```

## MTP

No embedded MTP. Speculative N/A from this GGUF.

## VITRIOL

N/A (dense).

## Historical runtime evidence (Measured (claw-quick, 2026-07-24))

| Setup | TPS | peak VRAM |
| --- | --- | --- |
| **65k f16 (alias)** | **180.6** | 3.3 GB |
| 128k q4_0 | 178.6 | 3.4 GB |
| 128k q8_0 | 177.6 | 3.7 GB |
| 128k f16 | — | est 11.5 GB |

Evidence: [session 2026-07-24](../performance/2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md). Sibling MoE: [lfm2.5-8b-a1b.md](lfm2.5-8b-a1b.md).

## Historical runtime evidence (Measured (claw-full, 2026-07-24))

| Setup | TPS | peak VRAM |
| :------- | ----: | ----------: |
| **65k f16 (alias)** | **166.4** | 3.7 GB |

Evidence: [session top-TPS full](../performance/2026-07-24-claw-full-top-tps.md). Leaderboard: [claw-eval-leaderboard.md](../performance/claw-eval-leaderboard.md).

## Historical runtime evidence (Measured (coding-10, 2026-07-24))

| Metric | Value |
| --- | --- |
| bench_tg | 167.6 t/s |
| peak VRAM | 4.0 GB |


## Sources / Verification

- Local GGUF `lfm2.context_length` via `GGUFReader` — 2026-07-24
- Harness `--validation` / `--agentic-full` / `--include-coding` rows in `results.tsv` — 2026-07-24
- HF `LiquidAI/LFM2.5-1.2B-Instruct` README + cardData via API: license `lfm1.0`, benchmarks, generation params (`temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`), block layout (10 conv + 6 GQA), context `32768` native — extracted **2026-08-29** (repo `lastModified`: 2026-08-24)

## Open questions

- Whether 128k q4 smoke 0.60 vs 65k f16 0.80 is KV quality or n=5 noise — optional 128k full later.
- HF native context is `32,768` tokens but our GGUF reports `lfm2.context_length = 128000`; origin of the GGUF context extension (upstream patch or lmstudio-community build) is unconfirmed — verify next time the GGUF is available.
