# Qwythos-9B-Claude-Mythos-5-1M — Model Card (Local)

**Source repo:** https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF
**Base model (fine-tune):** https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M → base `Qwen/Qwen3.5-9B`
**MTP GGUF repo:** same repo above — `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` is a sibling file (see HF siblings); the older "mradermacher re-host" note below is obsolete
**License:** Apache-2.0 (GGUF weights, inherited from Qwen3.5-9B base which is also Apache-2.0; fine-tune weights likewise Apache-2.0 — publisher: "Weights are released under Apache-2.0, inherited from the Qwen3.5-9B base")
**Local files:**
- `models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` (5.3 GB) — base, no `nextn`
- `models/Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` (~5.4 GB) — MTP, `nextn` embedded
- `models/Qwythos-9B-v2-Q4_K_M.gguf` (5.74 GB) — v2, no CUDA MTP
**Family:** Qwythos (based on Qwen 3.5 architecture)
**Architecture type:** Dense (all params active per token)
**MTP:** Optional training head (orthogonal to architecture — both dense and MoE models can have MTP)
**Quantization:** `Q4_K_M` (file_type=15)

## Architecture (from GGUF metadata)
- Causal LM (hybrid Attention + SSM — Qwen 3.5 arch)
- **`block_count` = 32 layers**
- Hidden **4096**, context 1,048,576** (1M tokens)
- **Hybrid Attention + SSM (Mamba-2 style)**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=32`, `inner_size=4096`
  - 8 full attention layers, 24 SSM layers
- `rope.freq_base = 10,000,000`

## Hardware Requirements (discrete 8 GB-class NVIDIA)
| Quant | Size | VRAM (idle) | VRAM (131k ctx) |
| --- | --- | --- | --- |
| Q4_K_M | 5.3 GB | ~5.5 GB | ~7.5 GB |

Fits entirely in 8 GB with 131k context and flash-attn.

**Hard constraint: ctx >= 100k always.** Small-ctx tests are irrelevant for this model's use case.

## Batch/Ubatch Sweet Spot (llama-bench 2026-06-30)

**8 GB-class discrete NVIDIA — Qwythos 9B Q4_K_M — pp512 / tg128:**

| ubatch | pp512 (t/s) | tg128 (t/s) |
| -------: | -----------: | ----------: |
| 64 | 1480.68 | **50.16** |
| 128 | 1814.17 | 42.15 |
| **256** | **1922.61** | **49.75** |
| 512 | 1939.81 | 41.03 |
| 1024 | 1529.87 | 41.50 |
| 2048 | 1917.10 | 48.98 |
| 4096 | 1849.93 | 42.12 |

**Winner: ubatch=256** — best balance of prompt processing (1922 t/s) and text generation (49.8 t/s). ubatch=512 gives +1% pp speed but loses 18% tg speed.

## Recommended Settings

| Param | Value | Rationale |
| --- | --- | --- |
| TEMP | 0.6 | Per HF card for Qwythos reasoning model |
| TOP_P | 0.95 | Default nucleus sampling |
| TOP_K | 20 | Focused token pool |
| REPEAT_PENALTY | 1.05 | Light penalty for repetition reduction |
| BATCH_SIZE | 1024 | Matched with ubatch=256 (agentic path) |
| UBATCH_SIZE | 256 | Sweet spot on 8 GB-class discrete NVIDIA (llama-bench) |
| SPEC_TYPE | None | Mythos MTP not worth it (measured +1% short-gen) |
| SPEC_DRAFT_N_MAX | 0 | Disabled |

## MTP (Multi-Token Prediction)

**MTP is not a model class.** Orthogonal to Dense/MoE — see [README.md](README.md).

### Local assets (2026-07-20)
- Base (no `nextn`): `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`
- MTP GGUF (embedded `nextn`): `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` — sibling file in the same HF repo (HF siblings list confirms `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf`). The older mradermacher re-host note is obsolete.

### Measured short-gen TPS (fair matrix)
- Base **40.8 t/s** vs MTP **41.2 t/s** (**+1%**); MTP wall clock worse (29.3s vs 18.4s).
- **Verdict for speed:** keep **non-MTP**. Evidence: [session](../performance/2026-07-20-small-model-tps-matrix.md).

### Long-ctx / agentic (still true)
- At 131k on 8 GB, MTP overhead historically hurt (timeouts / collapsed TPS in mid-2026 Beellama runs below). Short cli-bench does not contradict that.

## MoE split (VITRIOL)

N/A — dense model; `--n-cpu-moe` not applicable.

## Qwythos-9B-v2 (same card family)
- Local: `Qwythos-9B-v2-Q4_K_M.gguf` — fair matrix base **40.1 t/s**
- **No CUDA GGUF MTP** on Hub as of 2026-07-20 (MLX-only MTP exists — useless here)
- **Historical runtime:** `Qwythos-9B-v2-MTP-Q4_K_M.gguf` runtime record (TPS 34.5, VRAM 7.9 GB). Useful CUDA MTP GGUF for v2 was hard to re-acquire from Hub (MLX-only common). Notes: [session](../performance/2026-07-23-nanbeige42-tps-matrix.md).

## Validation Bench (2 tasks each, 2026-06-30)

All runs with `config.py` defaults unless noted.

| Run | Server | Model | Batch/Ubatch | TPS | VRAM |
| ----- | -------- | ------- | ------------- | ---- | ------ |
| 1 | turboquant | non-MTP | 512/128 | 52.6 | 7.3 GB |
| **2** | **turboquant** | **non-MTP** | **1024/256** | **51.2** | **7.5 GB** |
| 3 | beellama | MTP | 1024/256 | — | — |
| 4 | beellama | MTP + draft-ctx 512 | 1024/256 | — | — |
| 5 | beellama | MTP @ 8k ctx | 1024/256 | 42.0 | 7.3 GB |
| 6 | upstream build-cuda | non-MTP | 1024/256 | 50.4 | 7.1 GB |

**Winner: Run 2** — non-MTP, 1024/256, turboquant.

Run 6 (2026-07-01): upstream build-cuda + cont-batching.

## Config Baseline (2026-06-30)

```python
MODEL = 'Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf'
CTX_SIZE = 131072
BATCH_SIZE = 1024
UBATCH_SIZE = 256
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
REPEAT_PENALTY = 1.05
```

## Tuning History
- 2026-06-30: Initial validation (512/128 → 0.1250)
- 2026-06-30: Batch sweep (1024/256 → 0.4250, 3.4× improvement)
- 2026-06-30: llama-bench ubatch sweep (ub=256 is sweet spot)
- 2026-06-30: MTP tested — no VRAM headroom on 8GB at 131k ctx
- 2026-07-01: MTP variant found no speedup — non-MTP preferred
- 2026-07-20: MTP GGUF re-downloaded; short-gen matrix +1% only — keep non-MTP for speed
- 2026-08-29: HF refresh — Sources/Verification dated, license lineage (Apache-2.0 inherited from Qwen3.5-9B base) recorded, MTP repo corrected to the primary empero-ai GGUF repo (sibling file, not mradermacher re-host). Measured data untouched.

## Sources/Verification
Extraction date: 2026-08-29.
Publisher URLs:
- GGUF repo: https://huggingface.co/api/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF (JSON includes `context_length: 1048576`, tags `1M-context`, `reasoning`, `long-context`)
- Base model card: https://huggingface.co/api/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M (JSON includes `license: apache-2.0`, `base_model: Qwen/Qwen3.5-9B`, chat_template jinja reads `enable_thinking`)
- HF README: https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/raw/main/README.md (covers recommended sampling, 1M token context, vision, MTP, reasoning notes)
License lineage: Apache-2.0 (GGUF weights and fine‑tune weights, inherited from Qwen3.5‑9B base).
Hardware & ctx verification: YaRN rope‑scaling documented in GGUF architecture; `context_length: 1048576` matches the 1 M‑token marketing claim.
Recommended settings derived from HF card: TEMP 0.6, TOP_P 0.95, TOP_K 20, REPEAT_PENALTY 1.05, BATCH_SIZE 1024, UBATCH_SIZE 256, SPEC_TYPE None.

## Open questions
- Reasoning control per publisher/harness: The GGUF chat template reads only `enable_thinking` (default true). The `--reasoning-effort` flag is a silent NO‑OP on qwen35‑family GGUFs; the effective levers are `--reasoning on/off` (Baseline REASONING), `--reasoning-budget N` (REASONING_BUDGET, server‑side), and `REASONING_PRESERVE` where `supports_preserve_reasoning` is true (verified for Ornith‑1.5 GGUFs but not confirmed for Qwythos). Confirmation needed whether Qwythos‑9B respects `enable_thinking` or requires `--reasoning on/off`.
- Whether the MTP head in this model produces measurable speed gains beyond the +1 % short‑gen increase already observed; additional long‑ctx or agentic benchmarks are lacking.
- Exact VRAM ceiling at 1 M‑token context with Q4_K_M on 8 GB discrete GPU (theoretical KV‑cache size vs measured 7.5 GB at 131k ctx; headroom unknown for full 1M).
- Whether the v2 quant (`Qwythos-9B-v2-Q4_K_M.gguf`) will receive a CUDA MTP build; currently only MLX‑only MTP exists on Hub.
