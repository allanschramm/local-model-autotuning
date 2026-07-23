# LFM2.5-8B-A1B — Model Card (Local)

**Source:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B  
**GGUF:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF  
**Base:** https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base  
**License:** LFM 1.0 (`general.license.name=lfm1.0`)  
**Local file:** `models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf` (~5.16 GB)  
**Family:** Liquid LFM2.5  
**Architecture type:** Hybrid MoE (`lfm2moe`) — 8.3B total / ~1.5B active  
**Quantization:** `Q4_K_M` (`general.file_type=15`)  
**Alias:** `lfm2.5-8b-a1b` (`model-up`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-23):

| Key | Value |
|---|---|
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

## Hardware requirements (RTX 4060 8GB)

| Quant | Size | Fit notes |
|---|---|---|
| Q4_K_M | ~5.16 GB | Full GPU with `--n-cpu-moe 0`. Peak VRAM **6.5 GB** @ ctx 65k / KV q4_0. |

- Context claim 128k; on 8 GB prefer **65k** + KV `q4_0` (validated).
- Do **not** leave `N_CPU_MOE=None` in harness Baseline for this size — see VITRIOL section.

## Recommended settings

From Liquid HF card (2026-07-23):

| Param | Value |
|---|---|
| temperature | 0.2 |
| top_k | 80 |
| repetition_penalty | 1.05 |

Local Baseline (validated):

| Param | Value | Notes |
|---|---|---|
| CTX_SIZE | 65536 | Fits 8 GB with q4_0 KV |
| KV_CACHE_K/V | q4_0 | |
| BATCH / UBATCH | 512 / 256 | |
| FLASH_ATTN | on | Required |
| CONT_BATCHING | True | |
| N_CPU_MOE | **0** | Full VRAM (see below) |
| TPS_FLOOR | 15.0 | MoE floor on this rig |
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

Large MoE that does **not** fit → use positive `--n-cpu-moe N` (Codacus VITRIOL). See [vitriol-technique.md](vitriol-technique.md).

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

Harness: `benchmark_search.py --validation` (claw-quick only; coding off). Runtime: PrismML CUDA binaries (first match in resolver).

| Mode | Bench tg | Peak VRAM | Claw-quick | Agentic wall | Verdict |
|---|---|---|---|---|---|
| Server `exps→CPU` (`N_CPU_MOE=None`) | 174.0 t/s | 2.3 GB | 0.20 (1/5) | ~106 s | KEEP |
| Full VRAM (`N_CPU_MOE=0`) | 174.1 t/s | **6.5 GB** | 0.20 (1/5) | **~38 s** | KEEP |

Only T010_contact_lookup passed in both runs. TPS identical because Combined TPS comes from `llama-cli` bench (already GPU); VRAM/wall-time change is server-side expert placement.

Related tiny dense sibling: alias `lfm2.5-1.2b` — claw-quick **0.80**, ~173 t/s @ ctx 8k f16 (same day queue).

## Sources / Verification

- HF model card LiquidAI/LFM2.5-8B-A1B — sampling + tool-use notes — extracted 2026-07-23.
- Local GGUF metadata via `gguf.GGUFReader` — 2026-07-23.
- Session: [2026-07-23-lfm2.5-8b-a1b-validation.md](../sessions/2026-07-23-lfm2.5-8b-a1b-validation.md).

## Open questions

- Claw-quick 0.20 vs LFM2.5-1.2B 0.80 — Liquid default tool calls are **Pythonic**; Claw-Eval may expect JSON. Confirm adapter / system prompt override before treating as quality fail.
- Whether reasoning / CoT budget flags improve tool-call formatting on this GGUF.
- 131k ctx on 8 GB with heavier KV compression — not tried (65k validated).
