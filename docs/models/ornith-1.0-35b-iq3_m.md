# Ornith-1.0-35B IQ3_M — Variant Card (Local)

**Source repo:** https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF
**License:** MIT
**Local file:** `models/deepreinforce-ai_Ornith-1.0-35B-IQ3_M.gguf` (15.74 GB)
**Family:** Ornith (based on Qwen 3.5 MoE architecture)
**Quantization:** `IQ3_M` (file_type=27, importance-quantized 3-bit)
**Base model card:** [ornith-1.0-35b.md](ornith-1.0-35b.md) (Q4_K_M variant)

## Architecture (from GGUF metadata, verified via gguf.GGUFReader 2026-07-02)
- Same architecture as Q4_K_M variant — only quantization differs
- **`block_count` = 40 layers**
- **35B total / 3B activated** (MoE, `expert_count=256, expert_used_count=8` + 1 shared expert)
- Hidden **2048**, vocab 248320, ctx **262144**
- Hybrid Attention + SSM (Mamba-2 style) + MoE: `full_attention_interval=4`
- `rope.freq_base = 10,000,000`
- **`general.name` = `Ornith 1.0 35B`**, file_type=27 (IQ3_M), quantization_version=2
- `general.license` = `mit` (different from Q4_K_M's Apache-2.0 — likely different upstream release)
- 733 tensors total, no MTP tensors

## Hardware requirements
| Quant | File Size | Est. VRAM |
|---|---|---|
| Q4_K_M (reference) | 19.7 GB | ~24 GB total |
| **IQ3_M (this variant)** | **15.74 GB** | **~19 GB total** |

Saves ~4 GB vs Q4_K_M. May fit on 8 GB VRAM + 16 GB RAM with VITRIOL split.

## Recommended Settings
Same as Q4_K_M variant:
- **Temperature:** 0.6
- **Top P:** 0.95
- **Top K:** 20
- **Repeat Penalty:** 1.0

## MTP (Multi-Token Prediction)
- **NO MTP tensors in this GGUF.**

## Testing Results
From `ornith-1.0-35b.md` tuning history:
- **IQ3_M is slower than Q4_K_M at every n-cpu-moe setting**
- Q3 decode kernel less optimized on RTX 4060
- **Not recommended** — smaller file does not translate to better performance

## Our config baseline
Same as Q4_K_M variant with adjusted file:
- `MODEL = 'deepreinforce-ai_Ornith-1.0-35B-IQ3_M.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `N_CPU_MOE = 32`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`

## Sources / Verification
- HuggingFace: `deepreinforce-ai/Ornith-1.0-35B-GGUF`
- GGUF metadata verified via `gguf.GGUFReader` on 2026-07-02
- Testing results from Q4_K_M card tuning history (2026-06-29)

## Open questions
- None — tested and rejected. Q4_K_M is the preferred quant for this model.
