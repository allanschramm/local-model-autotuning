# Gemma-4-26B-A4B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
**MTP guide:** https://unsloth.ai/docs/models/mtp
**License:** Apache-2.0 (Gemma 4 license)
**Local file:** `/mnt/d/LLM-Models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16.2 GB)
**Symlink:** `models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -> /mnt/d/LLM-Models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
**Family:** Gemma 4 (Google DeepMind)
**Quantization:** Unsloth Dynamic 2.0 — `UD-Q4_K_M`

## Architecture (from GGUF metadata, verified via gguf lib)
- Causal LM (text-only, this GGUF has NO vision encoder tensors)
- **`block_count` = 30 layers** (not 50-60 as I guessed earlier — Gemma 4 is shallower)
- **26B total / 4B activated** (MoE, `expert_count=128, expert_used_count=8`)
- Hidden **2816** (not 3072 or 4096), vocab 262144, ctx 256K
- **Hybrid attention: full + sliding window**
  - `rope.freq_base = 1,000,000` (global), `rope.freq_base_swa = 10,000` (sliding window)
  - `rope.dimension_count = 512` (global), `rope.dimension_count_swa = 256` (SWA)
  - **25 layers have separate `attn_v.weight`** (full attention)
  - **5 layers share V with K** (sliding window — V embedded in `attn_k`?)
- `attn_q_norm.weight`, `attn_k_norm.weight` per layer (Qwen-style QK norms)
- **No shared expert** (Gemma 4 MoE doesn't use shared expert — only 128 routed, all specialized)
- MoE structure differs from Qwen:
  - `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` — small dense path (likely for "always-on" expert subset)
  - `ffn_gate_up_exps.weight` — gate+up fused for the 128 routed experts
  - `ffn_down_exps.weight` — down projection
  - `ffn_gate_inp.weight` + `ffn_gate_inp.scale` — router
- **5 norms per layer**: `attn_norm`, `post_attention_norm`, `pre_ffw_norm_2`, `post_ffw_norm`, `post_ffw_norm_1`, `post_ffw_norm_2` (Gemma 4 uses extra norms for stability)
- `layer_output_scale.weight` × 30 — per-layer output scaling (recent training technique)
- Token embeddings: Q8_0 (2816 × 262144)
- `rope_freqs.weight` × 1 — pre-computed RoPE cache (Gemma-style)
- **`general.name` = `Gemma-4-26B-A4B-It`**, file_type=15 (Q4_K_M), quantization_version=2
- 658 tensors total
- **NO MTP tensors in this GGUF.** Unsloth says the MTP file is in a "sub-folder within the GGUF package" — but our downloaded file has no MTP tensors. **TBD: investigate if sub-folder downloads are needed.**

## Variant lineup (Gemma 4 family)
| Variant | Type | Best fit |
|---|---|---|
| E2B | Dense + PLE (128K) | Phone / edge, ASR |
| E4B | Dense + PLE (128K) | Laptop, fast multimodal |
| 12B Unified | Dense (256K) | Laptop, multimodal |
| **26B-A4B** | **MoE (256K)** | **Best speed/quality tradeoff for computer use** |
| 31B | Dense (256K) | Max quality, slower |

## Hardware requirements (per Unsloth)
| Variant | 4-bit | 8-bit | BF16/FP16 |
|---|---|---|---|
| E2B | 4 GB | 5–8 GB | 10 GB |
| E4B | 5.5–6 GB | 9–12 GB | 16 GB |
| 12B | 7–8 GB | 13–14 GB | 25 GB |
| **26B-A4B** | **16–18 GB** | **28–30 GB** | **52 GB** |
| 31B | 17–20 GB | 34–38 GB | 62 GB |

**Our target:** 8 GB VRAM + 16 GB RAM = 24 GB. **We are OVER** the 4-bit budget (16-18 GB) if we naively load all to RAM+VRAM. SSD/HDD offload is the path; partial GPU for active 4B + rest on CPU/RAM.

## Recommended Settings (per Unsloth)
TBD — Unsloth doc section was truncated at 5k chars. **Re-extract** the "🦙 llama.cpp Guide" subsection before first run.

## MTP (Multi-Token Prediction)
- Gemma 4 MTP released **Jun 9** (Unsloth)
- Trained separately from the base model by Google
- For Gemma 4: **MTP is in a sub-folder inside the GGUF package** (per Unsloth), NOT embedded in the main `*.gguf` file. Our downloaded `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` has **no MTP tensors** (verified via `GGUFReader`).
- **To get MTP working, investigate the HF repo's file tree** — there may be a sibling `mtp-*` file or a sub-folder that needs explicit download.
- llama.cpp flags (from our turboquant build, `common/arg.cpp`):
  ```
  --spec-type mtp
  --spec-draft-n-max 2    # try 1-6
  --spec-draft-type-k q4_0
  --spec-draft-type-v q4_0
  --n-cpu-moe-draft N
  ```
- Hardware with MTP (per Unsloth):
  | Variant | 4-bit | 8-bit | BF16/FP16 |
  |---|---|---|---|
  | **26B-A4B** | **17–18 GB** | **29–31 GB** | **53 GB** |
  | 31B | 18–21 GB | 35–39 GB | 63 GB |
- **+2 GB RAM/VRAM headroom** for MTP draft
- **Speedup:** 1.5× – 2.2× faster (Gemma 4 QAT + MTP benchmark from Unsloth)
- Dense models (like 31B) benefit MOST from MTP (>1.4×). MoE (26B-A4B) benefits less.

## QAT variants
Gemma 4 QAT (Quantization-Aware Training) reduces memory ~3× while preserving quality. Already have QAT variants for E4B and 12B locally (`*qat-UD-Q4_K_XL`). For 26B-A4B, the `UD-Q4_K_M` is what we downloaded — should also be QAT-derived via the "smart 4-bit recovery process" Unsloth describes.

## Sampling
TBD — Unsloth doc truncated. Likely similar pattern to other Gemma 4 variants. To be re-extracted.

## VITRIOL / Split strategy (Codacus technique)
Same source as Qwen3.6 card: https://www.youtube.com/watch?v=ZwNCsUTNWOA

**The 2-knob split for MoE on small VRAM:**
1. `--n-gpu-layers 99` — attention + shared expert + routing on GPU
2. `--n-cpu-moe <N>` — first N layers' MoE experts on CPU

Gemma 4 26B-A4B specifics:
- 26B total, 4B active per token (vs Qwen3.6's 3B)
- 16 GB file → ~16 GB CPU/RAM
- Active 4B per token = ~2.5-3.5 GB on VRAM (vs 1.8-2.5 for Qwen3.6 3B)
- Compared to Qwen3.6: **slightly more VRAM pressure per token, slightly less RAM pressure overall**
- Layer count TBD (not in extracted spec yet) — need to count from the GGUF or HF model card. Estimate: 50-60 layers (similar to Gemma 3 27B which had 62). **Run `llama-server` once with verbose to see, or extract from GGUF metadata.**

Expected performance:
- 4B active > 3B active → marginally slower per token than Qwen3.6
- Smaller file (16 vs 22 GB) → marginally faster to load, less RAM swapping
- Codacus got 18 tok/s on i5+1070+16GB with Qwen3.6 35B-A3B. Our 4060+similar RAM should hit 20-30 tok/s with Gemma4 26B-A4B.

MTP: `--spec-type mtp` (corrected flag, not `draft-mtp` — see Qwen3.6 card for the bug story).

## Our config baseline
- `CTX_SIZE`: **65536** (atualizado em commit `78d54e2`; Validar se valor ótimo para 26B-A4B é o mesmo ou se precisa reduzir)
- `KV_CACHE_K = KV_CACHE_V`: `q4_0` primeiro; [Validar quais TurboQuant types nosso build expõe para KV cache]
- `SPEC_TYPE`: Validar — Gemma 4 MTP está em sub-pasta do GGUF, não confirmado que este arquivo local tem os tensores MTP. Se não tiver, fica `none` por ora.
- `SPEC_DRAFT_N_MAX = 2` (se MTP funcionar)
- `THREADS = 8`
- `BATCH_SIZE = 512` `UBATCH_SIZE = 128`
- `FLASH_ATTN = "on"`
- `--n-gpu-layers`: Validar (autoloop vai descobrir)
- `--n-cpu-moe`: **15** (para 30 layers — mantém metade dos experts no CPU/RAM, padrão configurado em commit `2bd795b`)
- Sampling: Validar — Unsloth doc truncado antes da seção "Recommended Settings"

### Teste real (commit `2bd795b`)
- Rodou `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` no nosso rig (8 GB VRAM) com VITRIOL (`--n-cpu-moe 15`)
- **HumanEval pass@1 = 0.533 (16/30)**
- **TPS ~13-18** (abaixo do floor de 20 — score zerado, descarte)
- Resultado fica como evidência da viabilidade da técnica; tuning posterior precisa melhorar TPS

## Sources / Verification
- HF model card (extracted 2026-06-15)
- Unsloth Gemma 4 doc (https://unsloth.ai/docs/models/gemma-4.md, extracted same day, truncated at 5k chars)
- Unsloth MTP guide (https://unsloth.ai/docs/models/mtp.md, extracted same day, truncated at 5k chars)

## Open questions
1. **[Resolvido]** `--spec-type draft-mtp` é inválido no turboquant build — usar `--spec-type mtp`. Verificado por inspeção de `common/arg.cpp`.
2. Validar: o GGUF local realmente NÃO tem tensores MTP (verificado por `GGUFReader`), ou seja, MTP para Gemma 4 requer re-download da sub-pasta do HF (Unsloth menciona que MTP fica em "sub-folder within the GGUF package").
3. Validar: exato valor de `--n-gpu-layers` para 4B-active MoE em 8 GB VRAM. Commit `2bd795b` rodou Gemma 4 só com `--n-cpu-moe 15`, sem anotar o `--n-gpu-layers` exato.
4. Validar: re-extrair seção "🦙 llama.cpp Guide" e "Recommended Settings" do Unsloth doc (estava truncado em 5k chars no extrato original).
5. Validar: sampling params exatos para Gemma 4 (doc truncado antes da seção).

## Notes on prior runs
Earlier 9B-MTP tuner runs included Gemma-4-E4B and Gemma-4-12B in `results.tsv` — both lost badly to Qwen3.5-9B on Coding/Retrieval. The 26B-A4B is the first large Gemma we test; expect different ranking.
