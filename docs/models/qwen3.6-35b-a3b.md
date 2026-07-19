# Qwen3.6-35B-A3B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.6
**MTP-specific repo:** https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
**License:** Apache-2.0
**Local file:** `$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.11 GB, baixado do repo MTP-GGUF em 2026-06-19). `$MODELS_DIR` defaults to `~/models` or set via env.
**Symlink:** `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -> $MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

**MTP-GGUF (validado 2026-06-19):** o arquivo em disco é do repo `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`. Validado via llama-server log: `qwen35moe.nextn_predict_layers u32 = 1` confirma que os tensores MTP estão integrados no `.gguf`. NÃO é o mesmo arquivo do repo base `Qwen3.6-35B-A3B-GGUF` (esse NÃO tem MTP). Filename **não contém "MTP"**, então o auto-detect do `llama_runner.py` (linha 189) só dispara se passar `spec_type` explícito.
**Family:** Qwen3.6 (Alibaba)
**Quantization:** Unsloth Dynamic 2.0 — `UD-Q4_K_M` (calibrated on real-world use-cases, important layers upcasted)

## Architecture (from GGUF metadata, verified via gguf lib)
- Causal LM (text-only, no vision encoder in this GGUF)
- **`block_count` = 40 layers** (confirmed from `qwen35moe.block_count`)
- **35B total / 3B activated** (MoE, `expert_count=256, expert_used_count=8` + 1 shared expert)
- Hidden 2048, vocab 248320, ctx 262144
- RoPE dim 64, freq_base 10,000,000
- **Hybrid architecture: 10 full-attention + 30 DeltaNet (linear attention) layers**
  - `full_attention_interval = 4` — every 4th block is full attention
  - 30 layers have `ssm_a`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out`, `ssm_alpha`, `ssm_beta` tensors
  - 10 layers have `attn_q`, `attn_k`, `attn_v`, `attn_output` (16 Q heads, 2 KV heads, dim 256)
  - Gated attention: 30 layers have `attn_gate.weight` (linear)
- `ssm.conv_kernel=4, ssm.group_count=16, ssm.inner_size=4096, ssm.state_size=128, ssm.time_step_rank=32`
- **Shared expert**: 1 per layer (`ffn_*_shexp.weight` in Q8_0). 256 routed experts in Q4_K.
- Token embeddings: Q8_0 (preserved precision). Output: Q6_K (preserved precision).
- **`general.name` = `Qwen3.6-35B-A3B`**, file_type=15 (Q4_K_M), quantization_version=2
- 733 tensors total, 19 per block × 40 blocks ≈ 760
- **NO MTP tensors in this GGUF.** The base repo (`Qwen3.6-35B-A3B-GGUF`) does NOT include MTP. To use MTP, re-download from the `Qwen3.6-35B-A3B-MTP-GGUF` repo.

## Hardware requirements (per Unsloth docs)
| Quant | Total RAM (RAM + VRAM) |
|---|---|
| 3-bit | 17 GB |
| **4-bit (our pick)** | **23 GB** |
| 6-bit | 30 GB |
| 8-bit | 38 GB |
| BF16 | 70 GB |

**Our target:** 8 GB VRAM (RTX 4060) + 16 GB RAM = 24 GB total. We are AT the 4-bit edge — split (active layers GPU / rest CPU) is mandatory.

**Warnings from Unsloth:**
- **Do NOT use CUDA 13.2** (gibberish outputs). Use <13.2 or 13.3+.
- If gibberish, ctx too low OR try `--cache-type-k bf16 --cache-type-v bf16`.
- VRAM+RAM should exceed quant file size; otherwise SSD/HDD offload works but is slower.

## Sampling (per Unsloth Recommended Settings)

### Thinking mode (default — Qwen3.6 is hybrid thinking)
| Param | General tasks | Precise coding (e.g. WebDev) |
|---|---|---|
| temperature | 1.0 | **0.6** |
| top_p | 0.95 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 0.0 |
| repeat_penalty | 1.0 (off) | 1.0 (off) |

### Instruct (non-thinking) mode
| Param | General |
|---|---|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | **1.5** |
| repeat_penalty | 1.0 (off) |

To disable thinking: `--chat-template-kwargs '{"enable_thinking": false}'` (TBD: confirm exact flag spelling in Qwen3.6 chat template — Unsloth doc was truncated at this point).
**Preserve Thinking** is a new Qwen3.6 feature (TBD: details, doc truncated).

## Context
- **Max native:** 262,144 tokens
- Extensible to 1M via YaRN
- **Adequate output length:** 32,768 tokens
- Recommended `presence_penalty` range: 0.0 to 2.0 (off by default; higher may slightly decrease perf)

## MTP (Multi-Token Prediction)
- **Status (2026-06-19):** MTP-GGUF já baixado e validado em disco. Tensores MTP presentes no arquivo (`nextn_predict_layers = 1` confirmado via log).
- **Speedup empírico medido (MESMO PROMPT, MESMAS FLAGS, só file swap):**
  - Base GGUF → 11.1 tok/s (`r_q4`)
  - MTP-GGUF (sem flag MTP ativa) → 22.5 tok/s (`mtp_baseline`)
  - **Ganho: 2.03× só com file swap.** Provável causa: MTP tensors carregados auxiliam mesmo inativos, OU perfil de quantização Unsloth SOTA.
- **Speedup projetado com MTP ativo** (próximo teste, não executado): +1.15-1.25× pra MoE → ~26-28 tok/s.
- llama.cpp flags (upstream build):
  ```
  --spec-type draft-mtp         # Use draft-mtp on standard/upstream llama.cpp (mtp is rejected by standard CLI)
  --spec-draft-n-max 2          # Unsloth recommends; test 1-6
  --spec-draft-type-k q4_0
  --spec-draft-type-v q4_0
  ```
- **Auto-detect:** `llama_runner.py:189-205` is configured to map speculative decoding.
- **Cuidado:** thinking mode é default, consome tokens sem output visível. Pra ver resposta final, `enable_thinking: false` no payload ou `max_tokens > 200`.
- **Distinção crítica:** speculative decoding com draft model externo (Qwen 3.5 800M) FALHA pra MoE+SSM (validado em YouTube 2026-06-19: 17→11 tok/s com 65% aceitação). MTP usa cabeças treinadas no próprio modelo e é mecanismo diferente.
- Ver session log completo: `docs/sessions/2026-06-19-mtp-baseline.md`.


## llama.cpp flags (per Unsloth)
TBD — Unsloth doc section was truncated at the 5k mark in our extraction. The Qwen3.6 doc references a "🦙 Llama.cpp Guide" subsection we did not fully capture. **Re-extract if needed before first run.**

Known good starting points (from Qwen3.5-MTP script that we already use):
- `--spec-type draft-mtp`
- `--cache-type-k q4_0` `--cache-type-v q4_0`
- `--flash-attn on`
- `--threads 8` `--threads-batch 8`
- `--batch-size 512` `--ubatch-size 128`
- `--ctx-size 32768`

## VITRIOL / Split strategy (Codacus technique)
Source: https://www.youtube.com/watch?v=ZwNCsUTNWOA (Qwen 3.6 35B on i5-12th + 16GB RAM + GTX 1070 8GB → 18 tok/s, 132k ctx).

**The core insight:** With MoE, you don't need to put the whole model on the GPU. You put **attention + shared expert + routing** on the GPU, and the **256 routed experts stay in CPU/RAM**. Per-token compute is small (3B active), so the CPU bottleneck is acceptable.

### The 2-knob split (our llama-server turboquant build, verified from `common/arg.cpp`)

1. **`-ngl 99` / `--n-gpu-layers 99`** — load as many attention+shared layers into VRAM as fit. Tells llama.cpp "try GPU for the always-active path."
2. **`-ncmoe 40` / `--n-cpu-moe 40`** — keep the **MoE expert weights of the first N layers in CPU**. For Qwen3.6 35B (40 layers total), `40` = ALL experts stay in CPU/RAM. Lower N = move some experts to GPU, but eats VRAM.

**Plus a 3rd for the draft model (MTP):**
3. **`-ncmoed N` / `--n-cpu-moe-draft N`** — same logic but for the MTP draft model. Draft is small, so probably fits fully on GPU regardless.

### Codacus flag stack (translated to our build)
| Flag (our build) | Codacus name | Meaning |
|---|---|---|
| `--n-gpu-layers 99` | `-ngl 99` | Max attention+shared layers on GPU |
| `--n-cpu-moe 40` | `-n-cpu-moe 40` (he said `-n-cpu-layers-ai`, transcript typo) | All 40 layers' MoE experts on CPU |
| `--cache-type-k q4_0` | `--cache-type-k Q4` | Quantize K cache (4-bit) |
| `--cache-type-v q4_0` | `--cache-type-v Q4` | Quantize V cache (4-bit) |
| `--ctx-size 132000` | `-c 132k` | Push to 132k context (he tested 64k first, then 132k) |
| `--no-mmap` | `--no-mmap` | **NOT used** (he explicitly said he skipped due to 16GB RAM) |
| `--spec-type draft-mtp` | (his video, n/a) | MTP speculative decoding |

### Codacus result: 18 tok/s, 132k ctx, 8GB VRAM + 16GB RAM
- 18 tok/s is **above our 20 tok/s floor only marginally** — but he was on i5-12th + 1070. We have i5/i7 12th+ + RTX 4060. 4060 has 2× CUDA cores of 1070. We should hit 25-35 tok/s easily.

### MTP flag verified from our source
Verified in July 2026: Our current standard llama.cpp binary accepts `--spec-type draft-mtp` (and rejects `mtp`). The alias configurations are configured to use `--spec-type draft-mtp` which successfully enables MTP, achieving ~1.8x speedup.

MTP adds ~2 GB RAM/VRAM headroom and 1.4-2.2× speedup (Unsloth). Unambiguously turn it ON.

## Our config baseline
- `CTX_SIZE`: **65536** (atualizado em commit `78d54e2` — tuning buscou 65k pra 9B-MTP; Validar se valor ótimo para 35B-A3B é o mesmo ou se precisa reduzir)
- `KV_CACHE_K = KV_CACHE_V`: tentar `q4_0` primeiro, depois `turbo2/3/4` se o llama-server suportar [Validar quais TurboQuant types nosso build expõe]
- `SPEC_TYPE = "draft-mtp"` (corrigido — o `draft-mtp` é o flag suportado pelo upstream/standard build)
- `SPEC_DRAFT_N_MAX = 2` (recomendado Unsloth; range 1-6)
- `THREADS = 8` (16/8 hyperthreaded cores)
- `BATCH_SIZE = 512` `UBATCH_SIZE = 128`
- `FLASH_ATTN = "on"`
- `--n-gpu-layers`: Validar (autoloop vai descobrir)
- `--n-cpu-moe`: 40 (= total de layers — mantém todos os experts no CPU/RAM, padrão Codacus) — suporte nativo adicionado em commit `2bd795b`
- Sampling for coding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0`

## TPS medidos (2026-06-19)
- `r_q4` (base GGUF, sem MTP): **11.1 tok/s** (medido Allan)
- `mtp_baseline` (MTP-GGUF, sem flag MTP): **22.5 tok/s** (medido M3, 5 runs × 100 tok, média 22.14/22.36/22.89/22.44/22.84)
- Próximo: `mtp_active` (MTP-GGUF + `--spec-type mtp --spec-draft-n-max 2`): projetado 26-28 tok/s (NÃO executado, aguardando OK Allan)

## Sources / Verification
- HF model card (extracted 2026-06-15)
- Unsloth Qwen3.6 doc (https://unsloth.ai/docs/models/qwen3.6.md, extracted same day, truncated at 5k chars)
- Unsloth MTP guide (https://unsloth.ai/docs/models/mtp.md, extracted same day, truncated at 5k chars)

## Open questions
1. **[Resolvido 2026-06-19]** `llama-server` (turboquant build) NÃO aceita `--spec-type draft-mtp` — o valor é silenciosamente rejeitado. Flag correta é `--spec-type mtp`. Validado por inspeção de `common/arg.cpp` e probe automático em `llama_runner.py:189-205`.
2. **[Resolvido 2026-06-19]** MTP-GGUF tem tensores MTP (`nextn_predict_layers = 1` confirmado). File swap sozinho dá 2× TPS. Validar: ganho adicional com `--spec-type mtp` ativo (próximo teste).
3. Validar: exato valor de `--n-gpu-layers` para nosso 8 GB VRAM + 3B active MoE — autoloop vai descobrir.
4. Validar: ganho com `--no-mmap` e `--mlock` (sweep adicional, ainda não testado).
5. Validar: re-extrair seção completa Unsloth Qwen3.6 llama.cpp pra confirmar comando canônico upstream (vs divergências do turboquant build).
