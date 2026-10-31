# Ling-3.0-tiny — Model Card

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/ling-3.0-tiny.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF
**Base model:** inclusionAI Ling-3.0-Tiny (MoE)
**License:** MIT (per HF card)
**Local file:** `models/bartowski/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf` (4.9 GB; Q5_K_M 5.7 / IQ4_XS 4.4 also in repo; Flash variant is 30G+ — does not fit)
**Family:** Ling 3.0 (inclusionAI)
**Quantization:** standard Q4_K_M

## Architecture (from GGUF via harness)

- **MoE**: `block_count` = **24**, experts offloaded via `N_CPU_MOE` = **24 (auto from block_count)**.
- Dense layers full GPU (`N_GPU_LAYERS 99`), expert streaming on CPU.

## Hardware requirements (discrete 8 GB-class)

- Fits with large headroom: measured peak **2.5 GB** @65536 + `q4_0` KV + flash-attn — lowest of any model trialed on this rig. Host preflight est 5063 MB.
- Day-eligible TPS (≥50 floor) at this config.
## Recommended settings

No publisher sampler split on the bartowski card → seeded from `UNIVERSAL_FALLBACK_SAMPLER` then matched to the Qwen-family profile used across the sweep: TEMP 0.6 · TOP_P 0.95 · TOP_K 20 · MIN_P 0.0. `JINJA=True`.
## MTP

- No MTP tensors in this GGUF. No spec decoding configured.
## MoE split ("VITRIOL split")

`--n-cpu-moe 24` auto-resolved from `block_count`. Works clean on b10549.
## Our config baseline (measured 2026-08-23)

Same shape as the Qwen distill baseline but `MODEL='Ling-3.0-tiny-Q4_K_M.gguf'`, `CTX_SIZE=65536`, `N_CPU_MOE=None` (auto).

Measured Objective Vector (complete, same Fingerprint):

| Axis | Value | Evidence |
| :--- | ---: | :--- |
| bench tg | 52.8 t/s | Day-eligible |
| Combined TPS (coding gen) | 62.0 | coding row |

Useful when co-running models or under tighter keepout. TSV `ed7b6627…` (validation) / `e2a298fb…` (full).

## Sources / Verification

- HF `bartowski/Ling-3.0-tiny-GGUF` — quant list via `hf --dry-run`, extracted 2026-08-29.
- Local measured: `docs/sessions/2026-08-29-ling-tiny-validation.md`, `2026-08-29-ling-tiny-full-trial.md`.
## Open questions

- **TBD:** 131072 ctx with turbo2 KV — VRAM headroom exists (2.5/7.7G used); would coding hold while ctx doubles?
