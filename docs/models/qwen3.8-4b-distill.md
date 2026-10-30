# Qwen3.8-4B-Distill — Model Card

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/qwen3.8-4b-distill.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF
**Base model:** full-parameter distillation of Qwen3.8-2.4T-A95B into the Qwen3.5-4B architecture (~45k curated teacher traces)
**License:** Apache-2.0
**Local file:** `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf` (2.8 GB; Q5_K_M 3.2 / Q6_K 3.6 / Q8_0 4.6 / BF16 8.7 also in repo)
**Family:** Qwen 3.8 distill on Qwen 3.5 architecture (Alibaba / Empero AI)
**Quantization:** standard Q4_K_M

## Architecture (from GGUF via harness)

- **Dense** (`block_count` = **33**, `expert_count` = 1) — harness `is_moe_model` reads dense ⇒ no `N_CPU_MOE`.
- Hybrid attention: Gated DeltaNet layers with sparse full-attention interval (Qwen3.5 pattern, 8/32 full-attn per publisher card) ⇒ ~¼ dense KV footprint at equal ctx.
- Native context class 262K-equivalent; runs 131072 configured on this rig.

## Hardware requirements (discrete 8 GB-class)

- Publisher tier: Q4_K_M / Q5_K_M → 4–6 GB GPUs; fits entirely (`N_GPU_LAYERS 99`).
- Measured @131072 + `q4_0` KV + flash-attn: peak **5.5 GB**, preflight est 5320 MB vs 7676 limit. Comfortable headroom; no OOM across validation + claw-full + coding-10.
- **KV quant (2026-08-25 A/B + 2026-08-28 fit probe, b10549):** whole-cache rows @65k — q4_0/q4_0 0.7333/0.58, q8_0/q8_0 **0.8000**/0.59, f16/f16 0.7333/**0.6250** (agentic/coding; all on_front). Speed @131k (2026-08-28): q8_0/q8_0 = pp 2767.5 / tg **71.6** / 6527 MB — **zero tg cost** vs q4_0/q4_0 (2764.7 / 71.2 / 5498). **MIXED K/V (K q8_0 + V q4_0) is BROKEN on this arch:** pp ~152-157 / tg ~30 at BOTH 131k and 65k (18× pp / 2.4× tg kernel cliff) with ZERO VRAM saving (5462/4641 MB — GDN-hybrid KV is small). Do not seed mixed K/V fingerprints on GDN-hybrid models; if KV precision is wanted, whole-cache q8_0 is the fast path. Session: [2026-08-28](../performance/2026-08-28-ornith-35b-ubatch-ot-ab.md).

## Recommended settings (publisher card)

- TEMP 0.6 · TOP_P 0.95 · TOP_K 20 · MIN_P 0.0 · REPEAT_PENALTY 1.0
- Chat template embedded; requires `JINJA=True`.

### Reasoning control (verified 2026-08-29 from embedded `tokenizer.chat_template`)

The GGUF template reads **`enable_thinking` ONLY** (the conditional block is `{% if enable_thinking is defined and enable_thinking is false %}` → empty assistant turn; else `</think>\n`). **The Qwen3.8-27B reasoning-effort ladder (xhigh / medium / low) does NOT apply to this GGUF** — `reasoning_effort` is not a variable in this template and `--reasoning-effort` is a silent NO-OP on qwen35-family GGUFs. Working levers: `--reasoning on/off` (Baseline REASONING), `--reasoning-budget N` (REASONING_BUDGET, server-side template-independent), and REASONING_PRESERVE where /props reports `supports_preserve_reasoning` (verified true for Ornith-1.5 GGUFs). **Common misconception guard:** do not attempt `--reasoning-effort medium/low` on this model expecting a ladder change — the template has no such variable. Verified by direct grep of the embedded Jinja (no `reasoning_effort`, `thinking_budget`, or `thinking_level` tokens); compare Qwen3.8-27B-UD-IQ1_S, which reads `reasoning_effort` 8× and defaults to xhigh.

## MTP

- **No MTP tensors in this GGUF** (plain distill; not the `-MTP-GGUF` variant). No embedded draft head; separate draft models fail on this arch family — do not add spec decoding without an explicit A/B.

## MoE split ("VITRIOL split")

N/A — dense model, all layers on GPU.

## Our config baseline (measured 2026-08-23)

```python
ENGINE_DEFAULTS = {
    'MODEL': 'Qwen3.8-4B-Q4_K_M.gguf',
    'CTX_SIZE': 131072,
    'KV_CACHE': 'q4_0', 'KV_CACHE_K': 'q4_0', 'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512, 'UBATCH_SIZE': 128, 'THREADS': 8, 'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on', 'NO_MMAP': False, 'JINJA': True,
}
SAMPLER_DEFAULTS = {'TEMP': 0.6, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.0}
```

Measured Objective Vector (complete, same Fingerprint):

| Axis | Value | Evidence |
| :--- | ---: | :--- |
| bench tg | 74.9 t/s | validation + claw-full rows agree |
| Combined TPS (coding gen) | 94.2 | coding row |

Status `on_front`; current Baseline pick. TSV `6b7e2a3f…` (validation) / `6069530a…` (full).

## Sources / Verification

- HF card `empero-ai/Qwen3.8-4B-Distill-GGUF` — sizes, tiers, mixed eval deltas (mmlu +0.199 / gsm8k_cot −0.065 vs Qwen3.5-4B base), Apache-2.0. Extracted 2026-08-29.
- HF README `empero-ai/Qwen3.8-4B-Distill-GGUF` — benchmark table, hardware guidance, usage notes. Extracted 2026-08-29.
- Base model card `empero-ai/Qwen3.8-4B-Distill` — teacher-trace details, training hyperparams, license inheritance. Extracted 2026-08-29.
- Local measured: `docs/sessions/2026-08-23-qwen38-4b-distill-validation.md`, `2026-08-23-qwen38-4b-distill-full-trial.md`.

## Open questions

- **REASONING_BUDGET interaction:** does `--reasoning-budget N` interact with the GDN hybrid pipeline (linear-attention layers) differently than on dense models? No A/B run yet.
- **TBD:** 262K native vs 131K YaRN practical ceiling on b10549 — verify from effective server log before raising `CTX_SIZE`.
