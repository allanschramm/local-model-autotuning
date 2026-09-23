# Qwen3.8-Flash-Next — Model Card (Candidate)

**Source repo:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next (2026-08-27, **787 525 downloads**, 5 591 likes)
**FP8 variant:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 (2026-08-31, 529 525 downloads)
**License:** Apache-2.0 (per Qwen org convention; verify in card on download)
**Local file:** **not yet downloaded** — full weights are 420 GB BF16 / ≈210 GB FP8 → **too large for the 8 GB rig as-is.** Distilled small variants (e.g. `IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini`) are the path to a usable local copy.
**Family:** Qwen3.8-Flash-Next (Qwen4 experimental architecture)
**Quantization:** BF16 (upstream); FP8 upstream variant; community quants via Unsloth/bartowski expected.

## Architecture (from HF config API, 2026-09-22)

Verified via `hf models info Qwen/Qwen3.8-Flash-Next`:

- `model_type: qwen4_exp` (new — **Qwen4 experimental architecture, not qwen35**)
- `architectures: ["Qwen4ExpForConditionalGeneration"]`
- Pipeline: `image-text-to-text` (multimodal)
- `inference: warm` (already production-served on HF Inference API)
- Files: 131-shard safetensors, largest ≈ 3.36 GB each → **total ≈ 420 GB BF16**

**Chat template reads:**
- `reasoning_effort` (validated values: `xhigh` default / `medium` / `low`; `xhigh` and `high` both render the xhigh instruction; **anything else raises an exception in Jinja**)
- `enable_thinking` (default true; renders empty `<think>` if false)
- `preserve_thinking` (default true; full-history think preservation)
- Native tool-call format: `<|vision_start|><|image_pad|><|vision_end|>` for vision + `<tool_call><function=name><parameter=name>value</parameter></function></tool_call>` for tools

**This is the only verified chat template with a working `reasoning_effort` ladder** outside the operator's existing `Qwen3.8-27B-UD-IQ1_S` card (which uses the same `xhigh/medium/low` ladder, verified 2026-08-29). The `qwen3.8-4b-distill` GGUF and all other Qwen3.5/3.6 lineage cards in this repo ignore `reasoning_effort` (template reads `enable_thinking` only).

**TBD:** `block_count`, `head_count_kv`, attention pattern (full / SWA / Gated DeltaNet / hybrid), `qwen4_exp.*` field specifics. The HF config response was truncated at the `chat_template` field; need `hf models info` re-run or `gguf.GGUFReader` after first download.

## Hardware requirements (estimated, no local measurement yet)

The full BF16 weight is ≈420 GB and the FP8 upstream is ≈210 GB — both exceed the 8 GB rig's VRAM and the typical 32 GB class RAM. **The family is intended for multi-GPU server deployment.** A 27 B-class distill (`IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini`) is the realistic local fit candidate.

Distill fit estimate (matching the operator's existing `Qwen3.8-35B-A3B-Distill` recipe, `models/aliases/qwen3.8-35b-a3b/config.yaml`):

| Quant | Size est | Fit on 8 GB / 32 GB RAM |
|---|---:|---|
| UD-IQ1_S | ~10 GB | tight, may need CPU offload |
| UD-Q3_K_M | ~16 GB | MoE with `--n-cpu-moe 31`, 10 GPU layers (operator's working recipe) |
| UD-Q4_K_M | ~22 GB | same MoE offload recipe |
| UD-Q4_K_XL | ~22 GB | same |

**Verify** exact quants on the IsValorum repo (`hf models ls IsValorum` for the file list).

## Recommended settings (publisher-pending — TBD)

**TBD.** The HF card for `Qwen3.8-Flash-Next` was not pullable via `hf models card` due to UTF-8 codec errors on the harness shell. Verify on first download. The chat template already encodes working defaults:

- `reasoning_effort: xhigh` (default; renders the "think carefully through the task..." instruction)
- `enable_thinking: true` (default; renders `<think>` block)
- `preserve_thinking: true` (default)
- Temperature / TOP_P / TOP_K — extract from the HF card; **mirror the operator's existing `Qwen3.8-27B-UD-IQ1_S` baseline (TEMP 1.0 thinking / 0.7 instruct, TOP_P 0.95, TOP_K 20)** as a first guess

## MTP / speculative

**TBD.** No `nextn` / `mtp` / `draft` field was visible in the truncated HF config response. The `IsValorum/...-MTP-APEX-I-Mini` distilled variant explicitly carries MTP (per its HF tag and download count), so the distill should be MTP-packaged. Verify on first download.

## MoE split (VITRIOL)

**TBD.** Verify `expert_count` / `num_experts_per_tok` on first download. If dense (single-expert) — no offload; if MoE — mirror `Qwen3.8-35B-A3B-Distill` recipe (`models/aliases/qwen3.8-35b-a3b/config.yaml`): `--n-gpu-layers 99 --n-cpu-moe 31` (10 GPU layers active).

## Our config baseline (TBD)

Marked `TBD` until first download + `benchmark_search.py --validation`. Working assumption for the IsValorum distill:

```python
ENGINE_DEFAULTS = {
    'MODEL': 'Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini.gguf',
    'CTX_SIZE': 65536,            # start conservative
    'N_GPU_LAYERS': 99,
    'N_CPU_MOE': 31,              # mirror existing Qwen3.8-35B-A3B distill recipe
    'KV_CACHE_K': 'q4_0',
    'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512, 'UBATCH_SIZE': 128, 'THREADS': 8, 'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on',
    'SPEC_TYPE': 'draft-mtp',     # if MTP tensors present (verify)
    'SPEC_DRAFT_N_MAX': 2,
    'JINJA': True,
    'VRAM_LIMIT_MB': 8000.0,
    'TPS_FLOOR': 15.0,
}
SAMPLER_DEFAULTS = {
    'TEMP': 1.0,                  # thinking mode default
    'TOP_P': 0.95,
    'TOP_K': 20,
    'MIN_P': 0.0,
    'REASONING_EFFORT': 'xhigh',  # only verified ladder that works
}
```

## Why this is worth investigating

1. **Brand-new architecture** (`qwen4_exp`) — first public release of the Qwen4 line. If it ships the same IQ lift as Qwen3→Qwen3.6→Qwen3.8 (each ≈+5 pt on IQ-min in the operator's measurements), it could displace the daily driver.
2. **Verified `reasoning_effort` ladder** — the only candidate in this fit class with a working `xhigh/medium/low` ladder other than `Qwen3.8-27B`. This means `--reasoning-effort medium` actually does something (unlike every `qwen35` lineage card in this repo, where it's a silent no-op).
3. **Multimodal** — image-text-to-text pipeline. Other operator-rig candidates that are text-only (Qwen3.6-35B-A3B, Ornith-1.5-35B) need a separate vision model for image tasks.
4. **Already warm on HF Inference API** — the `inference: warm` field means there's an actively-maintained serving setup. Helps if a future Tool Use / agent path needs a cloud fallback.

## Sources / Verification

- HF info `Qwen/Qwen3.8-Flash-Next` — extracted 2026-09-22 (`qwen4_exp` arch, multimodal, 1 M context class implied by `qwen35.context_length`-style family).
- HF info `Qwen/Qwen3.8-Flash-Next-FP8` — extracted 2026-09-22 (FP8 sibling).
- HF `hf models ls --search "qwen3" --sort last_modified` returned `IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-Mini` (6 046 downloads, 12 likes, 2026-09-22) as the strongest small distill candidate.
- HF `hf models card Qwen/Qwen3.8-Flash-Next` — **failed** (UTF-8 codec error on the harness shell; the README renders fine in a browser). Re-attempt on download.
- Cross-referenced with operator's existing `Qwen3.8-27B-UD-IQ1_S` card (`docs/models/qwen3.8-27b.md`) for the only known working `reasoning_effort` ladder in this lineage.
- Cross-referenced with operator's existing `Qwen3.8-35B-A3B-Distill` recipe (`models/aliases/qwen3.8-35b-a3b/config.yaml`) for the working MoE offload pattern.

## Open questions

- **`qwen4_exp` architecture specifics** — block count, attention pattern, expert count (MoE vs dense). HF config truncated at chat_template; verify via `gguf.GGUFReader` after download.
- **Exact distill basenames** — the `hf models info` command returned "Model not found" on the truncated name from `ls`. Run `hf models ls IsValorum` to get the canonical filename before `hf download`.
- **Reasoning effort ladder behavior** — verified `xhigh/medium/low` in the chat template; need to confirm whether `medium` and `low` measurably reduce wall-time / tokens-per-turn on a real eval (the operator's `Qwen3.8-27B` card claims it works but never measured token reduction).
- **Multimodal `mmproj` sidecar** — image-text-to-text pipeline needs a `mmproj-*.gguf` to run vision inference. Most Qwen3 multimodal checkpoints ship one in the same repo; verify presence.
- **Trial comparison baseline** — once downloaded, the IQ-min comparison should be against `Qwen3.6-35B-A3B-MTP` (operator's empirical daily driver) and `Qwen3.8-4B-Q4_K_M` (Pareto Day #1), not against the lower-IQ distill cards.
