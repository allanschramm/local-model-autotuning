# DeepSeek-V4.1-Flash-DSpark — Model Card (Candidate)

**Source repo (target):** https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
**GGUF + DSpark packagings:**
- https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF (2026-09-14, 2 983 ↓) — single-file GGUF, llama.cpp native, **MIT**, base_model `deepseek-ai/DeepSeek-V4.1-Flash`, tags `speculative-decoding`, `dspark`
- https://huggingface.co/kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF (2026-09-17, 2 246 ↓) — split-file MXFP4, `arch: deepseek41-dspark`
**Sibling:** https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark (2026-07-04, 1 046 039 ↓) — earlier V4 Flash with DSpark
**License:** MIT (per the JigSawPT GGUF repo card)
**Local file:** **not yet downloaded** — JigSawPT GGUF is the safer first target (standard GGUF, no MXFP4 dependency)
**Family:** DeepSeek V4.1 Flash (DeepseekV41ForCausalLM)
**Quantization:** standard GGUF (JigSawPT); MXFP4 (kernelpool — verify llama.cpp b10867 support before use)

## Architecture (from HF config API, 2026-09-22)

Verified via `hf models info deepseek-ai/DeepSeek-V4.1-Flash`:

- `model_type: deepseek_v41`
- `architectures: ["DeepseekV41ForCausalLM"]`
- Pipeline: `image-text-to-text` (multimodal)
- Quantization on the upstream weights: `fp8`
- License: see repo (Apache-2.0 family for upstream DeepSeek-V4; JigSawPT GGUF is MIT)

Verified via `hf models info JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`:

- `gguf.architecture: dflash`
- `gguf.context_length: 1048576` (1 M tokens)
- `gguf.totalFileSize: 7972959936` (≈ 8 GB single-file)
- `library_name: llama.cpp`
- `chat_template`: supports `thinking` / `enable_thinking`, `reasoning_effort: max`, and a `DSML` tool-call XML format — **note: NOT qwen3-style tool calls**, requires a `tools`-aware client or `--tool-call-parser` flag

**TBD:** `num_experts_per_tok` (active params), `expert_count`, `block_count`, MoE vs dense classification. Must verify on first download via `scripts/model_info.py` or `gguf.GGUFReader`.

## Hardware requirements (estimated from sibling V4-Flash-DSpark)

`deepseek-ai/DeepSeek-V4-Flash-DSpark` (2026-07-04) sibling weights:

- `num_experts_per_tok: 6` (per HF config API)
- `safetensors.total: 165 265 454 782` bytes ≈ 154 GB BF16+FP8 (full upstream)
- Pipeline: `text-generation` (text-only, not multimodal like V4.1)

**Fit estimate for the operator rig (8 GB VRAM / 32 GB RAM):**

- 8 GB GGUF fits comfortably in 32 GB RAM (full CPU offload possible)
- Active expert count is the binding question. The operator's existing `qwen3.6-35b-a3b` (3B-active MoE) sits at 6.1 GB VRAM peak with `--n-cpu-moe 99`. If V4.1-Flash lands in the 3B-active envelope, the same recipe applies.
- If active experts are larger (say 6B), expect ~10 GB active path → **partial layer offload required** (`--n-gpu-layers` less than `-ngl 99`), TPS drops.
- 1 M context class fits the rig only if active path is small enough to leave KV cache room. Likely realistic ctx on 8 GB: 32k–65k q4_0 KV.

## Recommended settings (publisher-pending — TBD)

**TBD.** The HF card for `DeepSeek-V4.1-Flash` is API-only at extraction time; the README was not pullable via `hf models card` due to UTF-8 codec errors on the harness shell. Verify on first download from the JigSawPT GGUF repo README.

Initial guess for `SAMPLER_DEFAULTS` (mirror the DeepSeek-V3 sibling chat template if no `enable_thinking` ladder is documented):

```python
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
REPEAT_PENALTY = 1.0
PRESENCE_PENALTY = 0.0
```

If a `reasoning_effort` ladder or `thinking` toggle is documented in the JigSawPT README, override per `docs/discovery/reasoning-levels-mapping.md`.

## MTP / speculative

DSpark is a built-in draft model (the GGUF tag is `speculative-decoding, dspark`). On llama.cpp, this surfaces as a built-in `--spec-type dflash` (verify in `upstream@b10867` `--help`). **No separate draft file needed.**

**TBD:** confirm `--spec-type dflash` (or whatever upstream calls it) exists in the operator's pinned build before adding to the alias.

## MoE split (VITRIOL)

Pending `num_experts_per_tok` confirmation. **Working assumption** (mirror `qwen3.6-35b-a3b`):

```text
--n-gpu-layers 99 --n-cpu-moe 99   # full expert offload if MoE
```

If the active path is too large for 8 GB VRAM, drop to:

```text
--n-gpu-layers 99 --n-cpu-moe <block_count>   # partial offload
```

…or push the dense path entirely onto CPU (`--n-gpu-layers 0` for ngl=0) and accept decode on CPU bandwidth.

## Our config baseline (TBD)

Marked `TBD` until first download + `benchmark_search.py --validation`. Once validated, populate:

```python
ENGINE_DEFAULTS = {
    'MODEL': 'DeepSeek-V4.1-Flash-DSpark.gguf',
    'CTX_SIZE': 65536,           # start conservative; expand if KV budget allows
    'N_GPU_LAYERS': 99,
    'N_CPU_MOE': None,           # → block_count (auto)
    'KV_CACHE_K': 'q4_0',
    'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512, 'UBATCH_SIZE': 128, 'THREADS': 8, 'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on',
    'SPEC_TYPE': 'dflash',       # verify upstream b10867 flag name
    'SPEC_DRAFT_N_MAX': 2,
    'JINJA': True,
    'VRAM_LIMIT_MB': 8000.0,
    'TPS_FLOOR': 15.0,
    'CTX_FLOOR': None,
}
SAMPLER_DEFAULTS = {'TEMP': 0.6, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.0}
```

## Why this is worth investigating

1. **Newest of the MoE 35B-class family with built-in draft** (2026-09-14 JigSawPT release, sibling V4-Flash-DSpark 2026-07-04 has 1 M downloads).
2. **MIT license** — operator can redistribute / ship / post-process without Apache-style obligations.
3. **1 M-token context** — the only candidate in this fit class that advertises a 1 M window (others: Qwen3.6-35B-A3B at 262k, Ornith-1.5-35B at 262k, Qwen3.8-27B at 262k).
4. **DSpark draft** — Perplexity's and others' DFlash variants are 2026-Q3 inventions; this is the canonical upstream DeepSeek implementation, more likely to receive upstream llama.cpp attention than third-party forks.
5. **Coding-first release** (pipeline `text-generation` on V4-Flash-DSpark; `image-text-to-text` on V4.1-Flash with vision capability).

## Sources / Verification

- HF info `deepseek-ai/DeepSeek-V4.1-Flash` — extracted 2026-09-22 (`hf models info`).
- HF info `deepseek-ai/DeepSeek-V4-Flash-DSpark` — extracted 2026-09-22 (active-expert count, sibling size).
- HF info `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF` — extracted 2026-09-22 (`arch: dflash`, 1 M ctx, MIT, 8 GB single-file).
- HF info `kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF` — extracted 2026-09-22 (MXFP4 split, 14 GB, `arch: deepseek41-dspark`).
- HF `hf models card deepseek-ai/DeepSeek-V4.1-Flash` — **failed** (UTF-8 codec error on the harness shell; the README renders fine in a browser). Re-attempt on download.
- Cross-checked against the operator's existing `qwen3.6-35b-a3b` MoE recipe (`models/aliases/qwen3.6-35b-a3b-mtp/config.yaml`) for `--n-cpu-moe 99` fit pattern.

## Open questions

- **Active param count** — must verify `num_experts_per_tok` on first download. If 3 B (like Qwen3.6-35B-A3B), the operator's existing recipe applies. If 6 B+, partial offload or `--n-gpu-layers 0` (CPU-only).
- **DSpark flag name in llama.cpp b10867** — verify `--spec-type dflash` or whatever the upstream spelling is before writing the alias.
- **MXFP4 in b10867** — `kernelpool` variant uses block-fp4. The operator's pinned engine may not support it. JigSawPT single-file GGUF (standard GGUF quant) is the safer first download.
- **Chat template** — the JigSawPT GGUF embeds a DSML tool-call format. The operator's existing `pi-coding-agent` / claude-code clients likely expect qwen3-style tool calls; may need a custom `--tool-call-parser` shim.
- **Coding + agentic vectors** — no `benchmark_search.py` Trial yet. Need `benchmark_search.py --validation` (5/5 claw-quick) → `--agentic-full` + coding-10 → ranked in `results.db` for the leaderboard.
- **TBD: Sweep ctx >65k with DSpark** — the V4 sibling MTP at 131k rejects at 7.92 GB on 8 GB (`ornith-1.5-9b` precedent). V4.1-Flash DSpark at 131k may behave similarly; verify VRAM preflight before raising `CTX_SIZE`.
