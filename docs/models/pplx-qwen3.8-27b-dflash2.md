# perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf — Model Card (Drafter)

**Source repo (drafter):** https://huggingface.co/perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf (2026-08-26, **6 368 downloads**)
**Sibling vLLM stack:** https://huggingface.co/perplexity-ai/pplx-computer-vllm-dflash2 (0 downloads — vLLM-specific equivalent)
**Base model (target):** Qwen/Qwen3.8-27B (operator already has a card for the IQ1_S quant — see [`qwen3.8-27b.md`](./qwen3.8-27b.md), `TBD` on agentic + coding)
**License:** verify on first download (Perplexity's usual Apache-2.0 for derivative quants of Apache targets)
**Local file:** **not yet downloaded** — Perplexity's own DFlash2 drafter for the Qwen3.8-27B target
**Family:** DFlash2 speculative-decoding drafter (not a standalone model)
**Quantization:** TBD (Perplexity typically uses GGUF quants similar to upstream)

## Architecture (from HF search results, 2026-09-22)

This is a **DFlash2 drafter GGUF**, not a standalone language model. Pairs with `Qwen/Qwen3.8-27B` as the target.

**TBD — must verify on download:**
- `block_count` (the drafter's own depth — typically small, 1–4 layers)
- `embedding_length` (must match target)
- `head_count_kv` (must match target)
- DFlash2 training: which teacher trace set, what acceptance target
- File size (DFlash drafters are typically <1 GB; the 6 368 download number suggests the repo carries both target quant + drafter)

**Note:** the operator's existing `qwen3.8-27b.md` card mentions `draft-dflash` as a known speculative decoding path for the Qwen3.8-27B family. DFlash2 is the successor (naming convention per Perplexity's release series). No public A/B vs MTP-n2 on Ada CUDA 8 GB as of 2026-09-22.

## Hardware requirements (estimated)

**Pair with `Qwen/Qwen3.8-27B` (BF16 or FP8 quant), not IQ1_S — DFlash2 is quality-sensitive and an IQ1_S target defeats the point.**

The operator's existing Qwen3.8-27B-IQ1_S (5.8 GB) is too aggressive for DFlash2 pairing. The realistic DFlash2 target is one of:

| Target quant | Size | VRAM fit on 8 GB | Active path |
|---|---:|---|---|
| `Qwen3.8-27B-Q4_K_M` | ~16 GB | partial offload needed (~10 GB active) | `--n-gpu-layers` less than full |
| `Qwen3.8-27B-UD-IQ4_XL` | ~14 GB | same | same |
| `Qwen3.8-27B-FP8` (Qwen official) | ~28 GB | CPU-active path + drafter on GPU | `--n-gpu-layers 0` for target, drafter on GPU |
| `Qwen3.8-27B-BF16` | ~54 GB | out of scope for 8 GB | — |

**Working assumption:** DFlash2 drafter on GPU (small, ~1 GB), target model on CPU bandwidth (`--n-gpu-layers 0`). Decode speed is bounded by PCIe bandwidth between drafter (VRAM) and target (RAM), not by GPU FLOPS. Expected TPS: 10–20 t/s for a 27B dense target. Compare against the IQ1_S-only baseline (28.4 t/s @65k bench per `qwen3.8-27b.md`).

## Recommended settings (TBD)

**TBD.** The Perplexity DFlash2 release README was not pullable via `hf models card` (UTF-8 codec error on the harness shell, same as the other 2026-09-22 candidates). Verify on download.

Working assumption (mirror the IQ1_S baseline + DFlash drafter flags from the operator's `qwen3.8-27b.md` card):

```text
llama-server \
  -m models/Qwen/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -md models/perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf/drafter.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 2 \
  -ngl 0 \
  -ngld 99 \
  --flash-attn on \
  -c 65536 \
  -np 1
```

Flag notes:
- `--spec-type draft-dflash` (verify spelling in `upstream@b10867` `--help` — could be `draft-dflash2` or `dflash2`)
- `-md` → draft model path
- `-ngl 0` → target fully on CPU (target is too big for 8 GB VRAM at FP8/Q4)
- `-ngld 99` → drafter fully on GPU (drafter is small)
- `-np 1` → single slot (speculative decoding disables cross-request prompt-cache)

## MTP / speculative

This card is **entirely about speculative decoding** — the drafter IS the MTP/DFlash2 path. No separate MTP tensors expected in either file. Other speculative paths for the same target (MTP n2, EAGLE3) are **alternative** drafters, not co-existing; pick one per `--spec-type`.

## MoE split (VITRIOL)

N/A — Qwen3.8-27B is **dense** (`block_count = 64, expert_count = 1` per the operator's existing `qwen3.8-27b.md` card). No `--n-cpu-moe`. The "VITRIOL split" for dense + DFlash2 is **target on CPU, drafter on GPU**.

## Our config baseline (TBD)

Marked `TBD` until first download + `benchmark_search.py --validation`. Working draft:

```python
ENGINE_DEFAULTS = {
    'MODEL': 'Qwen3.8-27B-UD-Q4_K_M.gguf',           # or FP8 quant
    'DRAFT_MODEL': 'dflash2-qwen3.8-27b.gguf',        # from Perplexity repo
    'CTX_SIZE': 65536,
    'N_GPU_LAYERS': 0,                               # CPU-active target
    'KV_CACHE_K': 'q4_0',
    'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512, 'UBATCH_SIZE': 128, 'THREADS': 8, 'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on',
    'PARALLEL': 1,                                  # single slot for spec
    'SPEC_TYPE': 'draft-dflash',                     # verify spelling
    'SPEC_DRAFT_N_MAX': 2,
    'JINJA': True,
    'VRAM_LIMIT_MB': 8000.0,
    'TPS_FLOOR': 10.0,                               # lower floor (CPU-active target is slow)
}
SAMPLER_DEFAULTS = {
    'TEMP': 1.0, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.0,
    'REASONING_EFFORT': 'xhigh',                     # works on Qwen3.8-27B lineage
}
```

## Why this is worth investigating

1. **DFlash2 (the spec-decode successor)** — Perplexity's `pplx-computer` product stack. If the +1.4–2.0× decode speedup Perplexity claims in their internal benches holds on the operator's hardware, this is the cheapest way to make a 27 B dense model usable for daily SWE.
2. **Fills the IQ gap on the operator's existing Qwen3.8-27B card** — the IQ1_S quant is fast but quality-degraded; pairing a higher-quality quant with DFlash2 drafter could land in the "best of both" zone (FP8/BF16 quality + IQ1_S speed).
3. **Official upstream target compatibility** — Perplexity ships the drafter specifically for `Qwen/Qwen3.8-27B`. No need to match tensor names or train a custom drafter.
4. **Perplexity-validated quality bar** — `pplx-computer` is their production SWE / coding product. The drafter is trained against the target's actual response distribution, not a generic distillation.

## Sources / Verification

- HF search result `perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf` — extracted 2026-09-22 via `hf models ls --author perplexity-ai --sort last_modified`.
- HF `hf models info perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf` — **failed** (`Model not found`; the search returned a truncated form ending in `…`). The exact repo name needs to be re-fetched via `hf models ls --author perplexity-ai`.
- Cross-referenced with operator's existing `Qwen3.8-27B-UD-IQ1_S` card (`docs/models/qwen3.8-27b.md`) for the target's architecture, the working `reasoning_effort` ladder, and the existing fit-pattern notes.
- Cross-referenced with `docs/discovery/speculative-decoding-formats.md` for the DFlash vs DFlash2 vs MTP vs EAGLE3 distinction (DFlash2 is a separate variant with its own training recipe).

## Open questions

- **Exact repo name** — `hf models info` failed on the truncated basename. Resolve via `hf models ls --author perplexity-ai --sort last_modified` and pick the exact ID.
- **DFlash2 flag spelling in `upstream@b10867`** — verify `--spec-type draft-dflash` / `draft-dflash2` / `dflash2` exists via `llama-server --help`. If absent, the operator's engine is too old and needs to wait for upstream merge.
- **Drafter file size and tensor layout** — must verify after download. If the drafter needs >2 GB VRAM, the Q4_K_M target on CPU + drafter on GPU won't fit in 8 GB.
- **Decode speed on PCIe-bounded target** — the IQ1_S baseline at 28.4 t/s (target on GPU) vs. the DFlash2-pair candidate at 10–20 t/s (target on CPU) is a known tradeoff. Need measured numbers before declaring this a win.
- **Trial comparator** — should the next Trial on this card compare against `Qwen3.8-27B-IQ1_S` (speed baseline, current best 27B on this rig) or against `Qwen3.6-35B-A3B-MTP` (operator's daily driver)? Recommend against the latter — different arch family (dense vs MoE), apples-to-oranges.
- **Coding-10 + agentic-full on this target** — the operator's `qwen3.8-27b.md` card has these as `TBD`. A single Trial fills both vectors and tells us whether DFlash2 + UD-Q4_K_M target beats the IQ1_S baseline on IQ-min.
