# Ternary-Bonsai-27B-Q2_0 — Model Card (Local, **not kept on disk**)

**Status:** **REJECTED / deleted from `models/`** (2026-07-23). Doc kept for future re-acquire decisions.
**Source:** PrismML Bonsai family (Q2_0 ternary pack). Related: [bonsai-27b.md](bonsai-27b.md) (Q1_0 keep).
**Runtime:** requires local vendor fork `llama.cpp-prismml/` (CUDA). Upstream `llama.cpp` cannot run this quant.

## Why deleted
Validation on RTX 4060 8 GB (PrismML `build-cuda`):

| ctx | Preflight | bench_tg | Verdict |
|---|---|---|---|
| 65536 | est 8567 MB > 7900 | — | VRAM_PREFLIGHT fail |
| 32768 | est 7850 MB OK | **10.6 t/s** | FAIL vs `TPS_FLOOR` 15 |

Same ~10.6 t/s reject already noted on [bonsai-27b.md](bonsai-27b.md) matrix (2026-07-21 @ 131k). Q1_0 target-only stays ~41 t/s on upstream CUDA — Ternary Q2_0 is not competitive for interactive use on this rig.

## If re-downloading later
- Prefer **Q1_0** (`Bonsai-27B-Q1_0.gguf`) unless PrismML ships a Q2_0 that beats ~15 t/s on 8 GB.
- Flags that loaded (speed still failed floor):
  ```
  AUTORESEARCH_LLAMA_CPP_ROOT=llama.cpp-prismml
  -ngl 99 -c 32768 -fa on -ctk q4_0 -ctv q4_0 --no-mmap --spec-type none
  ```
- Do **not** enable speculative for max TPS on Bonsai Q1_0 (DSpark loses to target-only); Ternary was validated with `SPEC_TYPE=None`.

## Open questions
- None for this rig until a new Ternary build claims ≥15–20 t/s @ ≤8 GB physical VRAM.
