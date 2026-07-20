# Small-model MTP inventory & TPS (8 GB)

Operator contract for dense ~9B / E4B models on an 8 GB CUDA box: which files have MTP, how to enable them, and measured generation TPS.

Evidence: [docs/sessions/2026-07-20-small-model-tps-matrix.md](../sessions/2026-07-20-small-model-tps-matrix.md). Architecture deep-dive: [speculative-decoding-formats.md](./speculative-decoding-formats.md). How to smoke-test: [mtp-baseline-guide.md](./mtp-baseline-guide.md).

---

## 1. MTP packaging (read this first)

MTP is a **training/inference feature**, not a Dense-vs-MoE class.

| Packaging | Tensors | Flags | Typical size |
|---|---|---|---|
| **Embedded** | `nextn` / `blk.*.nextn.*` inside main GGUF | `--spec-type draft-mtp --spec-draft-n-max N` (no `-md`) | same as base (~5–6 GB) |
| **External draft** | separate `gemma4-assistant` GGUF | `--spec-type draft-mtp --spec-draft-model <draft> --spec-draft-n-max N` | draft ~60 MB + base |

**Do not assume** “Gemma has MTP headers in the main UD file.” Main Gemma-4 E4B UD has **no** `nextn`. Pair with `models/draft/mtp-gemma-4-E4B-it.gguf`.

**Detect:** scan GGUF metadata for `nextn` or `gemma4-assistant`. No hit → need Hub MTP quant or a draft file.

---

## 2. Local inventory (this repo)

| Model | Base GGUF | MTP form | Local MTP asset |
|---|---|---|---|
| Gemma-4 E4B | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | external draft | `draft/mtp-gemma-4-E4B-it.gguf` |
| Qwen3.5-9B | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | embedded | same file (`nextn`) |
| Ornith-1.0-9B | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | Hub MTP GGUF | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` |
| Mythos 5-1M | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | Hub MTP GGUF | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` |
| Qwythos-9B-v2 | `Qwythos-9B-v2-Q4_K_M.gguf` | none (CUDA) | — |

### Hub sources used
- Ornith MTP: `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` → `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Mythos MTP: `mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF` → `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf`

```powershell
hf download protoLabsAI/Ornith-1.0-9B-MTP-GGUF Ornith-1.0-9B-MTP-Q4_K_M.gguf --local-dir models
hf download mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf --local-dir models
```

Always use `hf` CLI (repo contract). On Windows, prefer UTF-8 / `--format quiet` if progress glyphs crash the console.

---

## 3. Measured TPS (RTX 4060, upstream CUDA, 2026-07-20)

Method: harness `run_llama_bench_validation` → `llama-cli` `-n 512`, shared knobs `q4_0` KV, batch 256/128, threads 6/8, `draft-mtp` n_max=4 when MTP on. No Claw/coding.

| Model | Base t/s | MTP t/s | Gain |
|---|---:|---:|---:|
| Gemma-4 E4B | 67.6 | **122.0** | **+80%** |
| Qwen3.5-9B | 38.7 | 57.3 | +48% |
| Ornith-1.0-9B | 38.7 | 56.3 | +46% |
| Mythos 5-1M | 40.8 | 41.2 | +1% |
| Qwythos-9B-v2 | 40.1 | — | no CUDA MTP |

**Default Baseline recommendation:** Gemma-4 E4B + draft MTP (`SPEC_DRAFT_N_MAX=4`). Edit via `autoresearch/core/config.py` only.

**Mythos:** MTP GGUF loads but short-gen gain is noise; wall time worse. Prefer non-MTP for speed. Long-ctx agentic may still lose VRAM headroom — see model card.

---

## 4. How to re-run (TPS-only)

1. Set Baseline in `config.py` (`MODEL`, `SPEC_*`, engine knobs) — no CLI soup.
2. Measure with harness cli-bench (not `llama-bench`, not `--validation` if you want to skip Claw):

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\venv\Scripts\python.exe -c @"
from pathlib import Path
from autoresearch.core.config import load_config
from autoresearch.runners.evaluation import run_llama_bench_validation
c = load_config()
print(run_llama_bench_validation(
    model_path=Path('models')/c['MODEL'],
    threads=c['THREADS'], threads_batch=c['THREADS_BATCH'],
    batch_size=c['BATCH_SIZE'], ubatch_size=c['UBATCH_SIZE'],
    flash_attn=c['FLASH_ATTN'], cache_type_k=c['KV_CACHE_K'], cache_type_v=c['KV_CACHE_V'],
    no_mmap=c['NO_MMAP'], spec_type=c.get('SPEC_TYPE'),
    spec_draft_n_max=int(c.get('SPEC_DRAFT_N_MAX') or 0),
    spec_draft_model=c.get('SPEC_DRAFT_MODEL') if c.get('SPEC_DRAFT_N_MAX') else None,
))
"@
```

For Gemma MTP, keep `SPEC_DRAFT_MODEL='draft/mtp-gemma-4-E4B-it.gguf'` (path relative to `models/`). For embedded MTP, set `SPEC_DRAFT_MODEL` unused / omit draft arg.

---

## 5. Hard rules (this rig)

1. Upstream KV only: `q4_0` (etc). Never `turbo*` — not in upstream `llama.cpp`.
2. PrismML binary = Bonsai/DSpark only — not for Gemma/Qwen baselines.
3. One validation/bench at a time.
4. `CTX_SIZE` never changed without explicit user permission; cli TPS gate does not pass `-c` anyway.
5. Filename contains `MTP` ≠ automatic win (Mythos). Always measure.

---

## Open questions

- Mythos MTP: does `SPEC_DRAFT_N_MAX` sweep (1–8) recover acceptance, or are heads ineffective for this quant/prompt?
- Qwythos-9B-v2: watch Hub for a CUDA GGUF MTP; MLX-only does not help this stack.
