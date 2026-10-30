# Small-model MTP inventory & TPS (8 GB)

Operator contract for dense ~9B / E4B models on an 8 GB CUDA box: which files have MTP, how to enable them, and measured generation TPS.

Evidence: [docs/sessions/2026-07-20-small-model-tps-matrix.md](../performance/2026-07-20-small-model-tps-matrix.md). Architecture deep-dive: [speculative-decoding-formats.md](./speculative-decoding-formats.md). How to smoke-test: [mtp-baseline-guide.md](./mtp-baseline-guide.md).

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
| Ornith-1.5-9B | `Ornith-1.5-9B-Q4_K_M.gguf` | **none** — `qwen35`, `gguf_has_mtp()==false` (0 `nextn` / 427 total, no `nextn_predict_layers`; 2026-08-22 `GGUFReader`) | — |
| Ornith-1.5-35B-A3B | `Ornith-1.5-35B-Q4_K_M.gguf` | **embedded** — `qwen35moe.nextn_predict_layers=1`, 4 `nextn` at `blk.40.nextn.*` / 753 total (2026-08-22 `GGUFReader`, `model_arch.py:126`) | same file (`nextn`) |
| Mythos 5-1M | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | Hub MTP GGUF | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` |
| Qwythos-9B-v2 | `Qwythos-9B-v2-Q4_K_M.gguf` | none (CUDA) | — |
### Hub sources used
- Ornith MTP: `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` → `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Mythos MTP: `mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF` → `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf`

```powershell
hf download protoLabsAI/Ornith-1.0-9B-MTP-GGUF Ornith-1.0-9B-MTP-Q4_K_M.gguf --local-dir models/local/Ornith-1.0-9B-MTP-Q4_K_M
hf download mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf --local-dir models/local/Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M
```

Always use `hf` CLI (repo contract). On Windows, prefer UTF-8 / `--format quiet` if progress glyphs crash the console.

---

## 3. Measured TPS (8 GB-class discrete NVIDIA, upstream CUDA, 2026-07-20)

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
2. Run a finite plan using [the current tuning protocol](../../autoresearch/TUNING.md). Compare TPS and PPL at the same configured context and Fingerprint. Historical CLI throughput above is a different measurement.

For Gemma MTP, keep `SPEC_DRAFT_MODEL='draft/mtp-gemma-4-E4B-it.gguf'` (path relative to `models/`). For embedded MTP, set `SPEC_DRAFT_MODEL` unused / omit draft arg.

---

## 5. Hard rules (the operator host)

1. Upstream KV only: `q4_0` (etc). Never `turbo*` — not in upstream `llama.cpp`.
2. External PrismML binary = Bonsai/DSpark only — not vendored here and not for Gemma/Qwen baselines.
3. One validation/bench at a time.
4. `CTX_SIZE` never changed without explicit user permission; cli TPS gate does not pass `-c` anyway.
5. Filename contains `MTP` ≠ automatic win (Mythos). Always measure.

---

## Open questions

- Mythos MTP: does `SPEC_DRAFT_N_MAX` sweep (1–8) recover acceptance, or are heads ineffective for this quant/prompt? Research prior (2026-08-18): draft length trades acceptance vs tokens/pass — Qwen3.6-27B q8_0 MTP n=3 accept 0.72 @ 21.6 t/s vs n=2 accept 0.83 @ 17.4 t/s on code (upstream [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)); acceptance <~0.6 makes MTP a net loss and aggressively quantized MTP heads are a known acceptance killer (dev.to, 2026-05-18). Mythos' ~+1% fits the "head damaged by quant" pattern — the sweep is still the falsifiable test. See [speculative-decoding-formats.md](./speculative-decoding-formats.md) §3A.
- Qwythos-9B-v2: watch Hub for a CUDA GGUF MTP; MLX-only does not help this stack.
