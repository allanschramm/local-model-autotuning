# Session Log: Small-model TPS matrix + MTP inventory (2026-07-20)

## Goal
1. Inventory which local small models have MTP (embedded `nextn` vs external draft vs Hub-only).
2. Download missing CUDA GGUF MTP quants via `hf`.
3. Measure pure generation TPS (`llama-cli`, `-n 512`) base vs MTP. No Claw-Eval, no coding bench.

## Hardware
- GPU: RTX 4060 (8 GB VRAM)
- OS: Windows 11 / PowerShell
- Runtime: upstream `ggml-org/llama.cpp` CUDA build in-repo (`llama.cpp/build-cuda/bin/llama-cli.exe`)
- **Not used:** PrismML fork (Bonsai-only), TurboQuant fork (removed; upstream has no `turbo*` KV types)

## Related sessions / ADRs
- [2026-07-20-llama-cli-validation.md](2026-07-20-llama-cli-validation.md) — harness moved from `llama-bench` → `llama-cli` for MTP-aware TPS; KV key mapping bug.
- [ADR 0005](../adr/0005-config-py-mutable-baseline.md) — `config.py` is mutable Baseline; state file = visited only.
- Discovery contracts: [speculative-decoding-formats.md](../discovery/speculative-decoding-formats.md), [mtp-baseline-guide.md](../discovery/mtp-baseline-guide.md), [small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md).

---

## Learnings (operator gold)

### L1 — Two MTP packaging forms (do not confuse)
| Form | Where tensors live | How to enable | Local examples |
|---|---|---|---|
| **Embedded `nextn`** | Inside main GGUF (`blk.N.nextn.*`, `*.nextn_predict_layers`) | `--spec-type draft-mtp --spec-draft-n-max N` **without** `--spec-draft-model` | `Qwen3.5-9B-UD-Q4_K_XL.gguf`, Ornith/Mythos `*-MTP*.gguf` |
| **External draft (assistant)** | Separate small GGUF (`gemma4-assistant.*`) | `--spec-type draft-mtp --spec-draft-model draft/... --spec-draft-n-max N` | `draft/mtp-gemma-4-E4B-it.gguf` paired with Gemma main |

**Gemma-4 E4B trap:** main `UD-Q4_K_XL` has **zero** `nextn` keys. MTP is **not** “internal header in the main file”. Architecture supports assistant MTP; weights live in the draft file (~60 MB).

### L2 — How to detect MTP in a GGUF (fast)
Scan ASCII metadata (first ~8–32 MB is enough for keys):

- Hit `nextn` / `blk.*.nextn.*` → embedded MTP.
- Hit `gemma4-assistant` → Gemma assistant draft (external).
- Neither → no MTP in that file (need Hub MTP quant or separate draft).

### L3 — Hub inventory (models that lacked local MTP)
| Local base | Local MTP? | Hub CUDA GGUF MTP | Chosen download |
|---|---|---|---|
| Ornith-1.0-9B UD | no | **yes** — `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` (~5.4 GB) |
| Mythos 5-1M | no | **yes** — `mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF` | `...-MTP.Q4_K_M.gguf` (~5.4 GB) |
| Qwythos-9B-v2 | no | **no** useful CUDA GGUF (MLX-only MTP) | skip |
| Gemma E4B | draft already local | many assistant MTP GGUFs | reuse `draft/mtp-gemma-4-E4B-it.gguf` |
| Qwen3.5-9B | embedded already | n/a | reuse local UD |

Download rule: `hf download <repo> <file> --local-dir models/<publisher>/<model>` (nested for LM Studio). Config/harness use basename; `resolve_model_path` finds the file. Drafts: `--local-dir models/draft`. UTF-8 / quiet if Windows charmap breaks on ✓.

### L4 — Upstream has no TurboQuant KV types
- Upstream `llama.cpp` allows KV: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.
- `turbo2/3/4` → `Unsupported cache type`. Those existed only in removed TurboQuant fork.
- Overnight “turbo” Search gains were **fake**: `ServerIntent` read `kv_k`/`kv_v` but autoloop wrote `KV_CACHE_K`/`KV_CACHE_V` → always fell back to `q4_0` while TSV labeled turbo. Fixed in same-day harness work; Baseline must use real `q4_0`.

### L5 — Measure MTP with harness `llama-cli`, not `llama-bench`
- `llama-bench` does **not** take `--spec-draft-model` / MTP flags → under-reports speculative TPS.
- Gate: `run_llama_bench_validation` in `autoresearch/runners/evaluation.py` (`-n 512`, quantum tutorial prompt, `--single-turn`).
- Contract: edit `config.py` Baseline first (`write_baseline`); do not drive Trials with CLI flag soup.
- `--validation` also forces Claw quick — for TPS-only, call the cli-bench path (or run with `--no-agentic-quick --no-agentic-full --no-coding`).

### L6 — Fair matrix knobs (this session)
Shared across all trials:

- `KV_CACHE_K/V = q4_0`
- `BATCH_SIZE = 256`, `UBATCH_SIZE = 128`
- `THREADS = 6`, `THREADS_BATCH = 8`
- `FLASH_ATTN = on`, `NO_MMAP = True`, `JINJA = True`
- MTP on: `SPEC_TYPE = draft-mtp`, `SPEC_DRAFT_N_MAX = 4`
- `CTX_SIZE` left at `131072` (unused by this cli gate; harness does not pass `-c`)

---

## Results (measured)

| Family | Mode | Model | TPS (t/s) | Wall (s) | Delta |
|---|---|---|---:|---:|---:|
| gemma-e4b | base | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | **67.6** | 14.7 | — |
| gemma-e4b | MTP draft | same + `draft/mtp-gemma-4-E4B-it.gguf` | **122.0** | 9.5 | **+80%** |
| qwen35-9b | base | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 38.7 | 19.4 | — |
| qwen35-9b | MTP embedded | same + `draft-mtp` n=4 | **57.3** | 14.4 | **+48%** |
| ornith-9b | base | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | 38.7 | 19.4 | — |
| ornith-9b | MTP GGUF | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | **56.3** | 16.0 | **+46%** |
| mythos-9b | base | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 40.8 | 18.4 | — |
| mythos-9b | MTP GGUF | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | 41.2 | 29.3 | **+1%** |
| qwythos-v2 | base | `Qwythos-9B-v2-Q4_K_M.gguf` | 40.1 | 19.2 | — (no CUDA MTP) |

### Ranking (absolute TPS)
1. Gemma E4B + draft MTP — **122.0**
2. Gemma E4B base — 67.6
3. Qwen3.5 MTP — 57.3
4. Ornith MTP — 56.3
5. Mythos base / MTP / Qwythos v2 — ~40–41

### Interpretation
- **Gemma draft MTP** = best absolute and best relative gain on this rig for short/sustained `-n 512` gen.
- **Qwen embedded** and **Ornith MTP GGUF** both ~+45–50% — worth enabling for speed.
- **Mythos MTP GGUF** ≈ no gain and **worse wall clock** (29.3s vs 18.4s). Treat as “file has `nextn`, but acceptance/overhead not worth it” until `n_max` sweep or stderr acceptance stats prove otherwise. Older card claim (MTP hurts long-ctx VRAM) still stands for agentic @131k.
- **Qwythos-9B-v2**: base-only until a CUDA GGUF MTP appears on Hub.

## Errors
None on the 9 TPS trials. All `OK`.

## Decisions
- Left `config.py` Baseline on winner: Gemma-E4B + `draft-mtp` + `SPEC_DRAFT_N_MAX=4` + draft path `draft/mtp-gemma-4-E4B-it.gguf`.
- Durable operator guide: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md).
- Model cards updated with matrix numbers + MTP packaging truth.
- Prefer Ornith MTP GGUF over UD when speed matters; keep Mythos on non-MTP for now.

## Cross-links
- Operator guide: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md)
- Formats: [docs/discovery/speculative-decoding-formats.md](../discovery/speculative-decoding-formats.md)
- Smoke how-to: [docs/discovery/mtp-baseline-guide.md](../discovery/mtp-baseline-guide.md)
- Model cards: [gemma-4-e4b](../models/gemma-4-e4b.md), [qwen3.5-9b](../models/qwen3.5-9b.md), [ornith-1.0-9b](../models/ornith-1.0-9b.md), [mythos](../models/qwythos-9b-claude-mythos-5-1m.md)
