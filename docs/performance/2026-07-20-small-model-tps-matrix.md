# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-20-small-model-tps-matrix.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

**Gemma-4 E4B trap:** main `UD-Q4_K_XL` has **zero** `nextn` keys. MTP is **not** “internal header in the main file”. Architecture supports assistant MTP; weights live in the draft file (~60 MB).

- Hit `nextn` / `blk.*.nextn.*` → embedded MTP.
- Hit `gemma4-assistant` → Gemma assistant draft (external).
- Neither → no MTP in that file (need Hub MTP quant or separate draft).

- `KV_CACHE_K/V = q4_0`
- `BATCH_SIZE = 256`, `UBATCH_SIZE = 128`
- `THREADS = 6`, `THREADS_BATCH = 8`
- `FLASH_ATTN = on`, `NO_MMAP = True`, `JINJA = True`
- MTP on: `SPEC_TYPE = draft-mtp`, `SPEC_DRAFT_N_MAX = 4`
- `CTX_SIZE` left at `131072` (unused by this cli gate; harness does not pass `-c`)


| Family | Mode | Model | TPS (t/s) | Wall (s) | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| gemma-e4b | base | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | **67.6** | 14.7 | — |
| gemma-e4b | MTP draft | same + `draft/mtp-gemma-4-E4B-it.gguf` | **122.0** | 9.5 | **+80%** |
| qwen35-9b | base | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 38.7 | 19.4 | — |
| qwen35-9b | MTP embedded | same + `draft-mtp` n=4 | **57.3** | 14.4 | **+48%** |
| ornith-9b | base | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | 38.7 | 19.4 | — |
| ornith-9b | MTP GGUF | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | **56.3** | 16.0 | **+46%** |
| mythos-9b | base | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 40.8 | 18.4 | — |
| mythos-9b | MTP GGUF | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | 41.2 | 29.3 | **+1%** |
| qwythos-v2 | base | `Qwythos-9B-v2-Q4_K_M.gguf` | 40.1 | 19.2 | — (no CUDA MTP) |
