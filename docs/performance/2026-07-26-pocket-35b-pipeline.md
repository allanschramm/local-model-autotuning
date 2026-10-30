# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-26-pocket-35b-pipeline.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

First-pass champion path for `POCKET-35B-Q3_K_M.gguf` (FINAL-Bench / Darwin-36B-Opus lineage): validation → Claw-Eval full → coding-10 on the same Baseline Fingerprint.

- discrete 8 GB-class NVIDIA (`VRAM_LIMIT_MB=7900`)
- ~32 GB system RAM
- Windows, upstream `llama.cpp` CUDA (`build-cuda`)

```python
MODEL = 'POCKET-35B-Q3_K_M.gguf'
CTX_SIZE = 65536
KV_CACHE_K = KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # → 40
TEMP = 0.6  # Darwin recommended
TOP_P = 0.95
TOP_K = 20
SPEC_TYPE = None  # nextn_predict_layers=0
```

GGUF: `models/FINAL-Bench/pocket-35b-gguf/POCKET-35B-Q3_K_M.gguf` (~16.8 GB).


| Stage | Result |
| --- | --- |
| Validation | KEEP — , tg **35.2 t/s**, peak VRAM **3.7 GB** |
| Historical run runtime | KEEP — , tg **35.7 t/s**, VRAM **3.7 GB**, ~26 min |
| Historical run runtime | VRAM 3.6 GB |


- ctx 65k × TPS ~35

- Quant pick: **Q3_K_M** (not publisher Q2_K) — GPU + VITRIOL; ~16.8 GB GGUF.
- No TPS autoloop this session — 35 t/s already above floor; quality gates first.
- Card + leaderboards updated.
