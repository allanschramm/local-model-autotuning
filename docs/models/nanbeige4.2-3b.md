# Nanbeige4.2-3B — Model Card (Local)

**Source (weights):** https://huggingface.co/Nanbeige/Nanbeige4.2-3B  
**Base (metadata):** https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base  
**Community GGUF:** `owao/nanbeige4.2-3b-gguf` → `nanbeige4.2-3b-Q4_K_M.gguf`  
**Arch fork:** https://github.com/Nanbeige/llama.cpp/tree/nanbeige42 → local `llama.cpp-nanbeige42/`  
**License:** Apache-2.0  
**Local file:** `models/owao/nanbeige4.2-3b-gguf/nanbeige4.2-3b-Q4_K_M.gguf` (2.4 GiB)  
**Family:** Nanbeige 4.2  
**Architecture type:** Dense **Looped Transformer** (`num_loops=2`)  
**Quantization:** `Q4_K_M` (`general.file_type=15`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-23):

| Key | Value |
|---|---|
| `general.architecture` | `nanbeige` |
| `nanbeige.block_count` | 22 |
| `nanbeige.num_loops` | **2** (each block applied twice → ~2× FLOPs vs non-looped) |
| `nanbeige.skip_loop_final_norm` | False |
| `nanbeige.embedding_length` | 3072 |
| `nanbeige.feed_forward_length` | 10752 |
| `nanbeige.context_length` | 262144 |
| `nanbeige.attention.head_count` | 48 |
| `nanbeige.attention.head_count_kv` | 8 |
| `nanbeige.attention.head_dim` / key / value | 128 |
| `nanbeige.vocab_size` | 166144 |
| `nanbeige.rope.freq_base` | 70_000_000 |
| MTP / `nextn` keys | **none** |

Upstream `ggml-org/llama.cpp` cannot load this arch. Use fork binaries only.

## Hardware requirements (RTX 4060 8GB)

| Quant | Size | Fit notes |
|---|---|---|
| Q4_K_M | 2.4 GiB | Dense full GPU (`-ngl 99`). No expert/layer offload. |

- Measured idle+ctx path ~6.7–6.9 GB VRAM at `CTX_SIZE=32768`, KV q4_0 (claw run).
- Prefer `CTX_SIZE=32768` on 8 GB with f16 KV; do not spill to shared GPU memory.

## Recommended settings

| Param | Value | Rationale |
|---|---|---|
| TEMP | 1.0 | Agentic / tool-use (Baseline 2026-07-23) |
| TOP_P | 0.95 | GGUF embedded sampling default |
| TOP_K | 20 | GGUF embedded sampling default |
| REPEAT_PENALTY | 1.0 | Off |
| REASONING | on | Model card / agentic path |
| JINJA | true | Chat template |
| CTX_SIZE | 32768 | 8 GB headroom with f16 KV |
| KV_CACHE_K/V | f16 | TPS matrix winner (PP + slight TG) |
| BATCH / UBATCH | 512 / 256 | Best PP/TG balance on 4060 |
| FLASH_ATTN | on | Required for speed |
| N_CPU_MOE | null | Dense — never VITRIOL offload |
| VRAM_LIMIT_MB | 7900 | Physical VRAM floor |

## MTP

No MTP tensors in this GGUF. Speculative decoding not applicable from embedded heads.

## VITRIOL split

N/A — dense model. Do not use `--n-cpu-moe` / partial layer offload (PC freeze risk on Windows shared memory).

## Our config baseline (2026-07-23)

```python
ENGINE_DEFAULTS = {
    'MODEL': 'nanbeige4.2-3b-Q4_K_M.gguf',
    'CTX_SIZE': 32768,
    'KV_CACHE': 'f16',
    'KV_CACHE_K': 'f16',
    'KV_CACHE_V': 'f16',
    'BATCH_SIZE': 512,
    'UBATCH_SIZE': 256,
    'THREADS': 8,
    'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on',
    'SPEC_TYPE': None,
    'SPEC_DRAFT_N_MAX': 0,
    'SPEC_DRAFT_MODEL': None,
    'NO_MMAP': False,
    'JINJA': True,
    'REASONING': 'on',
    'CONT_BATCHING': False,
    'N_CPU_MOE': None,
    'VRAM_LIMIT_MB': 7900,
}
SAMPLER_DEFAULTS = {
    'TEMP': 1.0,
    'TOP_P': 0.95,
    'TOP_K': 20,
    'MIN_P': 0.0,
    'REPEAT_PENALTY': 1.0,
    'PRESENCE_PENALTY': 0.0,
    'FREQUENCY_PENALTY': None,
}
```

Runtime for Trials / `model-up`: set `AUTORESEARCH_LLAMA_CPP_ROOT=llama.cpp-nanbeige42` or alias field `llama_cpp_root: llama.cpp-nanbeige42`.

### Status
- **llama-bench matrix:** tg128 peak **56.00** (f16, b1024/ub512); alias/Baseline **~55 tg** at f16 b512/ub256 with best PP (~2727). Evidence: [session](../sessions/2026-07-23-nanbeige42-tps-matrix.md).
- **Claw-Eval full keep:** **0.2667** (n=15) under prior q4_0 Baseline; see `results.tsv` / same session.

## Sources / Verification

| Claim | Source | Date |
|---|---|---|
| Arch + loops from GGUF | Local `GGUFReader` on `nanbeige4.2-3b-Q4_K_M.gguf` | 2026-07-23 |
| Upstream HF card | https://huggingface.co/Nanbeige/Nanbeige4.2-3B | 2026-07-22 |
| Fork branch | https://github.com/Nanbeige/llama.cpp/tree/nanbeige42 @ `26cfdc4` | 2026-07-22 |
| TPS matrix | [session 2026-07-23](../sessions/2026-07-23-nanbeige42-tps-matrix.md) | 2026-07-23 |

## Open questions

- None blocking local use. Optional later: re-run claw-full under f16 Baseline to refresh Val Score vs 0.2667 (q4_0-era). Superseded as Val Score leader by Laguna-XS **0.6667** (2026-07-24) — see [claw-eval-leaderboard.md](../discovery/claw-eval-leaderboard.md).
