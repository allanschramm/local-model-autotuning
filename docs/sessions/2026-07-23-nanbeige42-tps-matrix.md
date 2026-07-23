# Session Log: Nanbeige4.2-3B fork + TPS matrix (2026-07-22/23)

## Goal
1. Validate `Nanbeige/Nanbeige4.2-3B` for local inference (Looped Transformer).
2. Build arch fork `Nanbeige/llama.cpp` branch `nanbeige42` (upstream cannot load `NanbeigeForCausalLM`).
3. Measure KV / batch TPS matrix on RTX 4060 8GB; lock Baseline + `model-up` alias.

## Hardware
- GPU: RTX 4060 (8 GB VRAM, CUDA 8.9)
- OS: Windows 11 / PowerShell
- Runtime: `llama.cpp-nanbeige42` CUDA build @ `26cfdc4` (`build-cuda/bin/`)
- Model: `models/owao/nanbeige4.2-3b-gguf/nanbeige4.2-3b-Q4_K_M.gguf` (2.4 GiB, from community GGUF)

## Related
- Model card: [nanbeige4.2-3b.md](../models/nanbeige4.2-3b.md)
- Alias: `models/aliases/nanbeige4.2-3b/config.yaml` (`llama_cpp_root: llama.cpp-nanbeige42`)
- Toolset: [llamacpp-toolset.md](../llamacpp-toolset.md) (local arch forks)

---

## Setup

```bash
# Fork (regular directory, not submodule)
git clone --branch nanbeige42 --single-branch https://github.com/Nanbeige/llama.cpp.git llama.cpp-nanbeige42
# Build CUDA Release into llama.cpp-nanbeige42/build-cuda (vcvars64 + CUDA 13.3 + Ninja)

hf download owao/nanbeige4.2-3b-gguf nanbeige4.2-3b-Q4_K_M.gguf \
  --local-dir models/owao/nanbeige4.2-3b-gguf
```

Windows claw-eval mock fixtures: set `PYTHONUTF8=1` (do **not** patch local `claw-eval/` vendor tree).

---

## Commands (reproducible)

Shared: `ngl=99`, `fa=on`, `-p 512 -n 128,512 -r 3`, dense (no CPU MoE).

```bash
llama.cpp-nanbeige42/build-cuda/bin/llama-bench.exe \
  -m models/owao/nanbeige4.2-3b-gguf/nanbeige4.2-3b-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk <KV> -ctv <KV> -b <B> -ub <UB> \
  -p 512 -n 128,512 -r 3
```

---

## Findings (measured)

| KV | b | ub | pp512 (t/s) | tg128 (t/s) | tg512 (t/s) |
|---|---:|---:|---:|---:|---:|
| q4_0 | 512 | 128 | 2410 ± 20 | 54.72 ± 0.47 | 54.55 ± 0.19 |
| q8_0 | 512 | 128 | 2395 ± 11 | 54.75 ± 0.23 | 54.53 ± 0.20 |
| **f16** | **512** | **128** | **2512 ± 43** | **55.92 ± 0.17** | **55.19 ± 0.31** |
| q8_0 | 256 | 128 | 2298 ± 47 | 55.06 ± 0.08 | 54.64 ± 0.06 |
| f16 | 256 | 128 | 2471 ± 16 | 55.82 ± 0.38 | 55.35 ± 0.22 |
| **f16** | **512** | **256** | **2727 ± 78** | **55.17 ± 0.34** | **55.17 ± 0.13** |
| f16 | 1024 | 512 | 2673 ± 28 | **56.00 ± 0.05** | 55.16 ± 0.32 |
| q4_0 | 256 | 64 | 1872 ± 17 | 55.00 ± 0.09 | 54.47 ± 0.12 |

### Learnings
- **KV type barely moves TG** (~54.7 → ~56). Ceiling is looped-arch compute (`num_loops=2` ≈ 2× FLOPs vs same-size dense).
- **f16 KV wins PP** and edges TG; preferred over q4/q8 for this model on 8 GB.
- **Alias sweet spot:** `f16` + `b512/ub256` (best PP/TG balance). Peak tg128 was `f16 b1024/ub512` at 56.00 — not worth the PP trade for daily use.
- Dense: no shared-memory / layer offload. `VRAM_LIMIT_MB=7900`, `CTX_SIZE=32768`.

### Claw-Eval full (same session window)
- `results.tsv` keep: **agentic_full=0.2667** (n=15), TPS gate ~32.4 under then-Baseline `kv=q4_0 b512/ub128`, VRAM ~6.9 GB.
- First attempt failed mock readiness (`:9113`) until `PYTHONUTF8=1`.

---

## Side findings (alias / champion audit — ephemeral scripts, not kept)

- Best dense **agentic_full keep** in `results.tsv`: `Qwythos-9B-v2-MTP-Q4_K_M.gguf` **0.5333** @ 34.5 t/s — **GGUF missing on disk**; Hub has no useful CUDA MTP GGUF for v2 (MLX-only). Local non-MTP `Qwythos-9B-v2-Q4_K_M.gguf` present.
- Several `models/aliases/*` entries point at GGUFs no longer on disk; INDEX lists duplicates / stale scores. Cleanup deferred (promote champion blocked on re-acquire).

---

## Errors
- PowerShell treats `ggml_cuda_init` stderr as `NativeCommandError` when capturing `llama-bench` — ignore; tables still valid.
- Do not edit `claw-eval/` for UTF-8; use env instead.
- Never treat `.scratch/*.py` ad-hoc rank scripts as durable docs — promote numbers here / model cards only.

---

## Decisions
1. Keep fork at `llama.cpp-nanbeige42/` (not submodule).
2. Baseline + alias: KV **f16**, batch **512/256**, ctx **32768**, reasoning **on**, temp **1.0** (agentic).
3. `model-up` gained optional `llama_cpp_root` so aliases can pin arch forks.
4. Session evidence lives here; delete ad-hoc `.scratch/nanbeige-tps-matrix.md` after promote.
