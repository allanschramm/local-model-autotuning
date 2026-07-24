# VITRIOL — MoE-on-small-VRAM technique

- **YouTube Video:** https://www.youtube.com/watch?v=ZwNCsUTNWOA (Codacus on Qwen 3.6 35B-A3B on i5-12th + 16GB RAM + GTX 1070 8GB → 18 tok/s, 132k ctx).
- **GitHub Repository:** https://github.com/Randozart/VITRIOL (Codacus/Randozart official repository).

## Core insight
For MoE models, you don't put the whole model on the GPU. You put **attention + shared expert + routing** on the GPU, and the **256 routed experts stay in CPU/RAM**. Per-token active compute is small (3-4B), so the CPU bottleneck is acceptable.

## The 2-knob split

```
-ncmoe 40            # --n-cpu-moe 40 — ALL 40 layers' MoE experts on CPU
-ngl 99              # --n-gpu-layers 99 — max attention+shared on GPU
-cache-type-k q4_0   # K cache quantized to 4-bit
-cache-type-v q4_0   # V cache quantized to 4-bit
-c 132000            # --ctx-size 132k (Codacus pushed to 132k, tried 64k first)
```

The 2 flags compose:
- `--n-gpu-layers N` decides attention/shared layers on GPU (N=99 = as many as fit)
- `--n-cpu-moe N` decides which layers' MoE experts are forced to CPU

For a 40-layer MoE, `--n-cpu-moe 40` keeps all experts on CPU. Lower N = move some experts to GPU (eats VRAM, may or may not help speed — needs testing).

## Harness default (auto block_count)

Initial Baseline for MoE that does **not** fit full GPU: leave `N_CPU_MOE=None`. `ServerIntent.from_config` reads GGUF `*.block_count` and passes `--n-cpu-moe {block_count}` (same intent as Codacus “all experts on CPU”). Set `N_CPU_MOE=0` only when the MoE fits physical VRAM. Explicit `N>0` overrides. The old `--override-tensor .*exps.*=CPU` path is gone.

## Preflight note (harness)
`estimate_vram_mb` / VRAM preflight **must** pass `n_cpu_moe`. Without it, the estimator charges the full GGUF size to VRAM and falsely rejects 14–20 GB MoE files on 8 GB cards. With `--n-cpu-moe N>0`, weight charge shrinks toward non-expert footprint (~28% of file when N≈32).

## Architecture class
MoE vs dense is decided by **GGUF metadata** (`autoresearch/core/model_arch.py`: `expert_count > 1` or arch name contains `moe`). Filename tokens are not used. Model cards under `docs/models/` must document the same class.

## When does it help vs not?
- Helps: large MoE models (35B+ total, ≤5B active) on ≤16GB VRAM rigs.
- Doesn't help: small dense models, or when you have enough VRAM to fit the full active path.
- Trade-off: lower tok/s than full-GPU, but **infinite context** if you have SSD offload.

## Codacus result vs ours
| | Codacus | Nosso rig |
|---|---|---|
| CPU | i5-12th gen | i5/i7 12th+ (WSL) |
| RAM | 16 GB | 16 GB |
| GPU | GTX 1070 8GB | RTX 4060 8GB |
| Result | 18 tok/s @ 132k ctx | Qwen3.6 35B: **22.15 t/s** @ 65k ctx (`--n-cpu-moe 40`). Gemma-4 26B: **19.00 t/s** @ 65k ctx (`--n-cpu-moe 30`). |
| Floor | nosso `TPS_FLOOR` default é 20 (Baseline `config.py`) | Qwen3.6 35B passa o floor (22.15 t/s). Gemma-4 26B chega muito perto (19.00 t/s) — baixe `TPS_FLOOR` pra MoE se precisar. |

## Related flags (from our llama-server build)
- `--n-cpu-moe-draft N` / `-ncmoed` — same for MTP draft model
- `--no-mmap` — force full model into RAM (Codacus skipped, we have 16GB so would OOM)
- `--spec-type draft-mtp` — MTP speculative decoding (standard/upstream llama.cpp)

## See also
- [Qwen3.6-35B-A3B model card](qwen3.6-35b-a3b.md) — o modelo que Codacus testou (35B total / 3B active)
- [Gemma-4-26B-A4B model card](gemma-4-26b-a4b.md) — mesma técnica, active params um pouco maior (4B), testado no nosso rig (commit `2bd795b`: HumanEval 0.533, TPS 13-18)
