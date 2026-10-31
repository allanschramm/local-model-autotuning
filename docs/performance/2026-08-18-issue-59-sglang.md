# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-18-issue-59-sglang.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Validate engine differences (TPS / agentic) on one model across llama.cpp and SGLang on the 8 GB-class rig (issue #59). Acceptance: ≥1 `results.tsv` row with `backend=sglang` + TPS vs a same-family llama.cpp baseline.

discrete_gpu 8 GB-class (Baseline `VRAM_LIMIT_MB` 7900, clamped 7676 = physical − 512 keepout), 32 GB-class RAM, Windows host. SGLang runs in WSL2 (Ubuntu 24.04, 24 GB RAM cap) — SGLang has no Windows build.

- `venv-sglang/` in WSL2: pip `sglang` 0.5.17 + `ninja` + pip `cuda-toolkit` (`nvidia/cu13`), with unversioned `libcudart.so` symlinks added; `venv-wsl/` harness env (repo requirements).
- Model: `empero-ai/Qwen3.8-2B` safetensors (fp16, single 4.55 GB shard) → `models/sglang/Qwen3.8-2B/`.
- Baseline seeded in gitignored `autoresearch/core/config.py`: MODEL `sglang/Qwen3.8-2B`, CTX 32768, sampler TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.05 (matches the `Qwen3.8-2B-BF16.gguf` llama.cpp rows).

```bash
# from WSL2, repo root
  --desc 'SGLang cross-engine validation (issue #59): Qwen3.8-2B fp16 vs llama.cpp Qwen3.8-2B-BF16.gguf @32k'
```

Restore the Baseline afterwards (config.py-only flow).


| Axis | SGLang (fp16 safetensors) | llama.cpp baseline (GGUF) |
| :-- | --: | --: |
| Model | `Qwen3.8-2B` | `Qwen3.8-2B-BF16.gguf` |
| Decode TPS (bench 512/512, batch 1) | **56.8** | 60.5 |
| ctx | 32768 | 32768 |


- Rows: `results.tsv` `backend=sglang` (validation; bench 56.8–58.2 across runs).
- Fit: 9B fp16 (18.5 GB) streams at 1–5 t/s (below `TPS_FLOOR` 20); 27B fp16 (≈54 GB) cannot fit the 24 GB WSL RAM cap; no AWQ/GPTQ pack exists for `Qwen3.8-9B` on HF (checked 2026-08-18). 2B is the only same-family fit with a meaningful engine TPS.
- Required SGLang 0.5.17 flags (harness `autoresearch/core/sglang_runner.py` now encodes these): explicit `--mem-fraction-static 0.9` (auto-computed pool rejects the hybrid linear-attention arch), `--attention-backend triton --sampling-backend pytorch` (FlashInfer JIT fails with the pip cu13 toolchain), `--dtype bfloat16` (fp16 trips a state-cache dtype bug), `--mm-feature-transport=cpu` (CUDA IPC multimodal pool crashes on first request), JIT toolchain env (venv nvcc + ninja + unversioned cudart). `--cpu-offload-gb` triggers a layernorm loader device-mismatch on this arch — do not pass.
- Agentic confound: the Qwen3.5 template renders thinking OFF under SGLang without request-level `chat_template_kwargs={"enable_thinking": True}`; the checkpoint then emits placeholder text and no `<tool_call>` (verified: with thinking on it emits valid `<tool_call>` blocks). llama.cpp's template defaults thinking ON — explains 0.8 vs 0.4. Not re-run through the harness (shared-runner change risk; documented in `docs/discovery/sglang-inference-engine.md` §9.1).
- Engine verdict: llama.cpp stays the baseline engine; SGLang ≈ same bandwidth-bound TPS — no win on the 8 GB-class host.

- SGLang 0.5.17 startup chain: auto memory-pool rejects hybrid arch → explicit 0.9; `nvcc` not found → venv cuda-toolkit + PATH/CUDA_HOME injection; `ninja` missing → venv install; FlashInfer JIT nvcc-vs-cccl header mismatch → Triton attention backend; `--cpu-offload-gb` layernorm `torch.add` cuda/cpu mismatch → flag dropped; first-request crash (`CUDA error: invalid resource handle` in the multimodal IPC pool) → `--mm-feature-transport=cpu`; fp16/bf16 state dtype bug → `--dtype bfloat16`.
- Bench parser: SGLang 0.5.17 prints `Decode. median latency … median throughput: N token/s` (not the legacy `Decode token/s:`) → parser updated, test added.
- **Store incident:** a `Remove-Item -Recurse -Force` on the external store path followed the `models/` junction and wiped the store (private notes + GGUF cache). Notes recovered via operator-side undelete; GGUF cache re-downloadable (list in `models/TASK.md`). Corrective actions below.

- SGLang backend stays supported for cross-engine rows (directory model path → `SGLangServerRunner`); llama.cpp remains the baseline engine.
- 9B/27B not viable for SGLang on this rig — documented, not chased.
- Agentic leg documented as confounded (thinking-off template), not fixed in the shared runner.
- **Junction removed permanently:** `models/` is now a real directory; the external store path no longer exists. Store-safety rule added to `AGENTS.md` / `GOLDEN-RULES.md` / `models/README.md`: recursive/forced deletes on the `models/` root are forbidden (per-file deletes in subdirs are normal maintenance).
- Follow-ups: `scripts/setup-sglang.sh` (reproducible WSL SGLang env), gate DENY rules for recursive deletes on the `models/` root (`.cursor`/`.pi`/`.gemini`/`.claude`), MFT recovery script → `scripts/recover-mft.py`, GGUF re-download queue.
