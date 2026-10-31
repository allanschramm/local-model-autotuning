# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-07-qwen36-35b-dflash-tps.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Hill-climb `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` for max TPS with focus on DFlash (`draft-dflash` + ggml-org `dflash-*` draft), then compare embedded MTP and document next TPS levers.

- Discrete GPU class; discrete **8 GB-class** NVIDIA physical VRAM; ~32 GB system RAM.
- `VRAM_LIMIT_MB=7900`; host budget ~27790 MB (discrete headroom policy).

- Runtime: upstream release `llama.cpp-releases/upstream/b10286` (`AUTORESEARCH_LLAMA_CPP_ROOT`).
- Target basename: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (Unsloth MTP-preserving GGUF; `nextn` confirmed earlier).
- DFlash draft basename: `dflash-Qwen3.6-35B-A3B-Q8_0.gguf` (ggml-org; BF16 sibling also downloaded, not smoked).
- Shared knobs: `CTX_SIZE=32768`, KV `q4_0/q4_0`, batch/ubatch `512/128`, threads `8/8`, `FLASH_ATTN=on`, `N_CPU_MOE=None` → harness `41`, sampler thinking/general (`TEMP=1.0`, `TOP_P=0.95`, `TOP_K=20`, `PRESENCE_PENALTY=1.5`).
- Harness fixes landed before successful DFlash/MTP smokes (same session):
  - MoE + external draft: VRAM estimate counts **draft file size only** (no flat `512 + 256×n` speculative workspace).
  - MoE `n_cpu_moe > 0`: effective VRAM budget = **configured** `VRAM_LIMIT_MB` (skip free−headroom clamp; OS-reserved VRAM was false-rejecting).


| Trial | Spec | `SPEC_DRAFT_N_MAX` | Draft | Bench tg (TPS) | Peak VRAM | Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `ef3094b2-d634-4308-b6f4-cc300c4a6d2b` | none | 0 | — | **27.2** | 4.1 GB |  |
| `06dce572-7122-45a3-a075-901c7460dda8` | `draft-dflash` | 15 | `dflash-…-Q8_0.gguf` | **17.5** | 5.6 GB | −36% vs no-spec; |
| `3810c77b-f108-4874-8ffe-21c0ade7209a` | `draft-mtp` | 2 | embedded | **27.5** | 4.6 GB | ≈ flat vs no-spec (+0.3); |


**Verdict on this Fingerprint:** DFlash loads on upstream `b10286` but **hurts** max TPS. Embedded MTP is a wash. Prefer no-spec or MTP as speed seeds; do not default DFlash for Day-speed on this MoE+offload path.


| Trial | Failure |
| --- | --- |
| `4c976320-656a-40ac-bf97-e50a173d1bd4` | DFlash n_max=15: est 11524 > limit 7900 (flat speculative workspace dominated estimate) |
| `9024437e-4edc-432b-ba3d-000ffc30ec55` | DFlash n_max=1: est 7940 > effective free−headroom (~6804) |


1. Lower `CTX_SIZE` (16k / 8k) for pure tg climbs.
2. Lighter quant (Q3 / IQ3) — separate Trial family; historical Q3+MTP rows elsewhere in TSV looked faster.
3. Sweep `N_CPU_MOE` around auto-41 (measure stall vs VRAM).
4. Batch / ubatch / threads; KV variants (stock vs TurboQuant `turbo3` on TQ runtime only).

Medium: BF16 DFlash A/B (low prior after Q8 loss); external `mtp-*` pair; n-gram (repetitive text only).

Other engines / formats: ExLlamaV3/EXL3 is the only serious Windows CUDA challenger in-repo research — **unverified** on this Qwen3.6 MoE; format + harness cost high. SGLang/vLLM = WSL/Linux-first; prior GPTQ Qwen3.6 attempts in TSV failed. Same llama.cpp wrappers (Ollama/kobold) do not raise the TPS ceiling.

- Autoloop `--mode tps` without `--models` prompted interactively → `EOFError` in non-interactive shell.
- One MTP smoke process died mid-run (exit −1, no log); restarted successfully (`3810c77b-…`).

- Keep DFlash drafts available; do **not** treat DFlash as max-TPS default for this basename @ 32k on upstream b10286.
- Stop before further hill-climb; document matrix + brainstorm for a later Trial ladder.
- Harness MoE VRAM exceptions remain in `autoresearch/core/llama_runner.py` + `autoresearch/AGENTS.md`.

- Model card: [qwen3.6-35b-a3b.md](../models/qwen3.6-35b-a3b.md)
- Prior no-spec Objective Vector @ 100k turbo3: [2026-08-02-qwen36-35b-unsloth-100k.md](2026-08-02-qwen36-35b-unsloth-100k.md)
- Spec formats: [speculative-decoding-formats.md](../discovery/speculative-decoding-formats.md)
- Engine landscape / EXL3: [fastest-tps-inference-engine.md](../discovery/fastest-tps-inference-engine.md)

## Preserved historical runtime tables

| Trial | Spec | `SPEC_DRAFT_N_MAX` | Draft | Bench tg (TPS) | Peak VRAM | Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `ef3094b2-d634-4308-b6f4-cc300c4a6d2b` | none | 0 | — | **27.2** | 4.1 GB |  |
| `06dce572-7122-45a3-a075-901c7460dda8` | `draft-dflash` | 15 | `dflash-…-Q8_0.gguf` | **17.5** | 5.6 GB |  |
| `3810c77b-f108-4874-8ffe-21c0ade7209a` | `draft-mtp` | 2 | embedded | **27.5** | 4.6 GB |  |
