# 2026-08-18 — Qwen3.8-27B Q1Q validation → rejected

## Goal

Execute the Qwen3.8-27B coding-loop queue from `models/TASK.md` (handoff of
[2026-08-17-qwen38-27b-candidate.md](../performance/2026-08-17-qwen38-27b-candidate.md)): download
`Qwen3.8-27B-Q1Q-XYZ-v2.gguf`, validate (MTP, KV, engine load), then run the 3-way
shootout vs KAT-Coder / POCKET-35B for the Night (issue-loop) profile.

## Hardware

discrete 8 GB-class NVIDIA (Baseline `VRAM_LIMIT_MB` 7900, clamped 7676 = physical −
512 keepout), Windows, 32 GB-class RAM. Engines: upstream `b10375` (pinned default) and
`turboquant/tqp-v0.3.0` (selected for turbo KV).

## Setup

- Download: `hf download quimmedes/Qwen3.8-27B-XYZ Qwen3.8-27B-Q1Q-XYZ-v2.gguf` →
  `models/quimmedes/qwen3.8-27b-xyz/` (6.15 GiB, 6300.1 MiB, text-only; mmproj skipped).
- Baseline seeded in `autoresearch/core/config.py`: CTX 65536, KV `turbo2`, sampler
  TEMP 0.6 / TOP_P 0.9 / TOP_K 40 / REPEAT_PENALTY 1.1 (model-card recommended).
- Runtime via `AUTORESEARCH_LLAMA_CPP_ROOT=llama.cpp-releases/turboquant/tqp-v0.3.0`.

## Commands (reproducible)

```bash
./venv/Scripts/python.exe scripts/model_info.py Qwen3.8-27B-Q1Q-XYZ-v2.gguf --ctx 65536
./venv/Scripts/python.exe scripts/model_info.py Qwen3.8-27B-Q1Q-XYZ-v2.gguf --ctx 131072
# raw key probe (gguf.GGUFReader): fields["qwen35.nextn_predict_layers"].parts / .data
AUTORESEARCH_LLAMA_CPP_ROOT=.../tqp-v0.3.0 AUTORESEARCH_SKIP_FREE_CLAMP=1 \
  ./venv/Scripts/python.exe benchmark_search.py --validation --desc "validation Qwen3.8-27B-Q1Q-XYZ-v2"
```

## Findings

1. **Metadata:** dense, 64 blocks, arch `qwen35`, native ctx 262144. KV f16 =
   **8192 MiB @64k / 16384 @128k** (hybrid SSM — full attention every 4th layer).
2. **No MTP head in Q1Q-v2.** Raw GGUF value bytes for `qwen35.nextn_predict_layers`
   are 0; 0 of 851 tensors are `nextn.*`. The 2026-08-17 candidate research claimed
   embedded MTP "per the quantizer pack" — wrong for this file. `gguf_has_mtp()`
   correctly returns False. Low-bit quants can drop the head: verify per-file.
3. **Runtime KV type matrix (measured `--help`, b10362/b10375 vs tqp-v0.3.0):**
   - upstream `-ctk/-ctv` allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
     — **no turbo types**.
   - TurboQuant `tqp-v0.3.0`: same list **plus turbo2, turbo3, turbo4**; also
     `--spec-type draft-mtp,ngram-mod` composite and `--reasoning-budget`.
   - `qwen35` arch loads on tqp-v0.3.0; thinking-mode generation works
     (`reasoning_content` carries the output while `content` stays empty).
4. **TPS:** bench tg 512 = **27.9 t/s** (passes TPS_FLOOR 20; fails Day floor 50).
5. **VRAM @64k turbo2:** preflight est 7419 MB; **measured peak 7720 MB > limit 7676**
   → runtime monitor killed the server. Implied real turbo2 KV factor ≈
   (7720 − 6300 − 300)/8 GB-class total ≈ **0.137×f16** vs harness `VRAM_QUANT_FACTORS["turbo2"]=0.10`.
   Filed as [issue #58](https://github.com/allanschramm/local-model-autotuning/issues/58).
6. **Max safe ctx ≈ 61k** (real peak ≈ est + 300 MB). Night floor 65536 unreachable;
   hard target ≥100k unreachable on any local runtime (turbo2 is the densest KV type
   available).
7. **Verdict:** fails both profile gates — Day (TPS ≥ 50) and Night (CTX ≥ 65536).
   Shootout not run: no profile the model can serve. Queue **parked**; `models/TASK.md`
   carries the rejection record; `config.py` restored to the prior Baseline.

## Errors

- **Free-VRAM clamp false-reject (issue #10 class):** first validation attempt failed
  with effective 6524 MB (free 7036 − 512 headroom) although the configured limit 7676
  fits. Used the documented `AUTORESEARCH_SKIP_FREE_CLAMP=1` escape; runtime VRAM
  monitoring remained the kill guard.
- **Initial false bug claim (correction):** `gguf_has_mtp()` returned False and
  `ReaderField.data` printed `[3]` — misread as "file declares 3 → harness bug". In the
  installed gguf-python, `ReaderField.data` is an **index list into `parts`**, not the
  value; the value bytes (`parts[3]`) are 0. No harness bug — the file genuinely lacks
  the MTP head. Do not read `field.data` as the scalar value.
- **Sampler seeding edit collision:** TOP_P / TOP_K / REPEAT_PENALTY edits matched both
  `UNIVERSAL_FALLBACK_SAMPLER` and `SAMPLER_DEFAULTS`; anchored with surrounding-context
  replacements to hit `SAMPLER_DEFAULTS` only.

## Decisions

- Runtime: TurboQuant `tqp-v0.3.0` over pinned upstream b10375 (only local runtime with
  turbo KV); engine tag recorded per-Trial.
- Queue parked (operator decision) after both profile gates failed.
- [Issue #58](https://github.com/allanschramm/local-model-autotuning/issues/58) filed for
  the turbo2 KV factor calibration gap (0.10 vs measured ~0.137).
