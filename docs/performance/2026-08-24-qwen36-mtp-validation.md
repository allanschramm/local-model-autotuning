# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-24-qwen36-mtp-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Follow-up to [`2026-08-24-codacus-cache-full-offload.md`](../sessions/2026-08-24-codacus-cache-full-offload.md). First repo-harness validation of the MTP-packaged GGUF on the pinned upstream engine.

- discrete_gpu, 8 GB-class VRAM, Windows; 32 GB host RAM.

- Engine: pinned upstream release b10549 via `AUTORESEARCH_LLAMA_CPP_ROOT` (fork not involved).
- Provenance note: this file comes from the `-MTP-GGUF` repo and carries `qwen35moe.nextn_predict_layers = 1` + a full `blk.40` draft layer (`block_count=41`). The same basename in the plain repo has none of these. Baseline seeded with a repo-relative `MODEL` ref so `resolve_model_path` hits the exact file despite the basename collision.
- Baseline (gitignored `config.py`, restored to the Qwen3.8-4B winner afterwards): `CTX_SIZE=65536`, `SPEC_TYPE='draft-mtp'`, `SPEC_DRAFT_N_MAX=2`, `N_CPU_MOE=None` (auto), KV `q4_0`, sampler temp 0.7 / top-k 20 / top-p 0.95 per model card.

```bash
  --desc "validation Qwen3.6-35B-A3B-UD-Q4_K_M-MTP 65536 draft-mtp"
```


| Field | Measured | Threshold |
| --- | --- | --- |
| Bench tg (512) | **32.1 t/s** | ≥ 20.0 t/s |
| Peak VRAM | **4.4 GB** | no shared spill |
| Category | `validation`, status `incomplete` | — |


1. Embedded-MTP speculative decoding works on the pinned engine: b10549 contains `draft-mtp` support and consumed the file's `nextn_predict_layers=1` without any fork features.
2. Measured decode (32.1 t/s @65k q4_0 KV) matches the fork's short-prompt server measurements (31.5–32.4 t/s) — the MTP gain is engine-independent.
3. VRAM peak 4.4 GB leaves headroom; host RAM stable throughout.
4.

- Baseline restored to `Qwen3.8-4B-Q4_K_M.gguf` @131072 after the run (off-winner seed must never linger).
- Full Trial (Claw-15 + coding-10) is the natural next step if the operator wants a Pareto vector; not run autonomously.
