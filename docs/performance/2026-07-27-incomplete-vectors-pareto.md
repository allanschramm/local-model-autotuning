# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-27-incomplete-vectors-pareto.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

2. Fix bad Day picks (max TPS → LFM 8B; speed-band → LFM 1.2B half-IQ).
3. Land **ADR 0008**: Day = IQ ε-band then max TPS; Night maximin unchanged.
4. Complete incomplete Objective Vectors; document failures.
5. Treat distinct quants as separate Trials (Q3 ≠ Q4).

discrete **8 GB-class** NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA.

* **Day ≠ max TPS** (ADR 0006 failure) → ADR 0007 speed-band (commit `40322f4`) → still half-IQ → **ADR 0008**: `min ≥ DAY_IQ_RATIO × IQ_best` then max TPS (default ratio **0.75**).
* **Quantizations are not duplicates.** Ornith-35B Q3_K_XL and Q4_K_XL both get full pipelines.
* **Qwen3.5-9B coding** closed as **failure** after VRAM kills (no further retries on the operator host).


| Model | VRAM |
| :--- | ---: |
| `Ornith-1.0-35B-UD-Q4_K_XL` | 5.1 GB |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | 5.2 GB |
| `Ornith-1.0-9B-MTP-Q4_K_M` | 7.4 GB |


| Step | TPS | Peak VRAM |
| :--- | ---: | ---: |
| Historical run runtime |  | 4.4 GB |
| Historical run runtime |  | 4.5 GB |
| Historical run runtime |  | 4.6 GB |


| Attempt | Outcome |
| :--- | :--- |
| 32k + MTP n=4 | VRAM kill **7913 > 7900** mid HumanEval |
| 65k, no MTP | VRAM kill **7931 > 7900** mid HumanEval (preflight est 7584 MB) |


Claw-full @ 32k+MTP still fits (~7.5 GB) because tool calls are short; long codegen does not. Coding axis incomplete (VRAM rejected).

* **Night:** POCKET-35B (`min` 0.615).

* Day = max TPS selected LFM 8B → ADR 0007.
* Day = speed-band then IQ selected LFM 1.2B (~half Night IQ) → ADR 0008 (IQ ε-band then max TPS).
* Dense Qwen3.5-9B: historical “128k ran” was **without MTP** / light Autoloop load; not comparable to coding-10.
* Preflight VRAM underestimate on dense 65k coding (7584 est → 7931 peak).

* ADR 0007 then **0008** + CONTEXT / AGENTS Day wording
* Cards: `ornith-1.0-35b`, `ornith-1.0-9b`, `gemma-4-26b-a4b`, `qwen3.5-9b`
* Leaderboards: claw, coding, **pareto** + selection method note
