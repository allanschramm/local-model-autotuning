# Session: tuning-only release smoke — 2026-10-31

## Scope

Validate the published tuning-only runtime tree at commit `b4d450ce4e1830c0d233d99f9def6c2aea430eef` after PR 92. No quality benchmark, cloud endpoint or autonomous loop was used. Release documentation does not change this runtime protocol.

Real inference ran on Windows with a discrete CUDA GPU, using the prebuilt upstream@b11577 runtime. These measurements apply to this run only; they do not establish support or performance for other hardware.

## Configuration and method

- Existing local model: Spark-X2.6-4B-Q4_K_M.gguf, dense, no speculative draft, N_CPU_MOE=None. No weights were downloaded for this session.
- KV K/V=q4_0, Flash Attention on, batch=512, threads/threads_batch=8, mmap enabled. Every process kept the runtime's resource and lifetime guards active.
- Frozen sampler: temperature=1.0, top_p=0.95, top_k=0, min_p=0, repeat_penalty=1, presence_penalty=0, frequency_penalty=0. The [publisher's card](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF) was checked on the run date; sampler settings were not searched.
- Preserve the checkout Baseline minimum of 131072 tokens. First validate one unchanged configuration; then run a separate Search with at most three attempts. Baseline context was not reduced.
- Protocol tuning-runtime-v1: PPL on the fixed 2048-token chunk, then three complete 512-token native generations per configuration. Median TPS comes from server timings. Check configured allocation in /props and the actual response slot in /slots.
- Recorded PPL text SHA256: `726d5f577118c3fa0951903254c3184b4eb3e54de4926d07df883eacfc4304e8`.

The bounded Search plan was:

```json
{
  "minimum_context": 131072,
  "max_trials": 3,
  "search_space": {
    "CTX_SIZE": [131072, 262144],
    "UBATCH_SIZE": [64, 128]
  }
}
```

Seed the selected model and frozen configuration in the ignored Baseline, pin the compatible runtime, then run `python -m autoresearch.tuning --plan <plan.json>` through the project venv. Preserve any larger operator Baseline context when adapting this plan; no model or runtime is bundled with the source release. See [TUNING.md](../../autoresearch/TUNING.md) for installation and exact measurement contracts.

## Recorded outcomes

All four attempts finished OK and confirmed the exact configured per-slot allocation. The validation and Search have independent PPL references; only the three Search attempts determine that Search's frontier.

| Attempt | Context allocation | Ubatch | Median TPS | PPL | Admission / Search frontier |
|---|---:|---:|---:|---:|---|
| Initial validation | 131072 | 128 | 64.84 | 27.2259 | Admitted; separate validation run |
| Search Baseline | 131072 | 128 | 65.66 | 27.2259 | Admitted; retained |
| Larger context | 262144 | 128 | 38.19 | 27.2259 | Admitted; retained |
| Smaller ubatch | 131072 | 64 | 65.28 | 27.1505 | Admitted; dominated |

The Search reference was 27.2259, with an admission ceiling of 27.498159. It remained fixed across attempts. Both retained points represent usable alternatives under this protocol; neither is a universal winner. The small difference between repeated 131072/128 measurements is not evidence of an optimization.

## Evidence and integration checks

Canonical evidence remains in the local results.db, with native logs, response samples, JSON artifacts and portable Fingerprints under tuning-artifacts/:

- Validation run_id: `82a92ddfabdf462bb4728b54b6ecf0d2`.
- Search run_id: `42da4f89cec94c67836e775b6a4e4c8f`.
- Real HTTP dashboard checks displayed the measured tuning rows/PPL without legacy quality fields.
- Fingerprint application was checked against a separate Baseline copy for the 262144-context alternative. No launcher selection was changed.
- SQLite integrity_check returned ok. The private Baseline was restored byte-for-byte; no full model server remained after teardown. The original checkout's Baseline and aliases were untouched.

For a local copy of the recorded database, the following query identifies these attempts without relying on a text export:

```sql
SELECT run_id, trial_id, ctx, tps, ppl, outcome, accepted, artifact
FROM tuning_trials
WHERE run_id IN (
  '82a92ddfabdf462bb4728b54b6ecf0d2',
  '42da4f89cec94c67836e775b6a4e4c8f'
)
ORDER BY rowid;
```

Raw artifacts include local paths and a machine Baseline, so they are not release assets. The source release provides the protocol and simulated E2Es for repeatable software verification; throughput on another rig must be measured independently.

## Interpretation limits

The generated prompt occupied only a small fraction of the allocated window. Confirming 262144 allocated tokens does not validate retrieval, answer quality, throughput at deep prompt fill, or long-context PPL. PPL admission is a degradation guard on the fixed corpus chunk, not a replacement for the independent quality Benchmark.

No real Linux GPU inference, live cloud-provider validation, exhaustive Search, deliberate hardware watchdog trip, or broad model-compatibility sweep was performed. Linux CI exercised simulated runtime adapters and platform branches.
