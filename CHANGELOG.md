# Changelog

## v0.1.0 — 2026-10-31

First versioned release of the independent local auto-tuning distribution.

### Runtime tuning

- Finite Search over engine settings, retaining TPS × configured-context alternatives.
- Preserve the greater of the Baseline context and the plan's minimum context. Admit candidates only within 1% of the fixed per-Search PPL reference.
- Native PPL measurement, monitored full-context server allocation, complete generation samples and slot-context verification.
- Canonical SQLite evidence, portable Fingerprints and a read-only tuning dashboard. Applying a selected Fingerprint remains explicit.
- Runtime-only installation through requirements.txt; development checks through requirements-dev.txt.

### Migration from unversioned checkouts

- Use `python -m autoresearch.tuning --plan <plan.json>` through the project venv; see [the installation guide](README.md) and [the protocol](autoresearch/TUNING.md).
- `benchmark_search.py`, `autoresearch.runners.run`, `autoloop.py`, legacy quality evaluators, ranking scripts and requirements-legacy.txt are retired in this distribution. They are not aliases for the new protocol.
- Historical SQLite tables remain intact; the dashboard displays only tuning-runtime-v1 evidence. Old quality scores are not comparable with the new two-axis frontier.
- Independent Pi Benchmark work remains on staging/independent-tuning-benchmark for later extraction. Teaching already belongs to the separate ailocal-teach project.
- Runtime/model guides stay here; original mixed quality records remain accessible through [the archive index](docs/retired-quality.md).

### Validation and limits

- Windows: Ruff and 144 tests passed, 2 skipped. Linux CI: 141 tests passed, 5 skipped; isolated tuning, dashboard and clean-install E2Es passed. Python CodeQL analysis reported no findings on the tested runtime commit.
- Real Windows/CUDA smoke on runtime upstream@b11577: Spark-X2.6-4B Q4_K_M completed four attempts. The bounded Search retained 131072 context at 65.66 TPS and 262144 context at 38.19 TPS, both with PPL 27.2259. [Reproduction and evidence](docs/sessions/2026-10-31-tuning-only-release-smoke.md).
- These are measurements from one local rig, not portable speed guarantees. TPS uses a short prompt and a fully allocated context; it does not establish long-context task quality. PPL uses one fixed 2048-token chunk.
- Linux validation used simulated model/hardware adapters. No live cloud-provider validation, real Linux GPU inference, or broad hardware/model compatibility claim is included.
- Source release only: provide model weights and a compatible prebuilt runtime separately. The private Baseline, local aliases, model files and raw machine-specific evidence are not distributed.
- Original AILOCAL contributions use PolyForm Noncommercial 1.0.0; imported material retains its own licenses. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
