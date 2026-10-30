# Local runtime workflow — good enough tuning

This repository tunes an operator-chosen GGUF; model task-quality selection belongs to the independent benchmark.

1. Follow [installation](../../README.md) and inspect hardware/model metadata with scripts/check_hardware.py and scripts/model_info.py through the venv.
2. Keep GGUFs in the [model store](../../models/README.md). Preserve Baseline and target context; choose a compatible prebuilt runtime and pin AUTORESEARCH_LLAMA_CPP_ROOT.
3. Seed a missing config.py from its public example. Freeze the chosen model, sampler and safety budgets.
4. Review a finite engine plan, minimum context and Trial budget. Run python -m autoresearch.tuning --plan <plan.json> through the venv.
5. Inspect SQLite and artifacts, then explicitly select/apply a Fingerprint if desired. The read-only dashboard shows tuning attempts and PPL/frontier evidence.

The exact method and constraints are in [TUNING.md](../../autoresearch/TUNING.md). Historical documents are linked in the [archive index](../retired-quality.md); past CLI smoke TPS does not establish full-context performance.
