---
name: validation
description: Download or inspect a chosen GGUF and validate local load, fixed-reference PPL, monitored TPS and configured-context allocation through one bounded tuning Trial. Use for model smoke/validation requests.
---

# Local runtime validation

Read root AGENTS.md, autoresearch/AGENTS.md and autoresearch/TUNING.md. This validates runtime behavior, not model task quality.

1. Use the repo venv. Inspect the chosen basename with scripts/model_info.py; download a requested missing GGUF with hf into models/<publisher>/<model>/. Never delete/prune models/ or its target, and never replace the operator's assets without authorization.
2. Pin and verify AUTORESEARCH_LLAMA_CPP_ROOT to a compatible prebuilt release. Do not rebuild vendor code or invoke model binaries directly.
3. Seed the ignored Baseline only if absent. Configure the selected model and publisher sampler settings. Preserve the current configured context and the user's minimum; do not lower either for fit. Address pressure through host hygiene, KV quantization, offload, batch or lighter requested quants.
4. Make a machine-local JSON plan with minimum_context at least Baseline CTX_SIZE, max_trials=1 and one unchanged engine value in search_space (for example BATCH_SIZE containing the current Baseline value). This is a single bounded runtime attempt, not an autonomous loop.
5. Run python -m autoresearch.tuning --plan <plan.json> through the venv. No ad-hoc model/engine overrides or command execution timeout. Keep all resource/lifetime guards active.
6. Query results.db tuning_runs/tuning_trials and inspect the printed artifacts. Report outcome, configured/observed context, PPL reference, TPS samples and diagnostic. Distinguish allocation from prompt occupancy. Failed infrastructure is not a zero model-quality score.
7. Leave Baseline, launcher aliases and selected Fingerprint unchanged unless explicit application was requested. Quality tasks belong to the independent benchmark work and are not part of this validation.

Software verification without model load: scripts/test_tuning_e2e.py, scripts/test_tuning_consumers_e2e.py and scripts/test_tuning_install_e2e.py retain SQLite evidence.
