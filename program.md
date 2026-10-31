# Operator contract — bounded auto-tuning

Read AGENTS.md and autoresearch/TUNING.md before a model test.

1. Use the project venv and compatible prebuilt runtime. Pin and verify AUTORESEARCH_LLAMA_CPP_ROOT. Do not rebuild vendor trees.
2. Configure the chosen GGUF, engine and publisher sampler seed in ignored core/config.py. Model selection and sampler optimization are outside this Search.
3. Review a JSON plan with minimum_context, max_trials and engine search_space. Preserve both Baseline context and plan minimum. Never reduce context to bypass resource pressure.
4. Run python -m autoresearch.tuning --plan <plan.json> through the venv. No background loop or ad-hoc runtime overrides.
5. Let it complete with RAM/VRAM/process guards active. PPL finishes before the monitored server starts; infrastructure failure must not become a zero model score.
6. Inspect results.db and artifacts. Keep one fixed PPL ceiling and distinguish allocated context from prompt occupancy. Compare matching protocols and conditions.
7. Select a Pareto alternative and explicitly apply its Fingerprint with scripts/apply_fingerprint.py if requested. Do not automatically replace Baseline or launcher selection.

Quality evaluation and teaching are outside this distribution. Archived commands are historical references, not current instructions.
