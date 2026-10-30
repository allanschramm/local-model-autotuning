# AutoResearch — local runtime tuning

**Baseline**: the operator's local model, engine and frozen sampler configuration in ignored core/config.py. Search does not automatically replace it.

**Search**: finite exploration of a supplied engine Search Space within a Trial budget.

**Search Space**: ordered supported engine values. Model, sampler and hardware-safety budgets are frozen.

**Neighbor**: one engine parameter changes to an adjacent allowed value, preserving batch consistency and the minimum context.

**Trial**: a protected configuration attempt: fixed-text PPL followed by monitored full-allocation generation. Infrastructure failure is an outcome, not a fabricated model score.

**Configured context**: allocated slot capacity, at least max(Baseline target, plan minimum). Prompt occupancy is separate evidence.

**TPS**: median server-reported generated tokens per second for complete generations under the identified protocol.

**PPL ceiling**: at most 1% above the first successful Baseline PPL measurement. Accepted Neighbors do not move this reference.

**Pareto frontier**: admitted configurations non-dominated on context and TPS. Equal-axis duplicates do not expand it; admission alone does not mean final frontier membership.

**Fingerprint**: portable model-basename, engine and optional sampler file. Selection and application are explicit operator actions.

**Evidence**: canonical SQLite tuning_runs/tuning_trials, checkpoints, logs and Fingerprints. Historical quality tables may coexist but are not tuning inputs or dashboard rows.

See [TUNING.md](autoresearch/TUNING.md) for the exact protocol and [the archive index](docs/retired-quality.md) for retired four-axis vocabulary.
