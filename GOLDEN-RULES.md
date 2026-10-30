# Runtime safety rules

[AGENTS.md](AGENTS.md) is the binding contract; [program.md](program.md) is the bounded procedure.

- Preserve configured context and minimum-context constraints.
- Keep one fixed PPL ceiling while optimizing TPS/context.
- Use GGUF metadata and upstream-supported flags.
- Keep RAM/VRAM, process lifetime and single-load safeguards active.
- Use the venv, prebuilt runtime, finite plan and canonical SQLite evidence.
- Keep model assets, aliases and Baseline local; explicitly select/apply Fingerprints.
