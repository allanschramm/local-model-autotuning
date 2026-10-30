---
name: onboarding
description: Explain this local GGUF auto-tuning repository and establish its venv, prebuilt runtime, model store, Baseline and finite plan. Use for setup, onboarding, primeiro uso or como começar.
---

# Onboarding local auto-tuning

Match the operator's language; default to pt-BR. Read root AGENTS.md, CONTEXT.md and README.md.

- Explain the scope: optimize TPS and configured context under fixed PPL and resource constraints. Model task quality and teaching are outside this distribution.
- Inspect hardware with the project tools. Create a project venv if missing and install requirements.txt; developer checks additionally use requirements-dev.txt. Never use system-global Python for project operations.
- Provision a compatible prebuilt llama.cpp release and pin AUTORESEARCH_LLAMA_CPP_ROOT. Do not rebuild vendor trees to repair stale runtime compatibility.
- Use a model chosen by the operator; model metadata and provisioning follow models/README.md and the validation skill. Keep private paths/aliases and the existing GGUF store untouched.
- Seed config.py from its public example only when absent. Preserve an existing Baseline and configured context; review publisher sampler recommendations as frozen inputs.
- Review a finite JSON plan and the minimum context. Use python -m autoresearch.tuning --plan <plan.json> only when a model test/Search is authorized. Do not start an autonomous background loop.
- Explain canonical SQLite and artifact/Fingerprint selection. Application is explicit, not automatic.
- The optional read-only dashboard runs with the venv's python -m ui on localhost:18765 and displays tuning data only.

Read autoresearch/TUNING.md for exact measurement/limits and program.md for execution discipline. Safe setup checks do not authorize an unrequested model download, inference run or Baseline replacement; existing explicit authorization persists.
