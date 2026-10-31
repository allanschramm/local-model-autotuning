# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-08-thinking-claw-harness-fix.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

discrete **8 GB-class** NVIDIA class rig, Windows, upstream `llama.cpp` CUDA. Baseline `VRAM_LIMIT_MB` used for Pareto budget bucket (not peak alone).

- `agentic_runner.py`: surface + history round-trip for reasoning.
- `evaluation.py`: agentic `max_tokens ≥ 2048`.


| Stage | note |
| :---: | :--- |
| Pre-fix remasure | False floor |
| Post-fix remasure | peak ~7.7 GB, TPS ~42.1 |


- First-non-None Pareto merge kept stale **0.3333** after a better remasure — fixed to best/max on maximize axes before trusting status.
- Dense free-at-start VRAM clamp (issue #10) can false-reject when desktop holds VRAM; escape env above; runtime monitor remains kill guard.

Start with Qwen3.5 / Qwen3.6 / Qwythos / remaining Ornith siblings / Gemma-4 / KAT-Coder / Nanbeige / Pocket.
