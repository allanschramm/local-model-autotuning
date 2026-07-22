# Teach — Course Materials

## Purpose
Course materials to teach anyone to run local AI from scratch: conceptual foundations for beginners (Módulo 0), local LLM performance (Semana 1), then quality (Semana 2). HTML lessons, quizzes, curriculum.

## Ownership
Course operator / instructors. Not part of the autotuning runtime loop.

## Local Contracts
- Purpose of repo course: Workshop AILOCAL Essentials (teach anyone to run local AI from scratch).
- Módulo 0: Conceptual foundations for absolute beginners (AI basics, Cloud vs Local, CPU/GPU/VRAM, Quantization, interactive VRAM calculator).
- Semana 1 scope locked: TPS / performance only — no model-quality scoring.
- Semana 2 complete: Quality & Sampling (s2d1), Skills & MCPs (s2d2), Sandboxes & Hooks/Gates (s2d3), and Integrated Use Case (s2d4).
- Student workflow: 100% local inside git checkout. Entrypoints: (1) Agent mode via `/teach` skill in CLI/IDE, (2) Browser mode via local `teach/index.html`.
- Student CLI tools: `scripts/check_hardware.py` (GPU/VRAM recommender) and `scripts/verify_setup.py` (server health & TPS benchmark).
- Agent guidance: During `/teach` sessions, agent acts as interactive tutor. Follow 5-step onboarding: (1) Check/install Python, (2) Create `venv`, (3) Install `requirements.txt`, (4) Run `check_hardware.py`, (5) Run `verify_setup.py`. Then proceed to Module 0.
- Quizzes: hashed answers only (`assets/QUIZ-HASH.md`); options simplified in pt-BR for beginners (no LM Studio references in quizzes).
- No GGUFs, results, or run logs in this tree.

## Work Guidance
- Prefer editing lesson HTML + `CURRICULUM.md` / `MISSION.md` together.
- Keep glossary/definitions accurate (VRAM ≠ “must fit”; offload/MoE = Dia 3).
- Ensure interactive HTML elements (quizzes, calculators, troubleshooting wizards) work 100% offline in static browser view.

## Verification
- Open `index.html` or lesson HTML in a browser; click quizzes (client-side hash check).
- Confirm `CURRICULUM.md` links resolve.

## Child DOX Index
- (none — flat under `teach/`)
