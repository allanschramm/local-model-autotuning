# Teach — Course Materials

## Purpose
Course materials to teach anyone to run local AI from scratch. Módulo 0 and Semana 1 are published; Semana 2 is curriculum work in progress.

## Ownership
Course operator / instructors. Not part of the autotuning runtime loop.

## Local Contracts
- Purpose of repo course: Workshop AILOCAL Essentials (teach anyone to run local AI from scratch).
- Módulo 0: Conceptual foundations for absolute beginners (AI basics, Cloud vs Local, CPU/GPU/VRAM, Quantization, interactive VRAM calculator).
- Semana 1 scope locked: TPS / performance only — no model-quality scoring.
- Semana 2 is unpublished curriculum work. Its four HTML drafts remain visible as “Em construção”, outside progress and next-lesson routing.
- Student workflow: 100% local inside git checkout. Entrypoints: (1) Agent mode via `/teach` skill in CLI/IDE, (2) Browser mode via local `teach/index.html`.
- Lesson HTML nav: always link `../index.html` as **Guia**; prev/next lesson HTML only. Never link `MISSION.md` / `CURRICULUM.md` from student-facing HTML (`file://` shows raw markdown).
- Student references stay HTML under `reference/` (glossário, flags). Prefer those over `docs/*.md` links in lesson bodies.
- Guia CTA: `index.html` “Próximo passo” uses `TeachProgress.getNextLesson` (first incomplete in curriculum order).
- Onboarding step 5 (`verify_setup.py`) is optional until a local server listens on the selected port; the repository server helper defaults to 18080.
- Student CLI tools: `scripts/check_hardware.py` (GPU/VRAM recommender) and `scripts/verify_setup.py` (server health & TPS benchmark).
- Agent guidance: During `/teach` sessions, agent acts as interactive tutor. Follow 5-step onboarding: (1) Check/install Python, (2) Create `venv`, (3) Install `requirements.txt`, (4) Run `check_hardware.py`, (5) Run `verify_setup.py`. Then proceed to Module 0.
- Quizzes: hashed answers only (`assets/QUIZ-HASH.md`); options simplified in pt-BR for beginners (no LM Studio references in quizzes).
- **Completion gate:** each published lesson requires its quiz plus a practice check (`assets/progress.js`; localStorage keys `teach_quiz_pass_v1` and `teach_practice_pass_v1`). Simulated practice counts as completion but remains labeled until replaced by real practice. Preserve draft Semana 2 state but ignore it in published progress.
- Dense GGUF guidance must require the model to fit physical VRAM; never recommend partial dense offload/shared-memory spill. Expert offload is MoE-only.
- No GGUFs, results, or run logs in this tree.

## Work Guidance
- Prefer editing lesson HTML + `CURRICULUM.md` / `MISSION.md` together.
- Keep glossary/definitions accurate (dense fits physical VRAM; expert offload/MoE = Dia 3).
- Ensure interactive HTML elements (quizzes, calculators, troubleshooting wizards) work 100% offline in static browser view (`file://`). Quiz/progress scripts are classic (no ES modules) for that reason.

## Verification
- Open `index.html` or lesson HTML in a browser; click quizzes (client-side hash check).
- Confirm “Concluir” stays locked until quiz and practice pass; simulated practice shows its pending-real-practice label.
- Confirm lesson headers link to `index.html` (Guia) and prev/next HTML lessons — no `.md` in student nav.

## Child DOX Index
- [GLOSSARY.md](GLOSSARY.md) — Canonical terms (HTML: `reference/glossario.html`).
- [reference/llamacpp-flags.html](reference/llamacpp-flags.html) — Student flag cheat sheet (`file://`).
- (otherwise flat under `teach/`)
