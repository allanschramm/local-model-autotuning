# `ui/` — Operator Dashboard Contract

## Purpose
Read-only localhost dashboard for monitoring Baseline, recent Trials, and Trial server log during Search. Humans watch; agents start the UI.

## Ownership
Repository developers. Port and panels owned here — not by `autoresearch/` runners.

## Local Contracts
- Bind **127.0.0.1:18765** only. UI language **pt-BR**.
- Start: `.\venv\Scripts\python.exe -m ui` (or `./venv/bin/python -m ui`).
- Poll `/api/status` (~2.5s). Payload includes `baseline`, `trials` (last 50), `run_state`, `log_tail`.
- Baseline from `autoresearch/core/config.py` via import+reload — never `.autoresearch_state.json`.
- Trials via `autoresearch.runners.run.read_rows` + `classify.is_on_front` (`on_front` only). Never write TSV.
- Log path pinned: `autoresearch/runners/llama_server.log`. `Em execução` when mtime within ~10s; else `Idle`. Missing log → pt-BR empty, no crash.
- **No external runtime deps; vanilla JS only** (`ui/requirements.txt`). Static assets allowed under `ui/static/` (CSS + bundled OFL fonts) and served by the stdlib `http.server` (ADR 0011). No CSS/JS frameworks, CDN, or build step.
- **AILOCAL design language** (ADR 0011): dark neutral base, accent-only blue `#339dff`, Inter + JetBrains Mono, static precision-grid, restrained motion. Blue is a highlight, never a dominant surface.
- **pt-BR display of Trial Status**: canonical English labels stay in the data/API; the UI renders localized pills — `on_front` → "na fronteira", `dominated` → "dominado", `incomplete` → "incompleto", `rejected` → "rejeitado" (CONTEXT.md `Status de exibição`).
- No process control (no start/stop server, autoloop, or Search from the UI).

## Work Guidance
- Agent starts the dashboard when the human wants to monitor a Trial/Search session.
- Human monitors the browser; agent does not treat the dashboard as a control plane.
- Keep panels additive and poll-friendly; prefer small helpers (`trial_reader.py`, `run_log.py`).

## Verification
Smoke: `python -m ui` → GET `/` (200, `lang=pt-BR`) and GET `/api/status` (JSON with baseline/trials/run_state).
Tests: `tests/test_ui_server.py` — 20 HTTP-level tests (real `HTTPServer` thread, ephemeral port). Static routes are guarded by `_resolve_static` (path containment: no `..`, no absolute/UNC paths, resolved path must stay inside `ui/static/`); Content-Type comes from a fixed suffix allowlist, never path data.
## Child DOX Index
None — `ui/` is a leaf.
