# 01 — Shell do dashboard localhost

**What to build:** From the operator's perspective, running `python -m ui` (after installing UI deps) starts a localhost-only dashboard at port 18765 with a Portuguese skeleton page that polls a status stub. An agent can start it and hand the human the link; the browser loads without crash.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `python -m ui` binds only to `127.0.0.1:18765` and serves an HTML shell in pt-BR
- [ ] UI deps live under `ui/requirements.txt` (not forced into root requirements)
- [ ] Page polls an `/api/status` (or equivalent) stub on a ~2–3s interval without errors when artifacts are missing
- [ ] Empty/missing state is readable (no blank hang, no stack trace in the browser)
