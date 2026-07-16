# 02 — Painel Baseline ao vivo

**What to build:** The dashboard shows the current Baseline's MODEL and runtime flags from local Search state, refreshing on the same poll as the shell. If no state file exists yet, the human sees a clear empty state instead of fake defaults.

**Blocked by:** 01 — Shell do dashboard localhost

**Status:** ready-for-agent

- [ ] Baseline panel displays MODEL plus key runtime flags from `.autoresearch_state.json`
- [ ] Values update on poll without requiring a page reload
- [ ] Missing or unreadable state shows an explicit empty/error message in pt-BR
- [ ] UI remains read-only (no controls that mutate Baseline or config)
