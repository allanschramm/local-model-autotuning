# 05 — Docs DOX + caminho do agente

**What to build:** Durable docs tell agents and humans how to install/start the dashboard, what it shows, and what it deliberately does not do (no process control / no agent log). DOX covers the new `ui/` boundary; README stays pt-BR for the user-facing section.

**Blocked by:** 02 — Painel Baseline ao vivo; 03 — Tabela de Trials (últimos 50); 04 — Status “Em execução” + log do modelo

**Status:** ready-for-agent

- [ ] `ui/AGENTS.md` exists and is listed from root Child DOX Index
- [ ] README (pt-BR) documents `pip install -r ui/requirements.txt`, `python -m ui`, and `http://127.0.0.1:18765`
- [ ] Agent onboarding (or equivalent) notes: agent starts UI; human monitors; Search stays with the coding agent
- [ ] Docs state non-goals: no start/stop, no autoloop control, no agent stdout tail
