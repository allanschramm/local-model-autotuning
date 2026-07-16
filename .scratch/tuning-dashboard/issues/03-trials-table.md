# 03 — Tabela de Trials (últimos 50)

**What to build:** The dashboard lists the last 50 Trials from `results.tsv` (newest first) with the minimal operator columns, including `diagnostic`, so the human can see keeps/fails and why a Trial failed without opening the TSV.

**Blocked by:** 01 — Shell do dashboard localhost

**Status:** ready-for-agent

- [ ] Table shows the last 50 rows from `results.tsv`, newest on top
- [ ] Columns include at least: status, outcome, val_score, tps, memory_gb, elapsed_sec, diagnostic, description
- [ ] Missing/empty `results.tsv` shows a clear empty state in pt-BR
- [ ] Poll refreshes the table; read-only (append-only TSV never written by UI)
