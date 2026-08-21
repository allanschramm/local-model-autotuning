#!/usr/bin/env python3
"""Dashboard shell server for localhost 18765."""

from __future__ import annotations

import importlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .run_log import run_state_and_tail
from .trial_reader import format_trial_for_ui, read_last_50_trials

# Baseline engine keys to surface on the live panel (#24).
_ENGINE_KEYS = (
    "MODEL",
    "CTX_SIZE",
    "KV_CACHE",
    "KV_CACHE_K",
    "KV_CACHE_V",
    "THREADS",
    "THREADS_BATCH",
    "BATCH_SIZE",
    "UBATCH_SIZE",
    "FLASH_ATTN",
    "SPEC_DRAFT_N_MAX",
    "SPEC_DRAFT_MODEL",
    "N_GPU_LAYERS",
    "NUMA",
    "N_CPU_MOE",
    "VRAM_LIMIT_MB",
    "VRAM_HEADROOM_MB",
    "HOST_MEMORY_HEADROOM_MB",
    "TPS_FLOOR",
)


def _load_baseline() -> dict[str, Any]:
    """Read live Baseline from config.py (ENGINE + SAMPLER), never state JSON."""
    config_module = importlib.import_module("autoresearch.core.config")
    config_module = importlib.reload(config_module)
    engine = getattr(config_module, "ENGINE_DEFAULTS", {}) or {}
    sampler = getattr(config_module, "SAMPLER_DEFAULTS", {}) or {}
    baseline: dict[str, Any] = {key: engine.get(key) for key in _ENGINE_KEYS}
    baseline["SAMPLER_DEFAULTS"] = dict(sampler)
    return baseline


_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }
  thead th { position: sticky; top: 0; z-index: 1; background: #f4f4f4; }
  tbody tr:nth-child(even) { background: #fafafa; }
  td.num, th.num { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px; color: #fff; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
  .trunc { max-width: 18rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; }
  dt { font-weight: 600; }
  .empty { color: #666; }
  #run-state { font-weight: 700; }
  #run-state.running { color: #0a7a2f; }
  #run-state.idle { color: #666; }
  #log-tail {
    background: #111; color: #ddd; padding: 0.75rem; max-height: 20rem;
    overflow: auto; white-space: pre-wrap; font-size: 0.8rem;
  }
</style>
</head>
<body>
<h1>Dashboard</h1>
<p>Estado: <span id="run-state" class="idle">—</span></p>
<p id="status">Carregando status...</p>
<section>
  <h2>Baseline</h2>
  <p id="baseline-empty" class="empty" hidden></p>
  <dl id="baseline"></dl>
</section>
<section>
  <h2>Últimos Trials</h2>
  <p id="trials-empty" class="empty" hidden></p>
  <table id="trials-table" hidden>
    <thead>
      <tr>
        <th>Status</th><th>Outcome</th><th class="num">ctx</th><th class="num">TPS</th>
        <th class="num">agentic</th><th class="num">coding</th><th class="num">memory</th><th class="num">elapsed</th>
        <th>description</th>
      </tr>
    </thead>
    <tbody id="trials-body"></tbody>
  </table>
</section>
<section>
  <h2>Log do servidor (Trial)</h2>
  <p id="log-empty" class="empty" hidden></p>
  <pre id="log-tail"></pre>
</section>
<script>
  const statusEl = document.getElementById('status');
  const runStateEl = document.getElementById('run-state');
  const baselineEl = document.getElementById('baseline');
  const baselineEmpty = document.getElementById('baseline-empty');
  const trialsEmpty = document.getElementById('trials-empty');
  const trialsTable = document.getElementById('trials-table');
  const trialsBody = document.getElementById('trials-body');
  const logEmpty = document.getElementById('log-empty');
  const logTail = document.getElementById('log-tail');

  const renderRunState = (d) => {
    const state = d.run_state || 'Idle';
    runStateEl.textContent = state;
    runStateEl.className = state === 'Em execução' ? 'running' : 'idle';
  };

  const renderLog = (d) => {
    if (d.log_tail == null || d.log_tail === '') {
      logEmpty.hidden = false;
      logEmpty.textContent = 'Log do servidor: nenhum arquivo encontrado.';
      logTail.textContent = '';
      return;
    }
    logEmpty.hidden = true;
    logTail.textContent = d.log_tail;
  };

  const renderBaseline = (d) => {
    baselineEl.innerHTML = '';
    if (d.error) {
      baselineEmpty.hidden = false;
      baselineEmpty.textContent = d.error;
      return;
    }
    const baseline = d.baseline || {};
    const keys = Object.keys(baseline);
    if (keys.length === 0) {
      baselineEmpty.hidden = false;
      baselineEmpty.textContent = 'Baseline: Nenhum dado encontrado.';
      return;
    }
    baselineEmpty.hidden = true;
    for (const key of keys) {
      const dt = document.createElement('dt');
      dt.textContent = key;
      const dd = document.createElement('dd');
      const value = baseline[key];
      dd.textContent = (value !== null && typeof value === 'object')
        ? JSON.stringify(value)
        : String(value);
      baselineEl.appendChild(dt);
      baselineEl.appendChild(dd);
    }
  };

  const textCell = (text) => {
    const td = document.createElement('td');
    td.textContent = text == null || text === '' ? '—' : String(text);
    return td;
  };

  const numCell = (text) => {
    const td = document.createElement('td');
    td.className = 'num';
    td.textContent = text == null || text === '' ? '—' : String(text);
    return td;
  };

  const statusCell = (t) => {
    const td = document.createElement('td');
    const label = t.status_pt || (t.status ? t.status : '—');
    const span = document.createElement('span');
    span.className = 'pill';
    span.style.background = t.status_color || '#999999';
    span.textContent = label;
    td.appendChild(span);
    return td;
  };

  const renderTrials = (d) => {
    trialsBody.innerHTML = '';
    if (d.error) {
      trialsEmpty.hidden = false;
      trialsEmpty.textContent = d.error;
      trialsTable.hidden = true;
      return;
    }
    const trials = d.trials || [];
    if (trials.length === 0) {
      trialsEmpty.hidden = false;
      trialsEmpty.textContent = 'Nenhum dado de Trial encontrado.';
      trialsTable.hidden = true;
      return;
    }
    trialsEmpty.hidden = true;
    trialsTable.hidden = false;
    for (const t of trials) {
      const tr = document.createElement('tr');
      tr.appendChild(statusCell(t));
      // Outcome carries the diagnostic as a tooltip.
      const outcome = textCell(t.outcome);
      outcome.className = 'trunc';
      if (t.diagnostic) {
        outcome.title = t.diagnostic;
      }
      tr.appendChild(outcome);
      for (const key of ['ctx','tps','agentic','coding','memory','elapsed']) {
        tr.appendChild(numCell(t[key]));
      }
      const desc = textCell(t.description);
      desc.className = 'trunc';
      if (t.description) {
        desc.title = t.description;
      }
      tr.appendChild(desc);
      trialsBody.appendChild(tr);
    }
  };

  const poll = () => {
    fetch('/api/status')
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          statusEl.textContent = d.error;
        } else {
          statusEl.textContent = 'Status: OK';
        }
        renderRunState(d);
        renderBaseline(d);
        renderTrials(d);
        renderLog(d);
      })
      .catch(() => {
        statusEl.textContent = 'Erro ao carregar status.';
        runStateEl.textContent = 'Idle';
        runStateEl.className = 'idle';
        baselineEmpty.hidden = false;
        baselineEmpty.textContent = 'Erro ao carregar Baseline.';
        trialsEmpty.hidden = false;
        trialsEmpty.textContent = 'Erro ao carregar Trials.';
        logEmpty.hidden = false;
        logEmpty.textContent = 'Erro ao carregar log.';
        trialsTable.hidden = true;
        baselineEl.innerHTML = '';
        trialsBody.innerHTML = '';
        logTail.textContent = '';
      });
  };
  poll();
  setInterval(poll, 2500);
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            try:
                run_state, log_tail = run_state_and_tail()
                payload = {
                    "run_state": run_state,
                    "log_tail": log_tail,
                    "baseline": _load_baseline(),
                    "trials": [format_trial_for_ui(t) for t in read_last_50_trials()],
                }
            except Exception as exc:  # noqa: BLE001 — surface any load failure to UI
                payload = {"error": f"Falha ao carregar status: {exc}", "run_state": "Idle"}
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 18765), DashboardHandler)
    print("Serving dashboard at http://127.0.0.1:18765")
    server.serve_forever()


if __name__ == "__main__":
    main()
