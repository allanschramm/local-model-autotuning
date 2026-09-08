#!/usr/bin/env python3
"""Dashboard shell server for localhost 18765 (AILOCAL redesign)."""

from __future__ import annotations

import importlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .run_log import run_state_and_tail
from .trial_reader import format_trial_for_ui, read_last_50_trials

# ── Paths ──────────────────────────────────────────────────────────────────

_UI_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _UI_DIR / "static"

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

# Engine keys shown as stat tiles in the Baseline panel.
_STAT_KEYS = (
    "MODEL",
    "CTX_SIZE",
    "KV_CACHE_K",
    "KV_CACHE_V",
    "THREADS",
    "THREADS_BATCH",
    "N_GPU_LAYERS",
    "N_CPU_MOE",
    "SPEC_DRAFT_N_MAX",
    "TPS_FLOOR",
)


# ── Baseline loader ────────────────────────────────────────────────────────


def _load_baseline() -> dict[str, Any]:
    """Read live Baseline from config.py (ENGINE + SAMPLER), never state JSON."""
    try:
        spec = importlib.util.spec_from_file_location(
            "config", str(_UI_DIR.parent / "autoresearch" / "core" / "config.py")
        )
        if spec is None or spec.loader is None:
            return {"error": "Baseline: Nenhum dado encontrado."}
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        return {"error": "Baseline: Nenhum dado encontrado."}
    engine = getattr(mod, "ENGINE_DEFAULTS", {})
    sampler = getattr(mod, "SAMPLER_DEFAULTS", {})
    return {**engine, **sampler}


# ── HTML shell ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Dashboard — Autotuning</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>

<!-- ── Fixed Frosted Header ──────────────────────────────────────────── -->
<header id="header">
  <span id="wordmark">AUTO<span>TUNING</span></span>
  <span id="header-model">—</span>
  <span id="header-spacer"></span>
  <span id="run-badge"><span class="dot"></span><span id="run-label">—</span></span>
  <span id="freshness">atualizado há —s</span>
</header>

<noscript><div id="stale-banner">JavaScript desabilitado — dados não atualizam.</div></noscript>

<!-- ── Stale Data Banner ─────────────────────────────────────────────── -->
<div id="stale-banner">dados obsoletos — falha na conexão.</div>

<!-- ── Main Grid ─────────────────────────────────────────────────────── -->
<div id="main">

  <!-- Baseline Rail -->
  <section id="baseline-section" class="card">
    <h2>Baseline</h2>
    <div id="baseline-stats"></div>
    <div id="baseline-sampler" class="sampler-group" hidden>
      <div class="label">SAMPLER</div>
      <div id="baseline-chips"></div>
    </div>
    <div id="baseline-details"></div>
    <p id="baseline-empty" class="baseline-empty" hidden></p>
  </section>

  <!-- Trials Table -->
  <section id="trials-section" class="card">
    <h2>Últimos Trials</h2>
    <p id="trials-empty" class="baseline-empty" hidden></p>
    <table id="trials-table" hidden>
      <thead>
        <tr>
          <th class="status-cell">Status</th>
          <th>Outcome</th>
          <th class="num">ctx</th>
          <th class="num">TPS</th>
          <th class="num">agentic</th>
          <th class="num">coding</th>
          <th class="num">memory</th>
          <th class="num">elapsed</th>
          <th>Descrição</th>
        </tr>
      </thead>
      <tbody id="trials-body"></tbody>
    </table>
  </section>

  <!-- Log Panel -->
  <section id="log-section" class="card">
    <h2>Log do servidor (Trial)</h2>
    <div id="log-toolbar">
      <button id="pin-toggle" title="Fixar/Desfixar scroll">📌 Fixar</button>
    </div>
    <p id="log-empty" class="baseline-empty" hidden></p>
    <pre id="log-tail"></pre>
  </section>

</div>

<script>
(function() {
  "use strict";

  /* ── Elements ─────────────────────────────────────────────────────── */
  const $ = id => document.getElementById(id);
  const headerModel   = $("header-model");
  const runBadge      = $("run-badge");
  const runLabel      = $("run-label");
  const freshness     = $("freshness");
  const staleBanner   = $("stale-banner");
  const baselineStats = $("baseline-stats");
  const baselineSampler = $("baseline-sampler");
  const baselineChips = $("baseline-chips");
  const baselineDetails = $("baseline-details");
  const baselineEmpty = $("baseline-empty");
  const trialsEmpty   = $("trials-empty");
  const trialsTable   = $("trials-table");
  const trialsBody    = $("trials-body");
  const logEmpty      = $("log-empty");
  const logTail       = $("log-tail");
  const pinToggle     = $("pin-toggle");

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  let pinned = false;
  let lastLogLen = 0;
  let pollCount = 0;
  let lastPollTime = Date.now();

  /* ── Run-State Badge ──────────────────────────────────────────────── */
  const renderRunState = (d) => {
    const state = d.run_state || "Idle";
    runLabel.textContent = state;
    runBadge.className = state === "Em execução" ? "running" :
                         (d._stale ? "stale" : "idle");
  };

  /* ── Baseline Panel ───────────────────────────────────────────────── */
  const renderBaseline = (d) => {
    if (d.error) {
      baselineEmpty.hidden = false;
      baselineEmpty.textContent = d.error;
      return;
    }
    baselineEmpty.hidden = true;
    const bg = d.baseline || {};
    if (bg.error) {
      baselineEmpty.hidden = false;
      baselineEmpty.textContent = bg.error;
      baselineStats.innerHTML = "";
      baselineChips.innerHTML = "";
      baselineDetails.innerHTML = "";
      baselineSampler.hidden = true;
      return;
    }

    // Stat tiles for critical keys
    baselineStats.innerHTML = "";
    for (const key of ["MODEL","CTX_SIZE","KV_CACHE_K","KV_CACHE_V",
                       "THREADS","THREADS_BATCH","N_GPU_LAYERS",
                       "N_CPU_MOE","SPEC_DRAFT_N_MAX","TPS_FLOOR"]) {
      const val = bg[key];
      if (val == null) continue;
      const tile = document.createElement("div");
      tile.className = "stat-tile";
      tile.innerHTML = `<div class="label">${key}</div><div class="value">${esc(String(val))}</div>`;
      baselineStats.appendChild(tile);
    }

    // Sampler chips
    baselineChips.innerHTML = "";
    let hasSampler = false;
    for (const key of ["TEMP","TOP_P","TOP_K","MINT_P","XTC_PROBA",
                        "XTC_THRESHOLD","REPETITION_PENALTY",
                        "DRY_MULT","DRY_RATIO","DRY_LAST_N"]) {
      const val = bg[key];
      if (val == null) continue;
      hasSampler = true;
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${key}=${esc(String(val))}`;
      baselineChips.appendChild(chip);
    }
    baselineSampler.hidden = !hasSampler;

    // Remaining ENGINE keys in a <details> disclosure
    const allKeys = new Set([...Object.keys(bg)]);
    const shown = new Set([...["MODEL","CTX_SIZE","KV_CACHE_K","KV_CACHE_V",
      "THREADS","THREADS_BATCH","N_GPU_LAYERS","N_CPU_MOE",
      "SPEC_DRAFT_N_MAX","TPS_FLOOR"], ...["TEMP","TOP_P","TOP_K","MINT_P",
      "XTC_PROBA","XTC_THRESHOLD","REPETITION_PENALTY","DRY_MULT",
      "DRY_RATIO","DRY_LAST_N"]]);
    const remaining = [...allKeys].filter(k => !shown.has(k) && k !== "error");
    if (remaining.length > 0) {
      let rows = "";
      for (const key of remaining) {
        rows += `<dt>${esc(key)}</dt><dd>${esc(String(bg[key]))}</dd>`;
      }
      baselineDetails.innerHTML = `<details><summary>ver todas</summary><dl>${rows}</dl></details>`;
    } else {
      baselineDetails.innerHTML = "";
    }

    // Model in header
    if (bg.MODEL) headerModel.textContent = bg.MODEL;
  };

  /* ── Trials Table ─────────────────────────────────────────────────── */
  const cell = (text) => {
    const td = document.createElement("td");
    td.textContent = text == null || text === "" ? "—" : String(text);
    return td;
  };
  const numCell = (text) => {
    const td = document.createElement("td");
    td.className = "num";
    td.textContent = text == null || text === "" ? "—" : String(text);
    return td;
  };

  const renderTrials = (d) => {
    trialsBody.innerHTML = "";
    if (d.error) {
      trialsEmpty.hidden = false;
      trialsEmpty.textContent = d.error;
      trialsTable.hidden = true;
      return;
    }
    const trials = d.trials || [];
    // Per-panel error from server fallback (one feed failed, others still render).
    if (trials.some((t) => t.error)) {
      trialsEmpty.hidden = false;
      trialsEmpty.textContent = trials.find((t) => t.error).error;
      trialsTable.hidden = true;
      return;
    }
    if (trials.length === 0) {
      trialsEmpty.hidden = false;
      trialsEmpty.textContent = "Nenhum dado de Trial encontrado.";
      trialsTable.hidden = true;
      return;
    }
    trialsEmpty.hidden = true;
    trialsTable.hidden = false;
    for (const t of trials) {
      const tr = document.createElement("tr");

      // Status pill (pt-BR)
      const sc = document.createElement("td");
      sc.className = "status-cell";
      const pill = document.createElement("span");
      pill.className = "pill";
      const st = t.status_pt || t.status || "";
      const cls = st === "na fronteira" ? "on-front" :
                  st === "dominado" ? "dominated" :
                  st === "incompleto" ? "incomplete" :
                  st === "rejeitado" ? "rejected" : "";
      pill.className += " " + cls;
      pill.textContent = st;
      sc.appendChild(pill);
      tr.appendChild(sc);

      // Outcome (tooltip gets diagnostic)
      const oc = document.createElement("td");
      oc.className = "outcome-cell";
      oc.textContent = t.outcome || "—";
      if (t.diagnostic) oc.title = t.diagnostic;
      tr.appendChild(oc);

      // Numeric columns
      tr.appendChild(numCell(t.ctx));
      tr.appendChild(numCell(t.tps));
      tr.appendChild(numCell(t.agentic));
      tr.appendChild(numCell(t.coding));
      tr.appendChild(numCell(t.memory));
      tr.appendChild(numCell(t.elapsed));

      // Description (truncated + tooltip)
      const dc = document.createElement("td");
      dc.className = "desc";
      dc.textContent = t.description || "";
      if (t.description) dc.title = t.description;
      tr.appendChild(dc);

      trialsBody.appendChild(tr);
    }
  };

  /* ── Log Panel (smart follow + pin) ───────────────────────────────── */
  const atBottom = (el) => {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 4;
  };

  const renderLog = (d) => {
    if (d.log_tail == null || d.log_tail === "") {
      logEmpty.hidden = false;
      logEmpty.textContent = "Log do servidor: nenhum arquivo encontrado.";
      logTail.textContent = "";
      return;
    }
    logEmpty.hidden = true;
    logTail.textContent = d.log_tail;
    // Auto-follow only when not pinned and already at bottom
    if (!pinned && atBottom(logTail)) {
      logTail.scrollTop = logTail.scrollHeight;
    }
    lastLogLen = d.log_tail.length;
  };

  /* ── Pin Toggle ───────────────────────────────────────────────────── */
  pinToggle.addEventListener("click", () => {
    pinned = !pinned;
    pinToggle.textContent = pinned ? "📌 Soltar" : "📌 Fixar";
    pinToggle.classList.toggle("pinned", pinned);
    // If unpinned and there's content, scroll to bottom
    if (!pinned && logTail.textContent) {
      logTail.scrollTop = logTail.scrollHeight;
    }
  });

  /* ── Freshness Timer ──────────────────────────────────────────────── */
  const updateFreshness = () => {
    const elapsed = Math.floor((Date.now() - lastPollTime) / 1000);
    freshness.textContent = elapsed < 30
      ? `atualizado há ${elapsed}s`
      : `atualizado há ${Math.floor(elapsed/60)}m`;
  };

  /* ── Poll ─────────────────────────────────────────────────────────── */
  const poll = () => {
    fetch("/api/status")
      .then(r => r.json())
      .then(d => {
        pollCount++;
        lastPollTime = Date.now();
        staleBanner.classList.remove("visible");
        d._stale = false;
        renderRunState(d);
        renderBaseline(d);
        renderTrials(d);
        renderLog(d);
      })
      .catch(() => {
        staleBanner.classList.add("visible");
        runBadge.className = "stale";
        runLabel.textContent = "Idle";
        baselineEmpty.hidden = false;
        baselineEmpty.textContent = "Erro ao carregar Baseline.";
        trialsEmpty.hidden = false;
        trialsEmpty.textContent = "Erro ao carregar Trials.";
        logEmpty.hidden = false;
        logEmpty.textContent = "Erro ao carregar log.";
        trialsTable.hidden = true;
        baselineStats.innerHTML = "";
        trialsBody.innerHTML = "";
        logTail.textContent = "";
      });
  };

  // Initial poll + interval
  poll();
  setInterval(poll, 2500);
  setInterval(updateFreshness, 1000);

})();
</script>
</body></html>"""


def _esc(text: str) -> str:
    """Minimal HTML escape for inline JS attribute values."""
    return (
        text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ── Static asset serving (exact-name allowlist) ─────────────────────────────

# Only these relative names under ui/static/ are served (ADR 0011: CSS +
# bundled OFL fonts). Anything else → 404. Content types come from this fixed
# table, never from request data.
_STATIC_FILES: dict[str, str] = {
    "style.css": "text/css",
    "fonts/Inter-Regular.woff2": "font/woff2",
    "fonts/Inter-Bold.woff2": "font/woff2",
    "fonts/JetBrainsMono-Regular.woff2": "font/woff2",
    "fonts/JetBrainsMono-Bold.woff2": "font/woff2",
}


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler: serves HTML shell, /api/status JSON, and static assets."""

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(_HTML)
        elif self.path == "/api/status":
            self._serve_status()
        elif self.path.startswith("/static/"):
            self._serve_static()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_html(self, html: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_status(self) -> None:
        baseline = {}
        trials = []
        log_tail = None
        run_state = "Idle"
        try:
            run_state, log_tail = run_state_and_tail()
        except Exception:  # noqa: BLE001
            pass
        try:
            baseline = _load_baseline()
        except Exception:  # noqa: BLE001
            baseline = {"error": "Falha ao carregar Baseline."}
        try:
            trials = [format_trial_for_ui(t) for t in read_last_50_trials()]
        except Exception:  # noqa: BLE001
            trials = [{"error": "Falha ao carregar Trials."}]
        payload = {
            "run_state": run_state,
            "log_tail": log_tail,
            "baseline": baseline,
            "trials": trials,
        }
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self) -> None:
        """Serve files from ui/static/ (CSS, fonts)."""
        rel = self.path[len("/static/") :].lstrip("/").replace("\\", "/")
        # Iterate the allowlist so the filesystem path is built from the
        # constant key, never from the user-supplied string.
        file_path = None
        content_type = None
        for name, ct in _STATIC_FILES.items():
            if name == rel:
                file_path = _STATIC_DIR / name
                content_type = ct
                break
        if file_path is None or not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 18765), DashboardHandler)
    print("Serving dashboard at http://127.0.0.1:18765")
    server.serve_forever()


if __name__ == "__main__":
    main()
