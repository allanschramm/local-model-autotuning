"""HTTP-level tests for the UI dashboard server (issue #42).

Tests the real DashboardHandler in a thread on an ephemeral port,
asserting over urllib.request — no mocking, no browser, no JS execution.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
import unittest.mock
import urllib.error
import urllib.request

from ui.server import _HTML, DashboardHandler
from ui.trial_reader import status_pt

# ── Helpers ────────────────────────────────────────────────────────────────


def _start_server() -> tuple[int, http.server.HTTPServer, threading.Thread]:
    """Start DashboardHandler on a free port. Returns (port, server, thread)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.HTTPServer(("127.0.0.1", port), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return port, server, t


def _stop_server(server: http.server.HTTPServer, t: threading.Thread) -> None:
    server.shutdown()
    t.join(timeout=2)


# ── HTML Shell ─────────────────────────────────────────────────────────────


def test_root_returns_200_with_pt_br():
    """GET / → 200, lang=pt-BR, contains AUTOTUNING wordmark."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert 'lang="pt-BR"' in body
        assert "AUTOTUNING" in body or "AUTO" in body
        assert "Baseline" in body
        assert "Últimos Trials" in body
        assert "Log do servidor" in body
    finally:
        _stop_server(server, t)


def test_html_contains_static_css_link():
    """HTML shell references /static/style.css."""
    assert 'href="/static/style.css"' in _HTML


def test_html_contains_log_pin_button():
    """HTML shell has the log pin toggle button."""
    assert 'id="pin-toggle"' in _HTML


def test_html_contains_stale_banner():
    """HTML shell has the stale-data banner."""
    assert "stale-banner" in _HTML


# ── /api/status ────────────────────────────────────────────────────────────


def test_api_status_returns_json():
    """GET /api/status → 200 JSON with expected top-level keys."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "run_state" in data
        assert "log_tail" in data
        assert "baseline" in data
        assert "trials" in data
    finally:
        _stop_server(server, t)


def test_api_status_trial_rows_have_status_pt():
    """Seeded on_front row → status_pt "na fronteira", canonical status intact (#46)."""
    row = {
        "status": "on_front",
        "outcome": "ok",
        "ctx": "65536",
        "tps": "40.0",
        "agentic": "0.5",
        "coding": "0.5",
        "memory_gb": "7.0",
        "elapsed_sec": "120",
        "diagnostic": "",
        "description": "seeded row",
    }
    port, server, t = _start_server()
    try:
        with unittest.mock.patch("ui.server.read_last_50_trials", return_value=[row]):
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
            assert resp.status == 200
            data = json.loads(resp.read())
        trials = data["trials"]
        assert trials[0]["status"] == "on_front"
        assert trials[0]["status_pt"] == "na fronteira"
    finally:
        _stop_server(server, t)


def test_failed_baseline_feed_does_not_blank_other_panels():
    """Baseline failure yields its own pt-BR error while run_state/trials/log survive (#42)."""
    port, server, t = _start_server()
    try:
        with unittest.mock.patch("ui.server._load_baseline", side_effect=RuntimeError("boom")):
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status")
            assert resp.status == 200
            data = json.loads(resp.read())
        # Other feeds are untouched by the baseline failure...
        assert "run_state" in data
        assert "log_tail" in data
        assert isinstance(data.get("trials"), list)
        # ...and the failing panel reports its own pt-BR error.
        assert data["baseline"]["error"] == "Falha ao carregar Baseline."
    finally:
        _stop_server(server, t)


# ── Static Assets ──────────────────────────────────────────────────────────


def test_static_css_returns_200():
    """GET /static/style.css → 200, text/css, contains token #171717."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/style.css")
        assert resp.status == 200
        ct = resp.headers.get("Content-Type", "")
        assert "text/css" in ct
        body = resp.read().decode("utf-8")
        assert "#171717" in body
    finally:
        _stop_server(server, t)


def test_static_font_returns_200():
    """GET /static/fonts/Inter-Regular.woff2 → 200."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/fonts/Inter-Regular.woff2")
        assert resp.status == 200
    finally:
        _stop_server(server, t)


# ── Static path traversal (CodeQL py/path-injection, GHSA audit 2026-09) ──


def test_static_absolute_windows_path_returns_404():
    """GET /static/C:/Windows/win.ini → 404 (absolute path must not escape _STATIC_DIR)."""
    port, server, t = _start_server()
    try:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/C:/Windows/win.ini")
            assert resp.status == 404, f"expected 404, got {resp.status}"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_server(server, t)


def test_static_absolute_posix_style_path_returns_404():
    """GET /static//etc/passwd → 404 (double slash keeps an absolute remainder)."""
    port, server, t = _start_server()
    try:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static//etc/passwd")
            assert resp.status == 404, f"expected 404, got {resp.status}"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_server(server, t)


def test_static_unc_and_extended_length_paths_return_404():
    """GET /static/\\\\?\\C:\\... and UNC paths → 404."""
    port, server, t = _start_server()
    try:
        for rel in (r"\\\\?\\C:\\Windows\\win.ini", r"\\\\localhost\\c$\\win.ini"):
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/{rel}")
                assert resp.status == 404, f"{rel}: expected 404, got {resp.status}"
            except urllib.error.HTTPError as exc:
                assert exc.code == 404, f"{rel}: {exc.code}"
    finally:
        _stop_server(server, t)


def test_static_traversal_dotdot_returns_404():
    """GET /static/../../setup.py → 404 (urllib does not normalize client-side)."""
    port, server, t = _start_server()
    try:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/../../setup.py")
            assert resp.status == 404, f"expected 404, got {resp.status}"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_server(server, t)


def test_static_content_type_from_allowlist():
    """Content-Type comes from a fixed suffix allowlist, never raw path data."""
    port, server, t = _start_server()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/style.css")
        assert resp.headers.get("Content-Type", "").startswith("text/css")
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/fonts/Inter-Regular.woff2")
        ct = resp.headers.get("Content-Type", "")
        assert ct in ("font/woff2", "application/octet-stream"), ct
    finally:
        _stop_server(server, t)


# ── 404 ────────────────────────────────────────────────────────────────────


def test_unknown_path_returns_404():
    """GET /nope → 404."""
    port, server, t = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        _stop_server(server, t)


# ── status_pt helper ───────────────────────────────────────────────────────


def test_status_pt_maps_on_front():
    assert status_pt("on_front") == "na fronteira"


def test_status_pt_maps_dominated():
    assert status_pt("dominated") == "dominado"


def test_status_pt_maps_incomplete():
    assert status_pt("incomplete") == "incompleto"


def test_status_pt_maps_rejected():
    assert status_pt("rejected") == "rejeitado"


def test_status_pt_passes_unknown_through():
    assert status_pt("some_unknown") == "some_unknown"
