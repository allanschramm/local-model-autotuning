"""HTTP-level tests for the UI dashboard server (issue #42).

Tests the real DashboardHandler in a thread on an ephemeral port,
asserting over urllib.request — no mocking, no browser, no JS execution.
"""

from __future__ import annotations

import http.server
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ui.server import _STATIC_DIR, DashboardHandler

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
        for rel in (r"\\\\?\\C:\\Windows\\win.ini", r"\\localhost\\c$\\win.ini"):
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


def test_static_unlisted_file_returns_404():
    """Only allowlisted static names are served; any other file in the tree → 404."""
    extra = Path(_STATIC_DIR) / "notes.txt"
    extra.write_text("do not serve", encoding="utf-8")
    try:
        port, server, t = _start_server()
        try:
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/notes.txt")
                assert resp.status == 404, f"expected 404, got {resp.status}"
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
        finally:
            _stop_server(server, t)
    finally:
        extra.unlink(missing_ok=True)
