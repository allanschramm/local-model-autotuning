"""Hidden fail-to-pass tests for issue-43 static route."""

from __future__ import annotations

import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import app as server


def _get(url: str) -> tuple[int, str, str]:
    try:
        with urlopen(url, timeout=2.0) as resp:
            body = resp.read().decode("utf-8")
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, body, ctype
    except HTTPError as exc:
        return exc.code, "", exc.headers.get("Content-Type", "")


def test_index_and_static_css():
    httpd = server.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        status, body, _ = _get(f"http://127.0.0.1:{port}/")
        assert status == 200
        assert 'lang="pt-BR"' in body
        status, css, ctype = _get(f"http://127.0.0.1:{port}/static/app.css")
        assert status == 200
        assert "text/css" in ctype
        assert "body" in css
        status, _, _ = _get(f"http://127.0.0.1:{port}/no-such-path")
        assert status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
