"""ASGI/network regression: private URLs, DNS rebinding, redirects, TLS and leaks.

Failure modes: private/numeric/mapped IP bypasses; mixed DNS answers; a second
DNS resolution after validation; redirected private destinations; proxy bypass;
unverified TLS; unbounded bodies; script/style text or exception details leaked.
All network traffic is intercepted or reaches only this test's loopback fixture.
"""

import argparse
import importlib.util
import io
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import urllib3
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
HTML = b"<title>Fixture title</title><p>Visible text</p><script>script-secret</script ><style>style-secret</style >"
SECRET = "fixture-internal-secret"


class Fixture(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.calls.append((self.path, self.headers.get("Host")))
        redirect = {
            "/private": f"http://127.0.0.1:{self.server.server_port}/ok",
            "/public": "http://other.fixture.test/ok",
            "/loop": "http://public.fixture.test/loop",
        }.get(self.path)
        self.send_response(302 if redirect else 200)
        if redirect:
            self.send_header("Location", redirect)
        self.end_headers()
        self.wfile.write(b"x" * (2 * 1024 * 1024 + 1) if self.path == "/large" else HTML)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="claw-web-security-"))
    output.mkdir(parents=True, exist_ok=True)
    os.environ["WEB_REAL_CACHE_DIR"] = str(output / "cache")
    sys.path[:0] = [str(ROOT), str(ROOT / "packs/claw-tiers/upstream")]
    adapter = ROOT / "claw_web.py"
    spec = importlib.util.spec_from_file_location("claw_web_under_test", adapter)
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    server = HTTPServer(("127.0.0.1", 0), Fixture)
    server.calls = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cases = []
    native_dns = socket.getaddrinfo
    native_connect = urllib3.util.connection.create_connection
    resolved = []
    connected = []

    def dns(host, port, *a, **kw):
        if host in ("public.fixture.test", "other.fixture.test"):
            resolved.append(host)
            ip = "8.8.8.8" if host == "public.fixture.test" else "8.8.4.4"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
        return native_dns(host, port, *a, **kw)

    def connect(address, *a, **kw):
        connected.append(address)
        assert address[0] in ("8.8.8.8", "8.8.4.4"), address
        return native_connect(("127.0.0.1", server.server_port), *a, **kw)

    with TestClient(web.app) as client:

        def fetch(url):
            web.reset_state()
            return client.post("/web/fetch", json={"url": url}).json()

        def case(name, check):
            try:
                check()
                cases.append({"case": name, "passed": True})
            except Exception as exc:
                cases.append({"case": name, "passed": False, "error": str(exc)})

        def blocked(url):
            # Old httpx path gets inert HTML; any new socket is an immediate failure.
            with (
                patch.object(httpx.Client, "get", return_value=httpx.Response(200, content=HTML)),
                patch(
                    "urllib3.util.connection.create_connection",
                    side_effect=AssertionError("Unexpected socket"),
                ) as network,
            ):
                result = fetch(url)
            assert result.get("status_code") == 400, result
            assert network.call_count == 0

        for url in (
            "http://127.0.0.1/",
            "http://localhost/",
            "http://10.0.0.1/",
            "http://169.254.169.254/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[64:ff9b::7f00:1]/",
            "http://[2002:7f00:1::]/",
            "http://[2001:0000:4136:e378:8000:63bf:3fff:fdd2]/",
            "http://2130706433/",
            "http://0x7f000001/",
            "http://100.64.0.1/",
            "http://224.0.0.1/",
            "file:///etc/passwd",
            "ftp://public.fixture.test/",
            "http://name:password@public.fixture.test/",
            "http://@public.fixture.test/",
            "http://public.fixture.test:8080/",
            "http://public.fixture.test:0/",
        ):
            case("blocked " + url, lambda url=url: blocked(url))

        for label, ips in (("private DNS", ["127.0.0.1"]), ("mixed DNS", ["8.8.8.8", "127.0.0.1"])):
            records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in ips]

            def check(records=records):
                with patch("socket.getaddrinfo", return_value=records):
                    blocked("http://public.fixture.test/")

            case(label, check)

        def public(path, expected):
            resolved.clear()
            connected.clear()
            with (
                patch("socket.getaddrinfo", side_effect=dns),
                patch("urllib3.util.connection.create_connection", side_effect=connect),
                patch.dict(
                    os.environ,
                    {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"},
                ),
            ):
                result = fetch("http://public.fixture.test" + path)
            assert result.get("status_code") == expected, result
            assert len(connected) == len(resolved), (connected, resolved)
            assert all(host.endswith(".fixture.test") for host in resolved)
            if expected == 200:
                assert "Visible text" in result["content"], result
                assert (
                    "script-secret" not in result["content"]
                    and "style-secret" not in result["content"]
                ), result
                assert result["title"] == "Fixture title"
            if path == "/private":
                assert len(connected) == 1
            if path == "/public":
                assert [a[0] for a in connected] == ["8.8.8.8", "8.8.4.4"]
            assert len(connected) <= 6

        case("pinned public fetch and HTML parser", lambda: public("/ok", 200))
        case("public redirect revalidated", lambda: public("/public", 200))
        case("private redirect blocked", lambda: public("/private", 400))
        case("redirect loop bounded", lambda: public("/loop", 400))
        case("response size bounded", lambda: public("/large", 502))

        def tls():
            response = urllib3.response.HTTPResponse(
                body=io.BytesIO(HTML), status=200, preload_content=False
            )
            with (
                patch("socket.getaddrinfo", side_effect=dns),
                patch("urllib3.HTTPSConnectionPool") as pool,
                patch.object(httpx.Client, "get", return_value=httpx.Response(200, content=HTML)),
            ):
                pool.return_value.__enter__.return_value.urlopen.return_value = response
                result = fetch("https://public.fixture.test/ok")
            assert result.get("status_code") == 200, result
            assert pool.call_args.args[0] == "8.8.8.8"
            assert pool.call_args.kwargs["assert_hostname"] == "public.fixture.test"
            assert pool.call_args.kwargs["server_hostname"] == "public.fixture.test"
            assert pool.call_args.kwargs["cert_reqs"] == "CERT_REQUIRED"

        case("TLS hostname and certificate verification", tls)

        def errors():
            def fail(*_a, **_kw):
                raise RuntimeError(SECRET)

            helper = types.SimpleNamespace(search_serp=fail)
            with patch.dict(sys.modules, {"search_serp": helper}):
                web.reset_state()
                result = client.post("/web/search", json={"query": "fixture"}).json()
            assert SECRET not in json.dumps(result), result
            with (
                patch("socket.getaddrinfo", side_effect=dns),
                patch("urllib3.util.connection.create_connection", side_effect=OSError(SECRET)),
                patch.object(httpx.Client, "get", side_effect=OSError(SECRET)),
            ):
                result = fetch("http://public.fixture.test/error")
            assert SECRET not in json.dumps(result), result
            assert result.get("status_code") == 502, result

        case("internal exceptions withheld", errors)

    server.shutdown()
    thread.join()
    server.server_close()
    bridge = subprocess.run(
        [
            sys.executable,
            str(ROOT / "claw_bridge.py"),
            str(ROOT / "packs/claw-tiers"),
            "T046_cve_research",
            "live",
        ],
        input=json.dumps(
            {"op": "call", "name": "web_fetch", "arguments": {"url": "http://127.0.0.1/"}}
        )
        + "\n",
        cwd=output,
        env={
            **{
                k: v
                for k, v in os.environ.items()
                if k.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
            },
            "SERP_DEV_KEY": SECRET,
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    messages = [json.loads(line) for line in bridge.stdout.splitlines()]
    passed = (
        bridge.returncode == 0
        and len(messages) == 2
        and messages[1].get("result", {}).get("result", {}).get("status_code") == 400
        and SECRET not in bridge.stdout + bridge.stderr
    )
    cases.append(
        {"case": "real bridge rejects private fetch without credential leakage", "passed": passed}
    )
    (output / "bridge.log").write_text(bridge.stdout + bridge.stderr, encoding="utf-8")
    with sqlite3.connect(output / "results.db") as db:
        db.execute(
            "CREATE TABLE security_cases (name TEXT PRIMARY KEY, passed INTEGER, diagnostic TEXT)"
        )
        db.executemany(
            "INSERT INTO security_cases VALUES (?,?,?)",
            [(c["case"], c["passed"], c.get("error")) for c in cases],
        )
    summary = {"status": "passed" if all(c["passed"] for c in cases) else "failed", "cases": cases}
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cases": len(cases),
                "failures": [c for c in cases if not c["passed"]],
                "artifacts": str(output),
            }
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
