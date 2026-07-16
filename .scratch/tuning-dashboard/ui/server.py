#!/usr/bin/env python3
"""Dashboard shell server for localhost 18765."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Dashboard</title></head>
<body>
<h1>Dashboard</h1>
<p id="status">Carregando status...</p>
<script>
  const poll = () => {
    fetch('/api/status')
      .then(r => r.json())
      .then(d => document.getElementById('status').textContent = JSON.stringify(d))
      .catch(() => {});
  };
  poll();
  setInterval(poll, 2500);
</script>
</body></html>"""
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    server = HTTPServer(("127.0.0.1", 18765), DashboardHandler)
    print("Serving dashboard at http://127.0.0.1:18765")
    server.serve_forever()


if __name__ == "__main__":
    main()
