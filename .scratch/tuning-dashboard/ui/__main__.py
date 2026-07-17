"""Entry point for `python -m ui`."""

from server import DashboardHandler
from http.server import HTTPServer


def main() -> None:
    server = HTTPServer(("127.0.0.1", 18765), DashboardHandler)
    print("Serving dashboard at http://127.0.0.1:18765")
    server.serve_forever()


if __name__ == "__main__":
    main()
