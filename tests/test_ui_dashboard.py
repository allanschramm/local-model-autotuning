"""HTTP seam + pt-BR localization for the dashboard (issue #46).

Verifies the live ``/api/status`` endpoint localizes Trial Status server-side:
the canonical English ``status`` stays intact in the payload while a new
``status_pt`` presentation field carries the pt-BR label (ADR 0006 / CONTEXT.md).
"""

from __future__ import annotations

import csv
import json
import threading
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from autoresearch.runners import run as run_mod
from ui import trial_reader

_HEADER = run_mod.CATEGORY_FIELDNAMES

# (canonical status, pt-BR label, pill color) — one entry per ADR 0006 status.
_STATUS_PT: list[tuple[str, str, str]] = [
    ("on_front", "na fronteira", "#0a7a2f"),
    ("dominated", "dominado", "#666666"),
    ("incomplete", "incompleto", "#b8860b"),
    ("rejected", "rejeitado", "#d70018"),
]


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows() -> list[dict[str, str]]:
    """One row per canonical status with the columns the UI renders."""
    rows = []
    for status, _label, _color in _STATUS_PT:
        rows.append(
            {
                "outcome": "OK" if status != "rejected" else "MODEL_REJECTED",
                "diagnostic": f"{status} diagnostic",
                "status": status,
                "agentic": "0.8000",
                "coding": "0.650000",
                "memory_gb": "6.2",
                "elapsed_sec": "180",
                "tps": "42.5",
                "ctx": "32768",
                "description": f"Ornith 1.5 35B ctx32k [{status}]",
            }
        )
    return rows


@contextmanager
def _live_server(tmp_path: Path, rows: list[dict[str, str]]) -> str:
    """Serve the real DashboardHandler with a temporary results.tsv."""
    from http.server import ThreadingHTTPServer

    from ui.server import DashboardHandler

    results = tmp_path / "results.tsv"
    _write_results(results, rows)
    with mock.patch.object(run_mod, "RESULTS_FILE", results):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()


def _get_json(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/status", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_html(base: str) -> str:
    with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
        return resp.read().decode("utf-8")


def test_api_status_returns_status_pt_for_on_front(tmp_path):
    """HTTP seam: an on_front row reports status_pt "na fronteira" with the
    canonical English status intact in the /api/status payload."""
    with _live_server(tmp_path, _sample_rows()) as base:
        payload = _get_json(base)

    trials = {t["status"]: t for t in payload["trials"]}
    on_front = trials["on_front"]

    assert on_front["status_pt"] == "na fronteira"
    assert on_front["status_color"] == "#0a7a2f"
    # Canonical English label is the source of truth and stays untouched.
    assert on_front["status"] == "on_front"


@pytest.mark.parametrize(("canonical", "label", "color"), _STATUS_PT)
def test_status_pt_localizes_every_canonical_status(tmp_path, canonical, label, color):
    """Every canonical ADR 0006 status maps to its pt-BR pill label + color,
    and the canonical English label is preserved unchanged."""
    with _live_server(tmp_path, _sample_rows()) as base:
        payload = _get_json(base)

    by_status = {t["status"]: t for t in payload["trials"]}
    assert by_status[canonical]["status_pt"] == label
    assert by_status[canonical]["status_color"] == color
    # Canonical status is never mutated by the pt-BR presentation layer.
    assert by_status[canonical]["status"] == canonical


def test_status_pt_helper_falls_back_to_raw_status():
    """Pure helper: unknown/empty statuses fall back to the raw value, never
    lose the label."""
    assert trial_reader.status_pt("on_front") == "na fronteira"
    assert trial_reader.status_pt("bogus") == "bogus"
    assert trial_reader.status_pt("") == "—"
    assert trial_reader.status_pt(None) == "—"


def test_canonical_status_survives_localization_helper():
    """format_trial_for_ui keeps the canonical status and adds status_pt."""
    formatted = trial_reader.format_trial_for_ui({"status": "on_front"})
    assert formatted["status"] == "on_front"
    assert formatted["status_pt"] == "na fronteira"


def test_empty_state_stays_pt_br(tmp_path):
    """The dashboard empty state renders in pt-BR."""
    with _live_server(tmp_path, []) as base:
        html = _get_html(base)
        payload = _get_json(base)

    assert payload["trials"] == []
    assert "Nenhum dado de Trial encontrado." in html
