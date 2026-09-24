"""Crash journal write/read/clear and autoloop consume helper."""

from __future__ import annotations

import csv

import autoloop
from autoresearch.core import crash_journal
from autoresearch.core.state import SearchState


def test_consume_writes_rejected(tmp_path, monkeypatch):
    journal = tmp_path / ".autoresearch_crash.journal"
    tsv = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(crash_journal, "JOURNAL_PATH", journal)
    monkeypatch.setattr(autoloop, "RESULTS_FILE", tsv)
    crash_journal.write_journal(
        {
            "model": "x.gguf",
            "config_key": "cfg-x",
            "config_json": '{"MODEL":"x.gguf"}',
        }
    )
    state = SearchState(state_path)
    autoloop.consume_crash_journal(state, retry=False, results_file=tsv)
    assert crash_journal.read_journal() is None
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
    assert rows[0]["status"] == "rejected"
    assert rows[0]["outcome"] == "CRASH"
    assert state.is_visited("cfg-x")


def test_consume_retry_clears_without_row(tmp_path, monkeypatch):
    journal = tmp_path / ".autoresearch_crash.journal"
    tsv = tmp_path / "results.tsv"
    monkeypatch.setattr(crash_journal, "JOURNAL_PATH", journal)
    crash_journal.write_journal({"model": "x.gguf", "config_key": "cfg-x"})
    state = SearchState(tmp_path / "state.json")
    autoloop.consume_crash_journal(state, retry=True, results_file=tsv)
    assert crash_journal.read_journal() is None
    assert not tsv.exists()
    assert not state.is_visited("cfg-x")
