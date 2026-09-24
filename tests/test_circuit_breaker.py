"""Unit tests for the permanent RAM circuit breaker.

Pure-logic tests: memory readers and the kill helper are monkeypatched so no
real process is ever touched and no platform dependency is required.
"""

from __future__ import annotations

import threading

import pytest

from autoresearch.core import circuit_breaker as cb


def test_preflight_ok(monkeypatch):
    monkeypatch.setattr(cb, "free_ram_mb", lambda: 20_000)
    free = cb.preflight_ram(model_bytes=5 * 1024 * 1024, margin_mb=512)
    assert free == 20_000


def test_preflight_rejects_when_insufficient(monkeypatch):
    monkeypatch.setattr(cb, "free_ram_mb", lambda: 8_000)
    # model 10 GiB + workspace 1 GiB + margin 0.5 GiB = 11.5 GiB > 8 GiB free.
    with pytest.raises(cb.CircuitBreakerError, match="RAM_PREFLIGHT"):
        cb.preflight_ram(model_bytes=10 * 1024**3, margin_mb=512, workspace_mb=1024)


def test_preflight_unmeasurable_does_not_block(monkeypatch):
    monkeypatch.setattr(cb, "free_ram_mb", lambda: None)
    # Cannot measure -> do not block (runtime watchdog may be degraded too).
    assert cb.preflight_ram(model_bytes=5 * 1024 * 1024, margin_mb=512) == 0.0


def test_watchdog_kills_on_low_free_ram(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(cb, "free_ram_mb", lambda: 1_000)
    monkeypatch.setattr(cb, "physical_ram_mb", lambda: 32_000)
    monkeypatch.setattr(cb, "process_rss_mb", lambda pid: 6_000)
    monkeypatch.setattr(cb, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cb, "kill_process_tree", lambda pid: killed.append(pid))

    fired = threading.Event()

    def on_kill(reason: str) -> None:
        fired.set()

    cb.start_ram_watchdog(123, floor_mb=2500, poll_s=0.01, on_kill=on_kill)
    assert fired.wait(2.0)
    assert killed == [123]


def test_watchdog_kills_on_rss_cap(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(cb, "free_ram_mb", lambda: 20_000)
    monkeypatch.setattr(cb, "physical_ram_mb", lambda: 32_000)
    monkeypatch.setattr(cb, "process_rss_mb", lambda pid: 30_000)
    monkeypatch.setattr(cb, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cb, "kill_process_tree", lambda pid: killed.append(pid))

    fired = threading.Event()

    def on_kill(reason: str) -> None:
        fired.set()

    cb.start_ram_watchdog(7, reserve_mb=4096, poll_s=0.01, on_kill=on_kill)
    assert fired.wait(2.0)
    assert killed == [7]


def test_watchdog_healthy_does_not_kill(monkeypatch):
    killed: list[int] = []
    monkeypatch.setattr(cb, "free_ram_mb", lambda: 20_000)
    monkeypatch.setattr(cb, "physical_ram_mb", lambda: 32_000)
    monkeypatch.setattr(cb, "process_rss_mb", lambda pid: 6_000)
    monkeypatch.setattr(cb, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cb, "kill_process_tree", lambda pid: killed.append(pid))

    cb.start_ram_watchdog(7, floor_mb=2500, reserve_mb=4096, poll_s=0.01)
    # Let a few polls happen; nothing should be killed.
    import time

    time.sleep(0.15)
    assert killed == []


def test_watchdog_degraded_warns_once(monkeypatch):
    """Unavailable readers must never kill; warn once and keep polling."""
    killed: list[int] = []
    monkeypatch.setattr(cb, "free_ram_mb", lambda: None)
    monkeypatch.setattr(cb, "physical_ram_mb", lambda: None)
    monkeypatch.setattr(cb, "process_rss_mb", lambda pid: None)
    monkeypatch.setattr(cb, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cb, "kill_process_tree", lambda pid: killed.append(pid))

    cb.start_ram_watchdog(7, poll_s=0.01)
    import time

    time.sleep(0.15)
    assert killed == []
