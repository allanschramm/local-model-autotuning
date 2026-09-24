"""Unit tests for the single-load gate (issue #41).

Refuse a second full server while one is live (default), let a speculative
draft ride on the same server without counting, and bypass via allow-multi
(--allow-multi / AUTORESEARCH_ALLOW_MULTI_SERVERS). No GPU, no real processes:
the #37 detection surface (listeners_on_ports / processes_by_name) is mocked.
"""

from pathlib import Path

import pytest

import autoresearch.core.single_load as single_load
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
from autoresearch.core.sglang_runner import SGLangServerRunner


def _intent(**overrides) -> ServerIntent:
    kwargs = dict(
        model_path=Path("models/test-model.gguf"),
        ctx_size=2048,
        kv_cache="q4_0",
        flash_attn="on",
        port=18080,
    )
    kwargs.update(overrides)
    return ServerIntent(**kwargs)


# ---------------------------------------------------------------------------
# allow-multi resolution
# ---------------------------------------------------------------------------


def test_allow_multi_env_absent_or_falsy(monkeypatch):
    monkeypatch.delenv(single_load.ALLOW_MULTI_ENV, raising=False)
    assert single_load.resolve_allow_multi() is False
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "0")
    assert single_load.resolve_allow_multi() is False
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "false")
    assert single_load.resolve_allow_multi() is False


# ---------------------------------------------------------------------------
# live-full-server detection
# ---------------------------------------------------------------------------


def test_live_full_server_is_name_plus_port_intersection(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {101, 202, 303})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {202, 404})
    assert single_load.live_full_server_pids() == [202]


# ---------------------------------------------------------------------------
# gate: refuse / allow
# ---------------------------------------------------------------------------


def test_gate_passes_without_live_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: set())
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: set())
    assert single_load.assert_single_load() == []


def test_gate_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    with pytest.raises(single_load.SingleLoadError) as ctx:
        single_load.assert_single_load()
    assert "4242" in str(ctx.value)
    assert single_load.ALLOW_MULTI_ENV in str(ctx.value)
    assert "--allow-multi" in str(ctx.value)


# ---------------------------------------------------------------------------
# Trial runner wiring (refuse + bypass, no GPU)
# ---------------------------------------------------------------------------


def test_llama_runner_enter_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.resolve_llama_server", lambda: Path("llama-server")
    )
    monkeypatch.setattr(LlamaServerRunner, "_start_vram_sampler", lambda self: None)
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    runner = LlamaServerRunner(_intent())
    with pytest.raises(single_load.SingleLoadError):
        runner.__enter__()
    assert sweep_calls == []
    assert runner._guard is None


def test_sglang_runner_start_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.sglang_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    runner = SGLangServerRunner(_intent())
    with pytest.raises(single_load.SingleLoadError):
        runner.start()
    assert sweep_calls == []
    assert runner._guard is None
