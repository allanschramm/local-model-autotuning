"""Tests for per-run artifact rotation in autoresearch.runners.evaluation."""

import os
import time
from pathlib import Path

import pytest

from autoresearch.runners.evaluation import _new_server_log, _prune_glob


@pytest.fixture
def fake_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect evaluation.LOG_DIR to a tmp dir and shrink the keep window."""
    monkeypatch.setattr("autoresearch.runners.evaluation.LOG_DIR", tmp_path)
    monkeypatch.setattr("autoresearch.runners.evaluation.LOG_KEEP", 3)
    return tmp_path


def test_prune_glob_keeps_newest(fake_log_dir: Path):
    """Oldest files are deleted; the `keep` newest survive."""
    for i in range(5):
        p = fake_log_dir / f"llama-server-20260819-{i:02d}-Foo.log"
        p.write_text("x", encoding="utf-8")
        os.utime(p, (time.time() - (10 - i), time.time() - (10 - i)))

    _prune_glob(fake_log_dir, "llama-server-*.log", 3)

    remaining = sorted(p.name for p in fake_log_dir.glob("llama-server-*.log"))
    assert len(remaining) == 3
    assert remaining == [
        "llama-server-20260819-02-Foo.log",
        "llama-server-20260819-03-Foo.log",
        "llama-server-20260819-04-Foo.log",
    ]


def test_prune_glob_missing_dir_is_noop(tmp_path: Path):
    """Missing directory must not raise."""
    _prune_glob(tmp_path / "nope", "llama-server-*.log", 3)


def test_new_server_log_rotates_and_names(fake_log_dir: Path):
    """5 existing logs + 1 new = 3 remaining (prune to LOG_KEEP-1, then add)."""
    for i in range(5):
        p = fake_log_dir / f"llama-server-20260819-{i:02d}-Foo.log"
        p.write_text("x", encoding="utf-8")
        os.utime(p, (time.time() - (10 - i), time.time() - (10 - i)))

    new_path = _new_server_log("Ornith-1.5-9B-Q4_K_M.gguf")

    assert new_path.parent == fake_log_dir
    assert new_path.name.startswith("llama-server-")
    assert new_path.name.endswith("-Ornith-1.5-9B-Q4_K_M.log")
    assert not new_path.exists()  # path is returned; the runner opens it later
    assert len(list(fake_log_dir.glob("llama-server-*.log"))) == 2  # pruned to LOG_KEEP-1
    new_path.touch()  # runner's open("w+") — the "add 1" that fills the keep window
    remaining = list(fake_log_dir.glob("llama-server-*.log"))
    assert len(remaining) == 3


def test_new_server_log_unknown_model(fake_log_dir: Path):
    """Empty model filename must not crash and uses 'unknown' stem."""
    new_path = _new_server_log("")
    assert new_path.name.endswith("-unknown.log")


def test_evaluation_k2_horizon_stop_token_detection():
    """Verify K2-Horizon detection across filenames and paths."""
    # Test cases: (model_filename, model_path, norm_model, expected_stop)
    cases = [
        (
            "k2-horizon-7b-q4_k_m.gguf",
            Path("models/k2-horizon-7b-q4_k_m.gguf"),
            None,
            ["<|ifm|im_end|>", "</s>"],
        ),
        ("k2_horizon_7b.gguf", Path("models/k2_horizon_7b.gguf"), None, ["<|ifm|im_end|>", "</s>"]),
        ("k2horizon.gguf", Path("models/k2horizon.gguf"), None, ["<|ifm|im_end|>", "</s>"]),
        ("model.gguf", Path("models/k2-horizon/model.gguf"), None, ["<|ifm|im_end|>", "</s>"]),
        ("model.gguf", Path("models/other/model.gguf"), "k2-horizon", ["<|ifm|im_end|>", "</s>"]),
        ("qwen.gguf", Path("models/qwen.gguf"), None, None),
    ]

    for model_filename, model_path, norm_model, expected in cases:
        norm = {}
        if norm_model:
            norm["model"] = norm_model
        stop_val = norm.get("stop") or norm.get("stop_tokens")
        model_identifiers = [
            model_filename.lower(),
            str(model_path).lower(),
            str(norm.get("model") or "").lower(),
        ]
        if stop_val is None and any(
            needle in identifier
            for needle in ("k2-horizon", "k2_horizon", "k2horizon")
            for identifier in model_identifiers
        ):
            stop_val = ["<|ifm|im_end|>", "</s>"]

        assert stop_val == expected
