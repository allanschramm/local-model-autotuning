"""Fingerprint file IO tests (issue #49, ADR 0014 bus).

The file is the TPS-climb -> launcher bus: GGUF basename + ENGINE_DEFAULTS,
optional SAMPLER_DEFAULTS. Machine-local, gitignored, no GPU needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.core.fingerprint import dump, load

_ENGINE = {
    "MODEL": "model.gguf",
    "CTX_SIZE": 65536,
    "N_GPU_LAYERS": -1,
    "KV_CACHE": "q4_0",
    "BATCH_SIZE": 512,
    "UBATCH_SIZE": 128,
    "THREADS": 8,
    "FLASH_ATTN": "on",
    "N_CPU_MOE": None,
    "VRAM_LIMIT_MB": 7900,
    "TPS_FLOOR": 20.0,
}
_SAMPLER = {"TEMP": 0.8, "TOP_P": 0.95, "TOP_K": 40, "MIN_P": 0.05}


def test_roundtrip_engine_only_returns_same_basename_and_engine(tmp_path: Path) -> None:
    path = dump(tmp_path / "model.json", model="model.gguf", engine=dict(_ENGINE))
    loaded = load(path)
    assert loaded["model"] == "model.gguf"
    assert loaded["engine"] == _ENGINE
    assert loaded["sampler"] is None


def test_roundtrip_with_optional_sampler(tmp_path: Path) -> None:
    path = dump(
        tmp_path / "model.json",
        model="model.gguf",
        engine=dict(_ENGINE),
        sampler=dict(_SAMPLER),
    )
    loaded = load(path)
    assert loaded["model"] == "model.gguf"
    assert loaded["engine"] == _ENGINE
    assert loaded["sampler"] == _SAMPLER


def test_load_requires_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "nover.json"
    path.write_text(json.dumps({"model": "model.gguf", "engine": dict(_ENGINE)}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load(path)


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = dump(tmp_path / "model.json", model="model.gguf", engine=dict(_ENGINE))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load(path)


def test_dump_strips_directories_to_basename(tmp_path: Path) -> None:
    loaded = load(dump(tmp_path / "m.json", model="some/dir/model.gguf", engine=dict(_ENGINE)))
    assert loaded["model"] == "model.gguf"
    assert loaded["engine"]["MODEL"] == "model.gguf"


def test_load_rejects_model_with_path_separators(tmp_path: Path) -> None:
    path = dump(tmp_path / "model.json", model="model.gguf", engine=dict(_ENGINE))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "some/dir/model.gguf"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="basename"):
        load(path)


@pytest.mark.parametrize(
    "engine",
    [
        {"MODEL": "model.gguf", "CTX_SIZE": 65536, "WEIGHTS": "D:/models/model.gguf"},
        {"MODEL": "model.gguf", "CTX_SIZE": 65536, "WEIGHTS": "/home/user/model.gguf"},
        {"MODEL": "model.gguf", "CTX_SIZE": 65536, "WEIGHTS": "\\\\server\\share\\m.gguf"},
        {"MODEL": "model.gguf", "CTX_SIZE": 65536, "WEIGHTS": "C:\\Users\\op\\model.gguf"},
    ],
)
def test_dump_rejects_absolute_user_paths(tmp_path: Path, engine: dict) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        dump(tmp_path / "m.json", model="model.gguf", engine=engine)


@pytest.mark.parametrize(
    "engine",
    [
        {"MODEL": "model.gguf", "hostname": "my-machine"},
        {"MODEL": "model.gguf", "alias_name": "daily-driver"},
        {"MODEL": "model.gguf", "gpu_sku": "RTX 9999"},
        {"MODEL": "model.gguf", "contact": "op@example.com"},
        {"MODEL": "model.gguf", "RELAY": "server.local"},
        {"MODEL": "model.gguf", "RELAY": "my-machine.example.com"},
        {"MODEL": "model.gguf", "RELAY": "localhost"},
    ],
)
def test_dump_rejects_private_keys_and_values(tmp_path: Path, engine: dict) -> None:
    with pytest.raises(ValueError, match="private"):
        dump(tmp_path / "m.json", model="model.gguf", engine=engine)
