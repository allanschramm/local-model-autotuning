"""resolve_model_path: flat + nested (LM Studio) layout under models/."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.core.llama_runner import resolve_model_path, ServerIntent


def test_resolve_prefers_direct_relative_path(tmp_path: Path):
    models = tmp_path / "models"
    target = models / "draft" / "mtp.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    assert resolve_model_path(models, "draft/mtp.gguf") == target


def test_resolve_basename_under_publisher_model(tmp_path: Path):
    models = tmp_path / "models"
    target = models / "lmstudio-community" / "gemma-4-e4b-it-gguf" / "gemma.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    assert resolve_model_path(models, "gemma.gguf") == target


def test_resolve_skips_aliases_and_cache(tmp_path: Path):
    models = tmp_path / "models"
    decoy = models / "aliases" / "x" / "gemma.gguf"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"d")
    cache = models / ".cache" / "gemma.gguf"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"c")
    real = models / "local" / "gemma" / "gemma.gguf"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"r")

    assert resolve_model_path(models, "gemma.gguf") == real


def test_resolve_missing_returns_direct_path(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    got = resolve_model_path(models, "missing.gguf")
    assert got == models / "missing.gguf"
    assert not got.exists()


def test_available_gguf_names_skips_draft_vision(tmp_path: Path):
    from autoloop import _available_gguf_names

    models = tmp_path / "models"
    main = models / "local" / "m" / "main.gguf"
    draft = models / "draft" / "d.gguf"
    vision = models / "vision" / "v.gguf"
    for p in (main, draft, vision):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    assert _available_gguf_names(models) == ["main.gguf"]


def test_from_config_resolves_nested_model_and_draft(tmp_path: Path):
    models = tmp_path / "models"
    main = models / "lmstudio-community" / "gemma-4-e4b-it-gguf" / "main.gguf"
    draft = models / "draft" / "mtp.gguf"
    main.parent.mkdir(parents=True)
    draft.parent.mkdir(parents=True)
    main.write_bytes(b"m")
    draft.write_bytes(b"d")

    intent, _ = ServerIntent.from_config(
        {
            "MODEL": "main.gguf",
            "SPEC_DRAFT_MODEL": "draft/mtp.gguf",
            "SPEC_DRAFT_N_MAX": 4,
            "CTX_SIZE": 2048,
        },
        models,
    )
    assert intent.model_path == main
    assert Path(intent.spec_draft_model) == draft
