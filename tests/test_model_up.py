from __future__ import annotations

from pathlib import Path

from scripts import model_up


def test_model_up_parses_flags_and_builds_command(tmp_path, monkeypatch):
    alias_dir = tmp_path / "models" / "aliases" / "demo"
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "demo.gguf"
    model_file.write_text("x", encoding="utf-8")
    alias_file = alias_dir / "config.yaml"
    alias_file.write_text(
        "\n".join(
            [
                'alias: demo-model',
                'model: models/demo.gguf',
                "port: 19090",
                "host: 127.0.0.1",
                "flags:",
                "  - --jinja",
                "  - --ctx-size 131072  # comment",
                "  - --flash-attn on",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")
    monkeypatch.setattr(model_up, "resolve_llama_server", lambda: Path("llama-server.exe"))

    cfg = model_up.load_alias_config(alias_file)
    cmd, model_path = model_up.build_command(cfg)

    assert cfg.name == "demo"
    assert cfg.alias == "demo-model"
    assert model_path == model_file
    assert cmd[:8] == [
        "llama-server.exe",
        "--model",
        str(model_file),
        "--alias",
        "demo-model",
        "--host",
        "127.0.0.1",
        "--port",
        "19090",
    ]
    assert cmd[8:] == ["--jinja", "--ctx-size", "131072", "--flash-attn", "on"]
