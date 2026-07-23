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
    assert cmd[:9] == [
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
    assert cmd[9:] == ["--jinja", "--ctx-size", "131072", "--flash-attn", "on"]


def test_model_up_resolves_nested_lmstudio_layout(tmp_path, monkeypatch):
    alias_dir = tmp_path / "models" / "aliases" / "nested"
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "lmstudio-community" / "demo-gguf" / "demo.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("x", encoding="utf-8")
    alias_file = alias_dir / "config.yaml"
    alias_file.write_text(
        "\n".join(
            [
                "alias: nested-demo",
                "model: models/demo.gguf",
                "port: 18080",
                "host: 127.0.0.1",
                "flags:",
                "  - --jinja",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")
    monkeypatch.setattr(model_up, "resolve_llama_server", lambda: Path("llama-server.exe"))

    cfg = model_up.load_alias_config(alias_file)
    _, model_path = model_up.build_command(cfg)
    assert model_path == model_file


def test_model_up_uses_llama_cpp_root_for_fork_binary(tmp_path, monkeypatch):
    alias_dir = tmp_path / "models" / "aliases" / "forked"
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "demo.gguf"
    model_file.write_text("x", encoding="utf-8")
    fork_bin = tmp_path / "llama.cpp-fork" / "build-cuda" / "bin"
    fork_bin.mkdir(parents=True)
    server = fork_bin / ("llama-server.exe" if model_up.IS_WINDOWS else "llama-server")
    server.write_text("x", encoding="utf-8")
    alias_file = alias_dir / "config.yaml"
    alias_file.write_text(
        "\n".join(
            [
                "alias: forked-demo",
                "model: models/demo.gguf",
                "llama_cpp_root: llama.cpp-fork",
                "port: 18080",
                "host: 127.0.0.1",
                "flags:",
                "  - --jinja",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")

    cfg = model_up.load_alias_config(alias_file)
    cmd, model_path = model_up.build_command(cfg)

    assert cfg.llama_cpp_root == "llama.cpp-fork"
    assert model_path == model_file
    assert Path(cmd[0]) == server.resolve()


def test_model_up_adds_repo_root_to_sys_path(monkeypatch):
    repo_root = str(model_up.REPO_ROOT)
    monkeypatch.setattr(model_up.sys, "path", [p for p in model_up.sys.path if p != repo_root])

    model_up._ensure_repo_root_on_sys_path()

    assert model_up.sys.path[0] == repo_root


def test_model_up_resolves_spec_draft_model_path(tmp_path, monkeypatch):
    alias_dir = tmp_path / "models" / "aliases" / "gemma-draft"
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "main.gguf"
    model_file.write_text("x", encoding="utf-8")
    draft_file = tmp_path / "models" / "draft" / "mtp-demo.gguf"
    draft_file.parent.mkdir(parents=True)
    draft_file.write_text("x", encoding="utf-8")
    alias_file = alias_dir / "config.yaml"
    alias_file.write_text(
        "\n".join(
            [
                "alias: gemma-draft",
                "model: models/main.gguf",
                "port: 18080",
                "host: 127.0.0.1",
                "flags:",
                "  - --spec-type draft-mtp",
                "  - --spec-draft-model models/draft/mtp-demo.gguf",
                "  - --spec-draft-n-max 2",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")
    monkeypatch.setattr(model_up, "resolve_llama_server", lambda: Path("llama-server.exe"))

    cfg = model_up.load_alias_config(alias_file)
    cmd, _ = model_up.build_command(cfg)

    assert cmd[cmd.index("--spec-draft-model") + 1] == str(draft_file)


def test_model_up_start_sets_cwd_to_repo_root(tmp_path, monkeypatch):
    alias_dir = tmp_path / "models" / "aliases" / "cwd-demo"
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "demo.gguf"
    model_file.write_text("x", encoding="utf-8")
    alias_file = alias_dir / "config.yaml"
    alias_file.write_text(
        "\n".join(
            [
                "alias: cwd-demo",
                "model: models/demo.gguf",
                "port: 18080",
                "host: 127.0.0.1",
                "flags:",
                "  - --jinja",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4242
        returncode = 1

        def poll(self):
            return 1

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")
    monkeypatch.setattr(model_up, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(model_up, "STATE_FILE", tmp_path / "state" / "model-up.state")
    monkeypatch.setattr(model_up, "LOGFILE", tmp_path / "state" / "model-up.log")
    monkeypatch.setattr(model_up, "resolve_llama_server", lambda: Path("llama-server.exe"))
    monkeypatch.setattr(model_up, "_is_healthy", lambda host, port: False)
    monkeypatch.setattr(model_up, "_is_listening", lambda host, port: False)
    monkeypatch.setattr(model_up.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(model_up.time, "sleep", lambda _: None)

    assert model_up.cmd_start("cwd-demo") == 1
    assert captured["kwargs"]["cwd"] == str(tmp_path)
