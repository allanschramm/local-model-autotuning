from __future__ import annotations

from pathlib import Path

from scripts import model_up


def _seed_fingerprint(monkeypatch, tmp_path, basename="demo.gguf", engine=None):
    """Seed fingerprints/<stem>.json + hermetic GGUF readers for fake model files."""
    from autoresearch.core import fingerprint as fp_mod

    fp_dir = tmp_path / "fingerprints"
    target = fp_mod.path_for(basename, fp_dir)
    fp_mod.dump(target, model=basename, engine=dict(engine or {"CTX_SIZE": 32768}))
    monkeypatch.setattr(model_up, "FINGERPRINTS_DIR", fp_dir)
    # Fake test models are text files: MoE auto-resolution would raise.
    monkeypatch.setattr(model_up, "resolve_n_cpu_moe", lambda path, raw: (raw, False))
    monkeypatch.setattr(model_up, "gguf_has_mtp", lambda path: False)
    return target


def _alias_env(tmp_path, monkeypatch, name="demo"):
    alias_dir = tmp_path / "models" / "aliases" / name
    alias_dir.mkdir(parents=True)
    model_file = tmp_path / "models" / "demo.gguf"
    model_file.write_text("x", encoding="utf-8")
    (alias_dir / "config.yaml").write_text(
        "\n".join(
            [
                "alias: demo-model",
                "model: models/demo.gguf",
                "port: 18080",
                "host: 127.0.0.1",
                "status: ready",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_up, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(model_up, "ALIASES_DIR", tmp_path / "models" / "aliases")
    monkeypatch.setattr(model_up, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(model_up, "STATE_FILE", tmp_path / "state" / "model-up.state")
    monkeypatch.setattr(model_up, "LOGFILE", tmp_path / "state" / "model-up.log")
    monkeypatch.setattr(model_up, "resolve_llama_server", lambda: Path("llama-server.exe"))
    _seed_fingerprint(monkeypatch, tmp_path)
    monkeypatch.setattr(model_up, "_is_healthy", lambda host, port: False)
    monkeypatch.setattr(model_up, "_is_listening", lambda host, port: False)
    monkeypatch.setattr(model_up.time, "sleep", lambda _: None)
    return model_file


def _fake_popen_capture(monkeypatch, captured):
    class FakeProc:
        pid = 4242
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        model_up.subprocess,
        "Popen",
        lambda cmd, **kwargs: captured.append(cmd) or FakeProc(),
    )


def test_model_up_start_refuses_second_full_server(tmp_path, monkeypatch):
    _alias_env(tmp_path, monkeypatch)
    captured: list = []
    _fake_popen_capture(monkeypatch, captured)
    monkeypatch.setattr(
        "autoresearch.core.single_load.live_full_server_pids",
        lambda ports, process_names: [1234],
    )

    assert model_up.cmd_start("demo") == 1
    assert captured == []
