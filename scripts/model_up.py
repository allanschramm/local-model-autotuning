from __future__ import annotations

import ast
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_root_on_sys_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_sys_path()

from autoresearch.core.llama_runner import IS_WINDOWS, resolve_llama_server, resolve_model_path
ALIASES_DIR = REPO_ROOT / "models" / "aliases"
STATE_DIR = (
    Path(os.environ["LOCALAPPDATA"]) / "local-model-autoresearch"
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA")
    else Path.home() / ".local" / "share"
)
STATE_FILE = STATE_DIR / "model-up.state"
LOGFILE = STATE_DIR / "model-up.log"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080


@dataclass(frozen=True)
class AliasConfig:
    name: str
    model: str
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST
    alias: str | None = None
    flags: tuple[str, ...] = ()
    path: Path | None = None


@dataclass(frozen=True)
class RunningState:
    pid: int
    name: str
    alias: str
    port: int
    host: str


def _strip_inline_comment(text: str) -> str:
    out: list[str] = []
    in_single = False
    in_double = False
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(raw: str):
    text = _strip_inline_comment(raw).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if text[:1] in {'"', "'"} and text[-1:] == text[:1]:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text[1:-1]
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _discover_alias_files() -> list[Path]:
    if not ALIASES_DIR.exists():
        return []
    return sorted(
        alias_dir / "config.yaml"
        for alias_dir in ALIASES_DIR.iterdir()
        if alias_dir.is_dir() and (alias_dir / "config.yaml").exists()
    )


def discover_aliases() -> list[AliasConfig]:
    return [load_alias_config(path) for path in _discover_alias_files()]


def load_alias_config(path: Path) -> AliasConfig:
    data: dict[str, object] = {}
    flags: list[str] = []
    in_flags = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if in_flags:
            if line.lstrip().startswith("- "):
                flag = _strip_inline_comment(line.lstrip()[2:]).strip()
                if flag:
                    flags.append(flag)
                continue
            in_flags = False

        if line[:1].isspace():
            continue
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = _parse_scalar(raw_value)

        if key == "flags":
            in_flags = True
            continue
        if key in {"alias", "model", "host", "description", "status"}:
            data[key] = value
        elif key == "port":
            data[key] = int(value)

    alias = str(data.get("alias") or path.parent.name)
    model = str(data.get("model") or "")
    if not model:
        raise ValueError(f"{path}: missing model")

    return AliasConfig(
        name=path.parent.name,
        model=model,
        port=int(data.get("port", DEFAULT_PORT)),
        host=str(data.get("host", DEFAULT_HOST)),
        alias=alias,
        flags=tuple(flags),
        path=path,
    )


def _resolve_model_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate

    models_dir = REPO_ROOT / "models"
    at_repo = REPO_ROOT / candidate
    if at_repo.exists():
        return at_repo

    ref = Path(*candidate.parts[1:]) if candidate.parts[:1] == ("models",) else candidate
    return resolve_model_path(models_dir, ref)


def build_command(cfg: AliasConfig) -> tuple[list[str], Path]:
    binary = resolve_llama_server()
    model_path = _resolve_model_path(cfg.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    cmd = [
        str(binary),
        "--model",
        str(model_path),
        "--alias",
        cfg.alias or cfg.name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
    ]
    for flag in cfg.flags:
        cmd.extend(shlex.split(flag))
    return cmd, model_path


def _is_healthy(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _pid_exists(pid: int) -> bool:
    if IS_WINDOWS:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_pid(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        return
    os.kill(pid, 15)


def _server_kwargs() -> dict[str, object]:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}


def _state_path() -> Path:
    return STATE_FILE


def _write_state(state: RunningState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(
        f"{state.pid}\t{state.name}\t{state.alias}\t{state.port}\t{state.host}\n",
        encoding="utf-8",
    )


def _read_state() -> RunningState | None:
    path = _state_path()
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        path.unlink(missing_ok=True)
        return None

    parts = raw.split("\t")
    try:
        if len(parts) == 1:
            return RunningState(pid=int(parts[0]), name="", alias="", port=DEFAULT_PORT, host=DEFAULT_HOST)
        return RunningState(
            pid=int(parts[0]),
            name=parts[1] if len(parts) > 1 else "",
            alias=parts[2] if len(parts) > 2 else "",
            port=int(parts[3]) if len(parts) > 3 else DEFAULT_PORT,
            host=parts[4] if len(parts) > 4 else DEFAULT_HOST,
        )
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def cmd_list() -> int:
    aliases = discover_aliases()
    if not aliases:
        print(f"No aliases found under {ALIASES_DIR}")
        return 1
    for cfg in aliases:
        print(f"{cfg.name}\t{cfg.alias}\t{cfg.model}")
    return 0


def cmd_status() -> int:
    state = _read_state()
    if state is None:
        print("Not running.")
        return 1
    if not _pid_exists(state.pid):
        _state_path().unlink(missing_ok=True)
        print("Not running.")
        return 1

    print(f"PID={state.pid}")
    if state.name:
        print(f"alias={state.name}")
    if state.alias:
        print(f"model={state.alias}")
    print(f"port={state.port}")
    print(f"host={state.host}")
    if _is_listening(state.host, state.port):
        print(f"health={'OK' if _is_healthy(state.host, state.port) else 'NOT READY'}")
        print(f"base_url=http://{state.host}:{state.port}/v1")
    return 0


def cmd_stop() -> int:
    state = _read_state()
    if state is None:
        print("Not running.")
        return 1
    if _pid_exists(state.pid):
        _kill_pid(state.pid)
        print(f"Killed PID {state.pid}")
    _state_path().unlink(missing_ok=True)
    return 0


def _pick_default_alias(aliases: list[AliasConfig]) -> AliasConfig | None:
    if not aliases:
        return None
    return aliases[0]


def cmd_start(alias_name: str | None) -> int:
    aliases = discover_aliases()
    if not aliases:
        print(f"No aliases found under {ALIASES_DIR}")
        return 1

    cfg = None
    if alias_name:
        for item in aliases:
            if item.name == alias_name or item.alias == alias_name:
                cfg = item
                break
        if cfg is None:
            print(f"Unknown alias: {alias_name}")
            print("Available aliases:")
            for item in aliases:
                print(f"  {item.name}")
            return 1
    else:
        cfg = _pick_default_alias(aliases)
        if cfg is None:
            print("No default alias found.")
            return 1

    try:
        cmd, _ = build_command(cfg)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if _is_healthy(cfg.host, cfg.port):
        print(f"Already up: {cfg.name} at http://{cfg.host}:{cfg.port}/v1")
        return 0
    if _is_listening(cfg.host, cfg.port):
        print(f"Port {cfg.port} is already in use.")
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **_server_kwargs(),
        )
        _write_state(RunningState(proc.pid, cfg.name, cfg.alias or cfg.name, cfg.port, cfg.host))

        for _ in range(180):
            if _is_healthy(cfg.host, cfg.port):
                print(f"OK {cfg.name} ready at http://{cfg.host}:{cfg.port}/v1")
                print(f"model={cfg.alias or cfg.name}")
                print(f"base_url=http://{cfg.host}:{cfg.port}/v1")
                return 0
            if proc.poll() is not None:
                print(f"FAIL: process exited with code {proc.returncode}. Log: {LOGFILE}")
                return 1
            time.sleep(1)

    print(f"WARN: server did not become healthy in 180s. Log: {LOGFILE}")
    return 1


def main(argv: list[str]) -> int:
    if not argv:
        return cmd_start(None)
    if argv[0] == "list":
        return cmd_list()
    if argv[0] == "status":
        return cmd_status()
    if argv[0] in {"stop", "down"}:
        return cmd_stop()
    return cmd_start(argv[0])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

