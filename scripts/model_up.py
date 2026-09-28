from __future__ import annotations

import ast
import os
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

from autoresearch.core import fingerprint
from autoresearch.core.fingerprint import FingerprintError
from autoresearch.core.llama_runner import (
    IS_WINDOWS,
    assert_flags_supported,
    is_spec_enabled,
    load_mode_flag,
    resolve_llama_server,
    resolve_model_path,
)
from autoresearch.core.model_arch import gguf_has_mtp, resolve_n_cpu_moe
from autoresearch.core.single_load import SingleLoadError, enforce_single_load

ALIASES_DIR = REPO_ROOT / "models" / "aliases"
FINGERPRINTS_DIR = REPO_ROOT / "fingerprints"
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
    # Optional repo-relative (or absolute) llama.cpp tree with build-cuda/bin.
    # Needed for arch forks (e.g. llama.cpp-nanbeige42) that upstream cannot load.
    llama_cpp_root: str | None = None


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
        if key in {"alias", "model", "host", "description", "status", "llama_cpp_root"}:
            data[key] = value
        elif key == "port":
            data[key] = int(value)

    alias = str(data.get("alias") or path.parent.name)
    model = str(data.get("model") or "")
    if not model:
        raise ValueError(f"{path}: missing model")

    llama_root = data.get("llama_cpp_root")
    return AliasConfig(
        name=path.parent.name,
        model=model,
        port=int(data.get("port", DEFAULT_PORT)),
        host=str(data.get("host", DEFAULT_HOST)),
        alias=alias,
        flags=tuple(flags),
        path=path,
        llama_cpp_root=str(llama_root) if llama_root else None,
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


def _resolve_alias_server(cfg: AliasConfig) -> Path:
    if not cfg.llama_cpp_root:
        return resolve_llama_server()
    root = Path(cfg.llama_cpp_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    exe = "llama-server.exe" if IS_WINDOWS else "llama-server"
    for candidate in (
        root / "build-cuda" / "bin" / exe,
        root / "build-cuda" / "bin" / "Release" / exe,
        root / "build-cpu" / "bin" / exe,
        root / "build" / "bin" / exe,
    ):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"llama-server not found under llama_cpp_root={cfg.llama_cpp_root!r} "
        f"(looked in {root}/build-cuda|build-cpu|build/bin)"
    )


# ENGINE_DEFAULTS server/harness split lives in the bus owner
# (autoresearch/core/fingerprint.py); same engine = same server Pi sees.
_SERVER_ENGINE_KEYS = fingerprint.SERVER_ENGINE_KEYS
_HARNESS_ONLY_ENGINE_KEYS = fingerprint.HARNESS_ONLY_ENGINE_KEYS


def _probe_spec_type(server_binary: Path) -> str:
    """Auto-select mtp vs draft-mtp like the trial runner (MTP GGUF, no SPEC_TYPE)."""
    try:
        help_text = subprocess.check_output(
            [str(server_binary), "--help"], stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        help_text = "mtp"
    return "mtp" if "mtp" in help_text else "draft-mtp"


def fingerprint_flags(engine: dict, *, model_path: Path, server_binary: Path) -> list[str]:
    """Map a Fingerprint engine mapping to llama-server flags (issue #52).

    Same subset and order as the trial runner so Pi sees the engine the TPS
    climb wrote. Unknown keys raise (a typo'd hand-written file must never
    silently serve defaults); harness-only keys are ignored.
    """
    unknown = sorted(
        k
        for k in engine
        if k not in _SERVER_ENGINE_KEYS and k not in _HARNESS_ONLY_ENGINE_KEYS and k != "MODEL"
    )
    if unknown:
        raise FingerprintError(f"unknown engine keys (not ENGINE_DEFAULTS): {unknown}")

    kv = engine.get("KV_CACHE") or "q4_0"
    cache_k = engine.get("KV_CACHE_K") or kv
    cache_v = engine.get("KV_CACHE_V") or kv
    ngl = engine.get("N_GPU_LAYERS")
    cmd = [
        "--ctx-size",
        str(engine.get("CTX_SIZE") or 131072),
        "--batch-size",
        str(engine.get("BATCH_SIZE") or 512),
        "--ubatch-size",
        str(engine.get("UBATCH_SIZE") or 128),
        "--threads",
        str(engine.get("THREADS") or 8),
        "--parallel",
        str(engine.get("PARALLEL") or 1),
        "--n-gpu-layers",
        str(999 if ngl is None else ngl),
    ]
    numa = engine.get("NUMA")
    if numa is not None:
        cmd += ["--numa", str(numa)]
    cmd += [
        "--cache-type-k",
        str(cache_k),
        "--cache-type-v",
        str(cache_v),
        "--flash-attn",
        str(engine.get("FLASH_ATTN") or "on"),
    ]
    threads_batch = engine.get("THREADS_BATCH")
    if threads_batch is not None:
        cmd += ["--threads-batch", str(threads_batch)]
    cmd += load_mode_flag(bool(engine.get("NO_MMAP")), bool(engine.get("MLOCK")))
    if engine.get("JINJA"):
        cmd += ["--jinja"]
    reasoning_budget = engine.get("REASONING_BUDGET")
    if reasoning_budget is not None:
        cmd += ["--reasoning-budget", str(reasoning_budget)]
    reasoning_budget_message = engine.get("REASONING_BUDGET_MESSAGE")
    if reasoning_budget_message is not None:
        cmd += ["--reasoning-budget-message", str(reasoning_budget_message)]
    reasoning = engine.get("REASONING")
    if reasoning is not None:
        cmd += ["--reasoning", str(reasoning)]
    reasoning_effort = engine.get("REASONING_EFFORT")
    if reasoning_effort is not None:
        cmd += ["--reasoning-effort", str(reasoning_effort)]
    reasoning_preserve = engine.get("REASONING_PRESERVE")
    if reasoning_preserve is True:
        cmd += ["--reasoning-preserve"]
    elif reasoning_preserve is False:
        cmd += ["--no-reasoning-preserve"]
    if engine.get("CONT_BATCHING"):
        cmd += ["--cont-batching"]
    cache_reuse = engine.get("CACHE_REUSE")
    if cache_reuse is not None and int(cache_reuse) > 0:
        cmd += ["--cache-reuse", str(int(cache_reuse))]

    spec_type = engine.get("SPEC_TYPE")
    draft_n_max = int(engine.get("SPEC_DRAFT_N_MAX") or 0)
    if spec_type is None and gguf_has_mtp(model_path) and draft_n_max > 0:
        spec_type = _probe_spec_type(server_binary)
    if is_spec_enabled(spec_type, draft_n_max):
        cmd += [
            "--spec-type",
            str(spec_type),
        ]
        if draft_n_max > 0:
            cmd += [
                "--spec-draft-n-max",
                str(draft_n_max),
                "--spec-draft-type-k",
                str(cache_k),
                "--spec-draft-type-v",
                str(cache_v),
            ]
        draft = engine.get("SPEC_DRAFT_MODEL")
        if draft:
            draft_path = Path(str(draft))
            if not draft_path.is_absolute():
                draft_path = model_path.parent / draft_path
            if not draft_path.exists():
                raise FileNotFoundError(f"draft model not found: {draft_path}")
            cmd += ["--spec-draft-model", str(draft_path)]

    moe_profile = engine.get("MOE_CACHE_PROFILE")
    if moe_profile:
        profile = Path(str(moe_profile))
        if not profile.is_absolute():
            profile = REPO_ROOT / profile
        cmd += ["--moe-cache-profile", str(profile)]
        moe_slots = engine.get("MOE_CACHE_SLOTS")
        if moe_slots:
            cmd += ["--moe-cache-slots", str(int(moe_slots))]

    try:
        resolved_moe, _ = resolve_n_cpu_moe(model_path, engine.get("N_CPU_MOE"))
    except ValueError as exc:
        raise FingerprintError(str(exc)) from exc
    if resolved_moe is not None:
        cmd += ["--n-cpu-moe", str(resolved_moe)]
    return cmd


def _load_fingerprint(cfg: AliasConfig) -> dict:
    """Load the Fingerprint file for the alias GGUF; fail closed when missing."""
    basename = Path(cfg.model).name
    target = fingerprint.path_for(basename, FINGERPRINTS_DIR)
    if not target.exists():
        raise FileNotFoundError(
            f"Fingerprint not found: {target} for model {basename!r} "
            f"(alias {cfg.name!r}). Run a TPS climb or hand-write one "
            "(ADR 0014); model-up never falls back to alias flags."
        )
    return fingerprint.load(target)


def build_command(cfg: AliasConfig) -> tuple[list[str], Path]:
    """Build the llama-server command: identity/bind from alias, engine from Fingerprint."""
    binary = _resolve_alias_server(cfg)
    model_path = _resolve_model_path(cfg.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    data = _load_fingerprint(cfg)

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
    cmd += fingerprint_flags(data["engine"], model_path=model_path, server_binary=binary)
    assert_flags_supported(cmd, binary)
    return cmd, model_path


def _probe_host(host: str) -> str:
    """0.0.0.0/:: are valid bind targets but not connect targets on Windows."""
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _is_healthy(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{_probe_host(host)}:{port}/health", timeout=2
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((_probe_host(host), port), timeout=1):
            return True
    except OSError:
        return False


def _pid_exists(pid: int) -> bool:
    if IS_WINDOWS:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_pid(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    os.kill(pid, 15)


def _server_kwargs() -> dict[str, object]:
    if IS_WINDOWS:
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        }
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
            return RunningState(
                pid=int(parts[0]), name="", alias="", port=DEFAULT_PORT, host=DEFAULT_HOST
            )
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


def cmd_start(alias_name: str | None, allow_multi: bool = False) -> int:
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
    except FingerprintError as exc:
        print(f"invalid Fingerprint: {exc}")
        return 1
    if cfg.flags:
        print(
            f"NOTE: ignoring {len(cfg.flags)} stale alias flags; "
            "the Fingerprint file is the engine source of truth."
        )

    if _is_healthy(cfg.host, cfg.port):
        print(f"Already up: {cfg.name} at http://{cfg.host}:{cfg.port}/v1")
        return 0
    if _is_listening(cfg.host, cfg.port):
        print(f"Port {cfg.port} is already in use.")
        return 1

    # Single-load gate (#41): refuse a second full server while one is live.
    # --allow-multi True bypasses; otherwise fall through to the env flag
    # (None lets resolve_allow_multi read AUTORESEARCH_ALLOW_MULTI_SERVERS).
    try:
        enforce_single_load(allow_multi=(allow_multi or None))
    except SingleLoadError as exc:
        print(f"Refusing: {exc}")
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GGML_CUDA_NO_PINNED"] = "1"
    with open(LOGFILE, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            cwd=str(REPO_ROOT),
            env=env,
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
    allow_multi = "--allow-multi" in argv
    argv = [arg for arg in argv if arg != "--allow-multi"]
    if not argv:
        return cmd_start(None, allow_multi=allow_multi)
    if argv[0] == "list":
        return cmd_list()
    if argv[0] == "status":
        return cmd_status()
    if argv[0] in {"stop", "down"}:
        return cmd_stop()
    return cmd_start(argv[0], allow_multi=allow_multi)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
