#!/usr/bin/env python3
"""serve-config.py — Start llama-server using autoresearch/core/config.py.

Reads MODEL + key llama-server flags from the mutable Baseline in config.py
and starts the server detached (survives terminal close). After autoloop
finishes, this is how a user (or an agent) gets the tuned model back up
for live use.

Usage:
    python3 scripts/serve-config.py                # start with current config.py
    python3 scripts/serve-config.py stop           # kill running server
    python3 scripts/serve-config.py status         # show running state
    python3 scripts/serve-config.py print-cmd     # show the resolved command (don't start)
    python3 scripts/serve-config.py print-config  # show parsed config

Exit codes:
    0 — server up (or cmd completed successfully)
    1 — server not running / not found
    2 — invalid config or missing dependency
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Repo-rooted paths (relative — never absolute).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.core.llama_runner import IS_WINDOWS, _binary_candidates
CONFIG = REPO_ROOT / "autoresearch" / "core" / "config.py"
STATE_DIR = (
    Path(os.environ["LOCALAPPDATA"]) / "local-model-autoresearch"
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA")
    else Path.home() / ".local" / "share"
)
LOG = STATE_DIR / "serve-config.log"
PIDFILE = STATE_DIR / "serve-config.pid"

# Defaults for llama-server-only fields not in config.py.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_ALIAS = "local-model-autoresearch"


# ────────────────────────────────────────────────────────────────────
# Config parsing
# ────────────────────────────────────────────────────────────────────


def parse_config(path: Path) -> dict:
    """Load Baseline from config.py (ENGINE_DEFAULTS + SAMPLER_DEFAULTS)."""
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(2)

    from autoresearch.core.config import load_config
    return load_config()


def derive_alias(cfg: dict) -> str:
    """Derive the OpenAI-compatible model name from the GGUF filename.

    Convention: strip `.gguf` and replace path separators. So
    'gemma-4-26B-A4B-it-UD-Q4_K_M.gguf' → 'gemma-4-26B-A4B-it-UD-Q4_K_M'.
    """
    model = cfg.get("MODEL", "")
    name = Path(str(model)).stem  # strip directory + .gguf
    return name or DEFAULT_ALIAS


def build_args(cfg: dict) -> tuple[list[str], str, int, str]:
    """Build llama-server CLI args from config constants.

    Returns: (args, host, port, alias)
    """
    args: list[str] = []

    # --- required ---
    model = cfg.get("MODEL")
    if not model:
        print(f"ERROR: MODEL is not set in {CONFIG}", file=sys.stderr)
        sys.exit(2)
    model_path = Path(str(model))
    if not model_path.is_absolute():
        models_candidate = REPO_ROOT / "models" / model_path
        repo_candidate = REPO_ROOT / model_path
        if models_candidate.exists():
            model_path = models_candidate
        elif repo_candidate.exists():
            model_path = repo_candidate
        else:
            model_path = models_candidate
    args += ["--model", str(model_path)]

    alias = derive_alias(cfg)
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    args += ["--alias", alias, "--host", host, "--port", str(port)]

    # --- speculative / MTP ---
    spec_type = cfg.get("SPEC_TYPE")
    if spec_type and str(spec_type).lower() != "none":
        args += ["--spec-type", str(spec_type)]
        draft_n = cfg.get("SPEC_DRAFT_N_MAX", 0)
        if draft_n and int(draft_n) > 0:
            args += ["--spec-draft-n-max", str(int(draft_n))]

    # --- KV cache ---
    # KV_CACHE_K / KV_CACHE_V override KV_CACHE if both set.
    kv_k = cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE")
    kv_v = cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE")
    if kv_k:
        args += ["--cache-type-k", str(kv_k)]
    if kv_v:
        args += ["--cache-type-v", str(kv_v)]

    # --- context ---
    if cfg.get("CTX_SIZE"):
        args += ["--ctx-size", str(int(cfg["CTX_SIZE"]))]

    # --- GPU/CPU split ---
    if cfg.get("N_GPU_LAYERS") is not None:
        args += ["--n-gpu-layers", str(int(cfg["N_GPU_LAYERS"]))]
    if cfg.get("N_CPU_MOE") is not None:
        args += ["--n-cpu-moe", str(int(cfg["N_CPU_MOE"]))]

    # --- threads + batching ---
    if cfg.get("THREADS"):
        args += ["--threads", str(int(cfg["THREADS"]))]
    if cfg.get("THREADS_BATCH"):
        args += ["--threads-batch", str(int(cfg["THREADS_BATCH"]))]
    if cfg.get("BATCH_SIZE"):
        args += ["--batch-size", str(int(cfg["BATCH_SIZE"]))]
    if cfg.get("UBATCH_SIZE"):
        args += ["--ubatch-size", str(int(cfg["UBATCH_SIZE"]))]

    # --- flash attention ---
    flash = cfg.get("FLASH_ATTN")
    if flash and str(flash).lower() != "off":
        args += ["--flash-attn", str(flash)]

    # --- mmap / mlock ---
    if cfg.get("NO_MMAP"):
        args += ["--no-mmap"]
    # NOTE: --mlock is intentionally NOT auto-added. It requires Docker IPC
    # lock + LXC perms on containerized setups; let user add explicitly.

    return args, host, port, alias


# ────────────────────────────────────────────────────────────────────
# llama-server discovery
# ────────────────────────────────────────────────────────────────────


def find_llama_server() -> Path | None:
    """Locate llama-server binary through the shared cross-platform resolver."""
    for candidate in _binary_candidates("llama-server"):
        try:
            if candidate.is_file() and (IS_WINDOWS or os.access(candidate, os.X_OK)):
                return candidate.resolve()
        except OSError:
            continue
    return None


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


def _kill_process_tree(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        return
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    time.sleep(2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _server_popen_kwargs() -> dict:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def is_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def is_healthy(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────────
# Subcommands
# ────────────────────────────────────────────────────────────────────


def cmd_status() -> int:
    if not PIDFILE.exists():
        print("Not running (no PID file at", PIDFILE, ").")
        return 1
    pid = int(PIDFILE.read_text().strip())
    if not _pid_exists(pid):
        print(f"PID {pid} not found, cleaning up stale PID file.")
        PIDFILE.unlink(missing_ok=True)
        return 1

    cfg = parse_config(CONFIG)
    _, host, port, alias = build_args(cfg)
    print(f"Running: PID={pid} alias={alias} port={port} host={host}")
    if is_listening(host, port):
        ok = is_healthy(host, port)
        print(f"  Health: {'OK' if ok else 'NOT READY'}")
        print(f"  Endpoint:    http://{host}:{port}/v1")
        print(f"  Model name:  {alias}")
        print()
        print("Plug into Pi Agent / Hermes Agent / Claude Code:")
        print(f"  base_url: http://{host}:{port}/v1")
        print(f"  model:    {alias}")
    return 0


def cmd_stop() -> int:
    if not PIDFILE.exists():
        print("Not running.")
        return 1
    pid = int(PIDFILE.read_text().strip())
    try:
        if _pid_exists(pid):
            _kill_process_tree(pid)
            print(f"Killed PID {pid}")
        else:
            print(f"PID {pid} not found.")
    finally:
        PIDFILE.unlink(missing_ok=True)
    return 0


def cmd_serve() -> int:
    cfg = parse_config(CONFIG)
    args, host, port, alias = build_args(cfg)

    if is_listening(host, port):
        if is_healthy(host, port):
            print(f"Already up: alias={alias} port={port}")
            print(f"  Endpoint: http://{host}:{port}/v1")
            print(f"  Model name (OpenAI-compat API): {alias}")
            return 0
        else:
            print(f"Port {port} in use but server not healthy.")
            print("Stop first: python3 scripts/serve-config.py stop")
            return 1

    binary = find_llama_server()
    if not binary:
        print("ERROR: llama-server binary not found.")
        print()
        print("Resolution order checked:")
        print("  1. $AUTORESEARCH_LLAMA_CPP_ROOT/build-cuda/bin[/Release]/llama-server[.exe]")
        print("  2. $AUTORESEARCH_LLAMA_CPP_ROOT/build/bin[/Release]/llama-server[.exe]")
        print(f"  3. {REPO_ROOT}/llama.cpp/build-cuda/bin[/Release]/llama-server[.exe]")
        print(f"  4. {REPO_ROOT}/llama.cpp/build/bin[/Release]/llama-server[.exe]")
        print(f"  5. {REPO_ROOT.parent}/llama.cpp/build-cuda/bin[/Release]/llama-server[.exe]")
        print("  6. PATH llama-server[.exe]")
        print()
        print("Either clone llama.cpp (or a fork) into the repo root, or set")
        print("AUTORESEARCH_LLAMA_CPP_ROOT to the directory containing build-cuda/bin/.")
        return 2

    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "w")
    proc = subprocess.Popen(
        [str(binary)] + args,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **_server_popen_kwargs(),
        close_fds=True,
    )
    PIDFILE.write_text(str(proc.pid))
    print(f"Starting {alias} (PID {proc.pid})...")

    for _ in range(180):
        if is_healthy(host, port):
            print(f"OK {alias} ready at http://{host}:{port} (PID {proc.pid})")
            print(f"  Endpoint: http://{host}:{port}/v1")
            print(f"  Model name (OpenAI-compat API): {alias}")
            print()
            print("Plug into Pi Agent / Hermes Agent / Claude Code:")
            print(f"  base_url: http://{host}:{port}/v1")
            print(f"  model:    {alias}")
            return 0
        if proc.poll() is not None:
            print(f"FAIL: process exited code {proc.returncode}. Log: {LOG}")
            return 1
        time.sleep(1)

    print(f"WARN: server didn't become healthy in 180s. Check log: {LOG}")
    return 1


def cmd_print_cmd() -> int:
    cfg = parse_config(CONFIG)
    args, host, port, alias = build_args(cfg)
    binary = find_llama_server()
    print(f"# Binary: {binary or 'NOT FOUND — see scripts/setup-check.sh'}")
    print(f"# Endpoint: http://{host}:{port}/v1")
    print(f"# Alias (model name for OpenAI-compat clients): {alias}")
    print()
    if binary:
        parts = [str(binary)] + [repr(a) if " " in a else a for a in args]
        print(" ".join(parts))
    return 0


def cmd_print_config() -> int:
    cfg = parse_config(CONFIG)
    print(f"# Parsed from: {CONFIG}")
    print()
    for k, v in cfg.items():
        print(f"{k} = {v!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="start llama-server with current config.py")
    sub.add_parser("stop", help="kill running llama-server")
    sub.add_parser("status", help="show running state + connection info")
    sub.add_parser("print-cmd", help="print the resolved llama-server command (don't start)")
    sub.add_parser("print-config", help="print parsed config.py constants")

    args = parser.parse_args()
    if args.cmd == "stop":
        return cmd_stop()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "print-cmd":
        return cmd_print_cmd()
    if args.cmd == "print-config":
        return cmd_print_config()
    return cmd_serve()


if __name__ == "__main__":
    sys.exit(main() or 0)
