"""
Llama Server Runner

Encapsulates the lifecycle of a llama.cpp server process, including:
- ServerIntent parsing & command building
- Hardware locality optimization (MTP, VITRIOL/MoE)
- Port discovery & binding
- Health checking
- VRAM usage sampling
- Safe teardown
"""

import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
LLAMA_CPP_ROOT = Path(os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", ROOT_DIR / "llama.cpp"))
LLAMA_SERVER_CANDIDATES = (
    LLAMA_CPP_ROOT / "build-cuda" / "bin" / "llama-server",
    LLAMA_CPP_ROOT / "build" / "bin" / "llama-server",
)


@dataclass(frozen=True)
class ServerIntent:
    """A pure data object describing high-level benchmark intent."""
    model_path: Path
    ctx_size: int
    kv_cache: str
    flash_attn: str
    port: int = 18080
    batch_size: int = 512
    ubatch_size: int = 128
    threads: int = 8
    parallel: int = 1
    ngl: int = 999


def resolve_llama_server() -> Path:
    for candidate in LLAMA_SERVER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "llama-server not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_SERVER_CANDIDATES)
    )


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))


class LlamaServerRunner:
    def __init__(self, intent: ServerIntent, timeout: int = 300, log_path: Path | None = None):
        self.intent = intent
        self.timeout = timeout
        self.log_path = log_path
        
        self.port: int | None = None
        self.peak_vram_mb: float = 0.0
        
        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        self._vram_thread: threading.Thread | None = None
        
        self.llama_server = resolve_llama_server()

    def _build_cmd(self, target_port: int) -> list[str]:
        cmd = [
            str(self.llama_server),
            "--model", str(self.intent.model_path),
            "--host", "127.0.0.1",
            "--port", str(target_port),
            "--ctx-size", str(self.intent.ctx_size),
            "--batch-size", str(self.intent.batch_size),
            "--ubatch-size", str(self.intent.ubatch_size),
            "--threads", str(self.intent.threads),
            "--parallel", str(self.intent.parallel),
            "--n-gpu-layers", str(self.intent.ngl),
            "--cache-type-k", self.intent.kv_cache,
            "--cache-type-v", self.intent.kv_cache,
            "--flash-attn", self.intent.flash_attn,
        ]

        # MTP Optimization: Detect MTP models and enable built-in speculative decoding.
        if "MTP" in self.intent.model_path.name.upper():
            print(f"  [MTP] Multi-Token Prediction detected for {self.intent.model_path.name}. Enabling draft-mtp.")
            cmd += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "1",
                "--spec-draft-type-k", self.intent.kv_cache,
                "--spec-draft-type-v", self.intent.kv_cache,
            ]

        # VITRIOL Optimization: Hardware Necromancy for large MoE models.
        moe_indicators = ["MOE", "A3B", "A4B", "A1B", "A2B", "8X3B", "8X4B", "10B", "11B", "12B", "13B", "14B", "15B", "16B", "17B", "18B", "19B", "20B", "21B", "22B", "23B", "24B", "25B", "26B", "35B"]
        model_name_up = self.intent.model_path.name.upper()
        is_moe = any(ind in model_name_up for ind in moe_indicators)
        is_small_dense = any(f"-{x}B" in model_name_up for x in ["2", "4", "7", "8", "9"]) and not ("MOE" in model_name_up or "A1B" in model_name_up)

        if is_moe and not is_small_dense:
            print(f"  [VITRIOL] MoE Expert Streaming enabled for {self.intent.model_path.name}. Offloading experts to CPU.")
            cmd += ["--override-tensor", ".*exps.*=CPU"]

        return cmd

    def __enter__(self):
        self._start_vram_sampler()
        
        server_env = os.environ.copy()
        llama_lib_dir = str(self.llama_server.parent)
        existing = server_env.get("LD_LIBRARY_PATH", "")
        server_env["LD_LIBRARY_PATH"] = f"{llama_lib_dir}:{existing}" if existing else llama_lib_dir
        
        startup_tail: list[str] = []
        for port in candidate_ports(self.intent.port):
            cmd = self._build_cmd(port)
            print(f"Starting server: {' '.join(cmd)}")
            
            if self.log_path:
                self._server_log = open(self.log_path, "w+", encoding="utf-8")
            else:
                self._server_log = tempfile.NamedTemporaryFile(
                    mode="w+",
                    encoding="utf-8",
                    prefix="autoresearch-llama-server-",
                    suffix=".log",
                    delete=True,
                )

            self._server_proc = subprocess.Popen(
                cmd,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                env=server_env,
                text=True,
            )
            
            self.port = port
            if self._wait_for_server(port):
                return self
            
            # If wait failed, grab the tail before cleaning up
            self._server_log.flush()
            if hasattr(self._server_log, "name"):
                log_content = Path(self._server_log.name).read_text(encoding="utf-8", errors="replace")
                startup_tail = log_content.splitlines()[-50:]
                
            self._cleanup_process()
            print(f"Failed to start on port {port}, trying next...")
            
        self._cleanup_all()
        print("FAIL: Server crashed during startup.")
        if startup_tail:
            print("Tail of startup log:")
            print("\n".join(startup_tail))
        raise RuntimeError("Failed to start llama-server on any candidate port.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_all()

    def read_log(self) -> str:
        if self._server_log:
            self._server_log.flush()
            if hasattr(self._server_log, "name"):
                return Path(self._server_log.name).read_text(encoding="utf-8", errors="replace")
        return ""

    def _wait_for_server(self, port: int) -> bool:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self._server_proc.poll() is not None:
                return False
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                with urllib.request.urlopen(req, timeout=0.5) as response:
                    if response.status == 200:
                        return True
            except Exception:
                time.sleep(0.1)
        return False

    def _start_vram_sampler(self) -> None:
        def sampler() -> None:
            while not self._stop_event.is_set():
                try:
                    res = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                            "-i",
                            "0",
                        ],
                        text=True,
                    )
                    current = float(res.strip() or 0.0)
                    if current > self.peak_vram_mb:
                        self.peak_vram_mb = current
                except FileNotFoundError:
                    print("Error: nvidia-smi not found. VRAM sampling stopped.")
                    break
                except (subprocess.CalledProcessError, ValueError) as e:
                    pass
                self._stop_event.wait(0.2)

        self._vram_thread = threading.Thread(target=sampler, daemon=True)
        self._vram_thread.start()

    def _cleanup_process(self):
        if self._server_proc:
            self._server_proc.kill()
            self._server_proc.wait()
            self._server_proc = None
        if self._server_log:
            self._server_log.close()
            self._server_log = None

    def _cleanup_all(self):
        self._stop_event.set()
        if self._vram_thread:
            self._vram_thread.join(timeout=1.0)
        self._cleanup_process()