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

_LLAMA_SERVER_HELP_CACHE = None

ROOT_DIR = Path(__file__).resolve().parent
LLAMA_CPP_ROOT = Path(os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", "/home/shark/workspace/Nexus-System/llama.cpp"))
LLAMA_SERVER_CANDIDATES = (
    LLAMA_CPP_ROOT / "build-cuda" / "bin" / "llama-server",
    LLAMA_CPP_ROOT / "build" / "bin" / "llama-server",
    ROOT_DIR / "llama.cpp" / "build-cuda" / "bin" / "llama-server",
    ROOT_DIR / "llama.cpp" / "build" / "bin" / "llama-server",
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
    kv_cache_k: str | None = None
    kv_cache_v: str | None = None
    threads_batch: int | None = None
    spec_draft_n_max: int = 1
    no_mmap: bool = False
    jinja: bool = False
    reasoning_budget: int | None = None
    reasoning_budget_message: str | None = None
    reasoning: str | None = None
    cont_batching: bool = False
    host: str = "127.0.0.1"
    spec_type: str | None = None

# VRAM Estimation Constants (calibrated for f16 KV cache and typical systems)
VRAM_KB_PER_TOKEN_F16 = 80.0
"""Calibrated memory consumption per context token at f16 precision (in kilobytes)."""

VRAM_OVERHEAD_MB = 300.0
"""Typical baseline VRAM overhead for CUDA runtime and system operations (in megabytes)."""

VRAM_DEFAULT_QUANT_FACTOR = 0.3
"""Fallback quantization multiplier for unknown/default KV cache types."""

VRAM_QUANT_FACTORS = {
    "f16": 1.0,
    "f32": 1.0,
    "q8": 0.55,
    "q5": 0.38,
    "q4": 0.28,
    "turbo4": 0.18,
    "turbo3": 0.14,
    "turbo2": 0.10,
}
"""KV cache quantization type memory usage scaling factors relative to f16."""


def estimate_vram_mb(model_path: Path, ctx_size: int, kv_cache_k: str | None = None, kv_cache_v: str | None = None, base_kv_cache: str = "q4_0") -> float:
    try:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
    except Exception:
        model_size_mb = 4000.0
    
    try:
        c_size = int(ctx_size)
    except Exception:
        c_size = 16384
    
    base_kv = base_kv_cache if base_kv_cache is not None else "q4_0"
    k_type = kv_cache_k if kv_cache_k is not None else base_kv
    v_type = kv_cache_v if kv_cache_v is not None else base_kv
    
    def get_quant_factor(q_type: Any) -> float:
        if q_type is None or not isinstance(q_type, str):
            return VRAM_DEFAULT_QUANT_FACTOR
        q = q_type.lower()
        for key, factor in VRAM_QUANT_FACTORS.items():
            if key in q:
                return factor
        return VRAM_DEFAULT_QUANT_FACTOR
        
    kf = get_quant_factor(k_type)
    vf = get_quant_factor(v_type)
    
    # Calibrated KV cache size per token at f16 is ~80 KB
    kv_base_mb = c_size * VRAM_KB_PER_TOKEN_F16 / 1024.0
    kv_est_mb = (kv_base_mb / 2.0) * kf + (kv_base_mb / 2.0) * vf
    
    # Baseline system/CUDA overhead
    return model_size_mb + kv_est_mb + VRAM_OVERHEAD_MB



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
        cache_type_k = self.intent.kv_cache_k if self.intent.kv_cache_k is not None else self.intent.kv_cache
        cache_type_v = self.intent.kv_cache_v if self.intent.kv_cache_v is not None else self.intent.kv_cache

        cmd = [
            str(self.llama_server),
            "--model", str(self.intent.model_path),
            "--host", str(self.intent.host),
            "--port", str(target_port),
            "--ctx-size", str(self.intent.ctx_size),
            "--batch-size", str(self.intent.batch_size),
            "--ubatch-size", str(self.intent.ubatch_size),
            "--threads", str(self.intent.threads),
            "--parallel", str(self.intent.parallel),
            "--n-gpu-layers", str(self.intent.ngl),
            "--cache-type-k", cache_type_k,
            "--cache-type-v", cache_type_v,
            "--flash-attn", self.intent.flash_attn,
        ]

        if self.intent.threads_batch is not None:
            cmd += ["--threads-batch", str(self.intent.threads_batch)]

        if self.intent.no_mmap:
            cmd += ["--no-mmap"]
        if self.intent.jinja:
            cmd += ["--jinja"]
        if self.intent.reasoning_budget is not None:
            cmd += ["--reasoning-budget", str(self.intent.reasoning_budget)]
        if self.intent.reasoning_budget_message is not None:
            cmd += ["--reasoning-budget-message", self.intent.reasoning_budget_message]
        if self.intent.reasoning is not None:
            cmd += ["--reasoning", str(self.intent.reasoning)]
        if self.intent.cont_batching:
            cmd += ["--cont-batching"]

        # MTP/Speculative Optimization: Detect MTP models and enable speculative decoding.
        spec_type_val = self.intent.spec_type
        if spec_type_val is None and "MTP" in self.intent.model_path.name.upper() and self.intent.spec_draft_n_max > 0:
            global _LLAMA_SERVER_HELP_CACHE
            if _LLAMA_SERVER_HELP_CACHE is None:
                try:
                    # Query help to see if draft-mtp is supported
                    _LLAMA_SERVER_HELP_CACHE = subprocess.check_output([str(self.llama_server), "--help"], stderr=subprocess.STDOUT, text=True)
                except Exception:
                    _LLAMA_SERVER_HELP_CACHE = "mtp"
                if "mtp" in _LLAMA_SERVER_HELP_CACHE:
                    spec_type_val = "mtp"
                else:
                    spec_type_val = "draft-mtp"
            print(f"  [MTP] Multi-Token Prediction detected for {self.intent.model_path.name}. Auto-selected spec-type: {spec_type_val}")

        if spec_type_val is not None and spec_type_val.lower() != "none" and self.intent.spec_draft_n_max > 0:
            cmd += [
                "--spec-type", spec_type_val,
                "--spec-draft-n-max", str(self.intent.spec_draft_n_max),
                "--spec-draft-type-k", cache_type_k,
                "--spec-draft-type-v", cache_type_v,
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

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self._cleanup_all()

    def read_log(self) -> str:
        if self._server_log:
            self._server_log.flush()
            if hasattr(self._server_log, "name"):
                return Path(self._server_log.name).read_text(encoding="utf-8", errors="replace")
        return ""

    def _wait_for_server(self, port: int) -> bool:
        deadline = time.time() + self.timeout
        delay = 0.05
        while time.time() < deadline:
            if self._server_proc.poll() is not None:
                return False
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                with urllib.request.urlopen(req, timeout=0.5) as response:
                    if response.status == 200:
                        return True
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 0.4)
        return False

    def _start_vram_sampler(self) -> None:
        import ctypes
        
        # Load NVML library using ctypes
        nvml = None
        device = None
        class nvmlMemory_t(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_uint64),
                ("free", ctypes.c_uint64),
                ("used", ctypes.c_uint64),
            ]

        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            nvml.nvmlInit_v2()
            device = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device))
            print("  [VRAM] NVML initialized successfully. High-frequency 20ms sampling enabled.")
        except Exception:
            nvml = None
            print("  [VRAM] NVML initialization failed. Falling back to subprocess nvidia-smi (200ms).")

        def sampler() -> None:
            nonlocal nvml
            while not self._stop_event.is_set():
                if nvml is not None and device is not None:
                    try:
                        mem_info = nvmlMemory_t()
                        nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(mem_info))
                        current = float(mem_info.used) / (1024.0 * 1024.0)
                        if current > self.peak_vram_mb:
                            self.peak_vram_mb = current
                        self._stop_event.wait(0.02)
                        continue
                    except Exception:
                        nvml = None
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
                except (subprocess.CalledProcessError, ValueError):
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