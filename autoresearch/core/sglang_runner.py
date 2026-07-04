import os
import subprocess
import time
import requests
import socket
import threading
from pathlib import Path
from typing import Any

from autoresearch.core.llama_runner import ServerIntent, ROOT_DIR

IS_WINDOWS = os.name == "nt"
REPO_ROOT = ROOT_DIR.parent.parent
SGLANG_BIN = REPO_ROOT / "venv-sglang" / ("Scripts" if IS_WINDOWS else "bin")
SGLANG_PYTHON = SGLANG_BIN / ("python.exe" if IS_WINDOWS else "python3")


def _popen_group_kwargs() -> dict[str, Any]:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
        return
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))

def run_sglang_bench_validation(
    model_path: Path,
    batch_size: int,
    n_prompt: int,
    n_gen: int,
) -> float:
    # --- Guard against 8GB VRAM OOM on large models ---
    model_name = model_path.name.upper()
    is_large_model = "35B" in model_name or "32B" in model_name
    if is_large_model:
        vram_gb: float | None = None
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass

        if vram_gb is None:
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                vram_gb = float(res.stdout.splitlines()[0].strip()) / 1024
            except Exception:
                # This benchmark path has already crashed WSL on 8GB GPUs. If
                # VRAM cannot be queried, fail closed for 32B/35B SGLang models.
                vram_gb = 0.0

        if vram_gb < 10.0:
            raise RuntimeError(
                f"SGLang disabled for {model_path.name}: {vram_gb:.1f}GB VRAM "
                "detected for a 32B/35B model. Refusing bench/server validation "
                "to prevent WSL crash."
            )

    print(f"  [bench] Running sglang.bench_one_batch for {model_path.name}")
    cmd = [
        str(SGLANG_PYTHON),
        "-m", "sglang.bench_one_batch",
        "--model-path", str(model_path),
        "--batch-size", str(batch_size),
        "--input-len", str(n_prompt),
        "--output-len", str(n_gen),
        "--dtype", "float16",
    ]
    if "GPTQ" in model_path.name.upper():
        cmd += ["--quantization", "gptq_marlin"]
    if "AWQ" in model_path.name.upper():
        cmd += ["--quantization", "awq"]

    server_env = os.environ.copy()
    sglang_bin = str(SGLANG_BIN)
    server_env["PATH"] = f"{sglang_bin}{os.pathsep}{server_env.get('PATH', '')}"
    server_env["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    server_env["SGLANG_MAMBA_SSM_DTYPE"] = "float16"

    try:
        res = subprocess.run(cmd, env=server_env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"sglang.bench_one_batch failed:\n{e.stderr}")
        raise

    # Parse output, looking for "Decode token/s:" or "Prefill token/s:" or similar.
    # We want to return the decode tokens per second (tg).
    tg_tps = 0.0
    for line in res.stdout.splitlines():
        if "Decode token/s:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    # Strip spaces and "tokens/s" if present, though split by ':' might just leave " 45.2"
                    tg_tps = float(parts[1].split()[0].replace("tokens/s", "").strip())
                except ValueError:
                    pass
    return tg_tps



class SGLangServerRunner:
    def __init__(self, intent: ServerIntent, timeout: int = 300, log_path: Path | None = None):
        self.intent = intent
        self.timeout = timeout
        self.log_path = log_path
        
        self.port: int | None = None
        self.peak_vram_mb: float = 0.0
        
        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        
    def _build_cmd(self, target_port: int) -> list[str]:
        print(f"  [SGLang] Directory detected. Using SGLang backend for {self.intent.model_path.name}")
        cmd = [
            str(SGLANG_PYTHON),
            "-m", "sglang.launch_server",
            "--model-path", str(self.intent.model_path),
            "--served-model-name", self.intent.model_path.name,
            "--host", str(self.intent.host),
            "--port", str(target_port),
            "--context-length", str(self.intent.ctx_size),
            "--mem-fraction-static", "0.8",
            "--cpu-offload-gb", "12",
            "--trust-remote-code",
            "--dtype", "float16",
            "--disable-cuda-graph",
        ]
        if "GPTQ" in self.intent.model_path.name.upper():
            cmd += ["--quantization", "gptq_marlin"]
        if "AWQ" in self.intent.model_path.name.upper():
            cmd += ["--quantization", "awq"]
        return cmd

    def is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.intent.host, port)) == 0

    def is_server_ready(self, port: int) -> bool:
        try:
            r = requests.get(f"http://{self.intent.host}:{port}/v1/models", timeout=1.0)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def start(self) -> int:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log = open(self.log_path, "w", encoding="utf-8")
        else:
            self._server_log = subprocess.DEVNULL

        server_env = os.environ.copy()
        sglang_bin = str(SGLANG_BIN)
        server_env["PATH"] = f"{sglang_bin}{os.pathsep}{server_env.get('PATH', '')}"
        server_env["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
        server_env["SGLANG_MAMBA_SSM_DTYPE"] = "float16"
        
        startup_tail: list[str] = []
        for port in candidate_ports(self.intent.port):
            cmd = self._build_cmd(port)
            if self.is_port_in_use(port):
                continue
            
            print(f"Starting server: {' '.join(cmd)}")
            self._server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=server_env,
                **_popen_group_kwargs(),
            )

            start_time = time.time()
            ready = False
            
            while time.time() - start_time < self.timeout:
                if self._server_proc.poll() is not None:
                    break
                
                try:
                    line = self._server_proc.stdout.readline()  # type: ignore
                    if line:
                        if self.log_path:
                            self._server_log.write(line)
                            self._server_log.flush()
                        startup_tail.append(line.rstrip())
                        if len(startup_tail) > 20:
                            startup_tail.pop(0)
                        
                        if "Uvicorn running on" in line or "The server is fired up and ready to roll!" in line:
                            ready = True
                            break
                except Exception:
                    pass
                
                if self.is_server_ready(port):
                    ready = True
                    break
                    
                time.sleep(0.5)

            if ready:
                self.port = port
                
                def consume_output():
                    try:
                        for log_line in self._server_proc.stdout:  # type: ignore
                            if self.log_path:
                                self._server_log.write(log_line)
                                self._server_log.flush()
                    except Exception:
                        pass
                
                threading.Thread(target=consume_output, daemon=True).start()
                return port
                
            else:
                if self._server_proc.poll() is None:
                    print(f"Failed to start on port {port}, trying next...")
                    _terminate_process_tree(self._server_proc)
                    self._server_proc.wait(timeout=5)
                else:
                    print("FAIL: Server crashed during startup.")
                    print("Tail of startup log:")
                    print("\n".join(startup_tail))
                    break

        raise RuntimeError("Failed to start SGLang server on any candidate port.")

    def stop(self) -> None:
        self._stop_event.set()
        
        if self._server_proc and self._server_proc.poll() is None:
            _terminate_process_tree(self._server_proc)
            
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
                
        if self._server_log and self._server_log != subprocess.DEVNULL:
            try:
                self._server_log.close()
            except Exception:
                pass

    def __enter__(self) -> "SGLangServerRunner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.stop()
