"""
Llama Server Runner

Encapsulates the lifecycle of a llama.cpp server process, including:
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
from pathlib import Path
from typing import Any


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))


class LlamaServerRunner:
    def __init__(self, cmd_base: list[str], server_env: dict[str, str], preferred_port: int, timeout: int = 300, log_path: Path | None = None):
        self.cmd_base = cmd_base
        self.server_env = server_env
        self.preferred_port = preferred_port
        self.timeout = timeout
        self.log_path = log_path
        
        self.port: int | None = None
        self.peak_vram_mb: float = 0.0
        
        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        self._vram_thread: threading.Thread | None = None

    def __enter__(self):
        self._start_vram_sampler()
        
        startup_tail: list[str] = []
        for port in candidate_ports(self.preferred_port):
            # Exclude existing --port arguments if accidentally passed
            filtered_cmd = []
            skip_next = False
            for arg in self.cmd_base:
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--port":
                    skip_next = True
                    continue
                filtered_cmd.append(arg)
                
            cmd = filtered_cmd + ["--port", str(port)]
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
                env=self.server_env,
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