"""Runtime measurements and shared Trial outcomes; no quality benchmark imports."""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from autoresearch.core.hardware import (
    detect_pid_gpu_shared_mb,
    detect_used_total_vram_mb,
    wait_gpu_near_idle,
)
from autoresearch.core.llama_runner import (
    WATCHDOG_KILL_LEGACY_MARKER,
    assert_flags_supported,
    dedicated_vram_kill_ceil,
    is_spec_enabled,
    load_mode_flag,
    resolve_llama_cli,
    resolve_llama_perplexity,
    resolve_shared_vram_limit_mb,
    resolve_vram_limit_mb,
    watchdog_kill_reason,
)
from autoresearch.core.model_arch import gguf_has_mtp
from autoresearch.core.process_guard import spawn_kwargs
from autoresearch.core.sglang_runner import run_sglang_bench_validation

BASE_DIR = Path(__file__).resolve().parents[2]
BENCH_N_PROMPT = 512
BENCH_N_GEN = 512
BENCH_CTX_CAP = 4096


class TrialOutcome(str, Enum):
    OK = "OK"
    INVALID_CONFIG = "INVALID_CONFIG"
    MODEL_REJECTED = "MODEL_REJECTED"
    # Runtime VRAM-policy kill (issue #72): the harness watchdog stopped a
    # healthy server on its VRAM budget — never a model OOM rejection.
    WATCHDOG_KILL = "WATCHDOG_KILL"
    # Environment/engine problem, not the model and not harness code: missing
    # binary, unprobeable output, or `FlagDriftError` (the engine build dropped
    # a flag the harness forwards). Triage as an engine bump, never a harness
    # defect — `CODE_ERROR` is reserved for our own logic.
    INFRA_ERROR = "INFRA_ERROR"
    CODE_ERROR = "CODE_ERROR"


@dataclass
class TrialResult:
    """Typed result of one trial. Replaces dict-as-return-type pattern.

    Every field has a safe default (0.0 or ""). Callers never need .get().
    """

    status: str = "OK"
    val_score: float = 0.0
    coding_val: float = 0.0
    coding_tps: float = 0.0
    lcb_val: float = 0.0
    he_val: float = 0.0
    mbpp_val: float = 0.0
    bigcode_val: float = 0.0
    swe_val: float = 0.0
    agentic_val: float = 0.0  # Claw-Eval quick/full tier score
    agentic_tier: str = ""  # "quick" or "full"
    agentic_task_count: int = 0  # number of tasks evaluated
    agentic_coding_val: float | None = None  # SWE-lite passed/N (ADR 0013); None = not measured
    agentic_coding_detail: str = ""
    mini_swe_agent_val: float | None = (
        None  # DM-Code-Agent passed/N via mini-swe-agent; None = not measured
    )
    mini_swe_agent_detail: str = ""
    avg_tps: float = 0.0
    peak_vram_gb: float = 0.0
    bench_tg_tps: float = 0.0
    bench_pp_tps: float = 0.0
    bench_ppl: float = 0.0
    elapsed_sec: float = 0.0
    outcome: TrialOutcome = TrialOutcome.OK
    diagnostic: str = ""
    task_ids: tuple[str, ...] = ()
    tps_source: str = ""
    gpu_temp_c: float | None = None
    tps_reps: list[float] = field(default_factory=list)
    tps_spread: float | None = None


def _tps_spread(values: list[float], median: float) -> float:
    if median == 0:
        return 0.0
    return (max(values) - min(values)) / median * 100.0


def median_llama_bench_validation(
    *args,
    reps: int = 3,
    idle_c: float | None = None,
    thermal_wait: bool = True,
    **kwargs,
) -> tuple[float, list[float], float | None]:
    reps = max(1, int(reps))
    values: list[float] = []
    last_temp: float | None = None
    for _ in range(reps):
        last_temp = wait_gpu_near_idle(idle_c=idle_c, enabled=thermal_wait)
        values.append(run_llama_bench_validation(*args, **kwargs))
    return statistics.median(values), values, last_temp


def median_sglang_bench_validation(
    *args,
    reps: int = 3,
    idle_c: float | None = None,
    thermal_wait: bool = True,
    **kwargs,
) -> tuple[float, list[float], float | None]:
    reps = max(1, int(reps))
    values: list[float] = []
    last_temp: float | None = None
    for _ in range(reps):
        last_temp = wait_gpu_near_idle(idle_c=idle_c, enabled=thermal_wait)
        values.append(run_sglang_bench_validation(*args, **kwargs))
    return statistics.median(values), values, last_temp


def run_llama_bench_validation(
    model_path: Path,
    ngl: int = 99,
    threads: int = 8,
    batch_size: int = 512,
    ubatch_size: int = 128,
    flash_attn: str = "on",
    cache_type_k: str = "q4_0",
    cache_type_v: str = "q4_0",
    ctx_size: int = 131072,
    threads_batch: int | None = None,
    no_mmap: bool = False,
    mlock: bool = False,
    cont_batching: bool = False,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
    spec_draft_model: str | None = None,
    n_cpu_moe: int | None = None,
    n_gen: int = BENCH_N_GEN,
    vram_limit_mb: float | int | None = None,
    reasoning: str | None = None,
) -> float:
    """Run llama-cli with given config. Returns tg t/s. Raises on failure.

    Caps ``-c`` at ``BENCH_CTX_CAP`` so TPS smoke does not allocate Trial-sized
    KV without a VRAM kill. Watches dedicated keepout ceil + absolute Shared GPU
    ceil (MoE+NO_MMAP can thrash Shared→pagefile while dedicated stays ~4–5 GB).
    Sets ``GGML_CUDA_NO_PINNED=1``. Carries ``NO_MMAP``/``MLOCK`` through
    ``load_mode_flag`` as an explicit ``--load-mode``.
    """
    llama_cli = resolve_llama_cli()
    bench_ctx = min(int(ctx_size), BENCH_CTX_CAP)
    if bench_ctx < int(ctx_size):
        print(
            f"  [cli-bench] ctx capped {ctx_size} -> {bench_ctx} "
            "(full Trial CTX only on monitored llama-server)",
            flush=True,
        )

    cmd = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-p",
        "Write a comprehensive, step-by-step tutorial explaining quantum computing, qubits, superposition, and entanglement, including a detailed Python simulation using NumPy.",
        "-n",
        str(n_gen),
        "-c",
        str(bench_ctx),
        "-t",
        str(threads),
        "-ngl",
        str(ngl),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-fa",
        flash_attn,
        "-ctk",
        cache_type_k,
        "-ctv",
        cache_type_v,
    ]
    cmd += load_mode_flag(no_mmap, mlock)
    cmd += [
        "--no-warmup",
        "--simple-io",
        "--single-turn",
        "--ignore-eos",
    ]

    if reasoning is not None:
        cmd += ["--reasoning", str(reasoning)]
    if threads_batch is not None:
        cmd += ["-tbd", str(threads_batch)]
    if n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]

    spec_type_val = spec_type
    if spec_type_val is None and gguf_has_mtp(model_path) and spec_draft_n_max > 0:
        spec_type_val = "draft-mtp"

    if is_spec_enabled(spec_type_val, spec_draft_n_max):
        cmd += [
            "--spec-type",
            spec_type_val.lower(),
        ]
        if spec_draft_n_max > 0:
            cmd += [
                "--spec-draft-n-max",
                str(spec_draft_n_max),
                "--spec-draft-type-k",
                cache_type_k,
                "--spec-draft-type-v",
                cache_type_v,
                "-ngld",
                str(ngl),
            ]
        if spec_draft_model:
            draft_path = Path(spec_draft_model)
            if not draft_path.is_absolute():
                draft_path = model_path.parent / draft_path
            cmd += ["--spec-draft-model", str(draft_path)]

    assert_flags_supported(cmd, llama_cli)
    print(f"  [cli-bench] {' '.join(str(a) for a in cmd)}")
    limit = resolve_vram_limit_mb(vram_limit_mb)
    shared_limit = resolve_shared_vram_limit_mb()
    stop = threading.Event()
    vram_killed = {"value": False, "reason": ""}
    cli_env = os.environ.copy()
    cli_env.setdefault("GGML_CUDA_NO_PINNED", "1")

    def _watch_vram(proc: subprocess.Popen[str]) -> None:
        while not stop.is_set():
            try:
                used, total = detect_used_total_vram_mb()
                ceil = dedicated_vram_kill_ceil(limit, total)
                if used > ceil:
                    vram_killed["value"] = True
                    vram_killed["reason"] = watchdog_kill_reason(
                        "nvml-used",
                        f"used={used:.0f}MB>ceil={ceil:.0f}MB",
                        f"llama-cli={model_path.name}",
                    )
                    print(
                        f"  [VRAM] LIMIT EXCEEDED used={used:.0f}MB > limit={ceil:.0f}MB "
                        f"(llama-cli={model_path.name}) — killing "
                        "(no Windows shared-GPU / pagefile spill)",
                        flush=True,
                    )
                    proc.kill()
                    stop.set()
                    return
                if proc.pid:
                    shared = detect_pid_gpu_shared_mb(int(proc.pid))
                    if shared is not None and shared > shared_limit:
                        vram_killed["value"] = True
                        vram_killed["reason"] = watchdog_kill_reason(
                            "shared",
                            f"shared={shared:.0f}MB>limit={shared_limit:.0f}MB",
                            f"llama-cli={model_path.name}",
                        )
                        print(
                            f"  [VRAM] SHARED GPU EXCEEDED shared={shared:.0f}MB > "
                            f"limit={shared_limit:.0f}MB (llama-cli={model_path.name}) — killing "
                            "(WDDM/PCI-e Shared→pagefile freeze)",
                            flush=True,
                        )
                        proc.kill()
                        stop.set()
                        return
            except FileNotFoundError:
                pass
            except (subprocess.CalledProcessError, ValueError, OSError):
                pass
            stop.wait(0.5)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=cli_env,
    )
    watcher = threading.Thread(target=_watch_vram, args=(proc,), daemon=True)
    watcher.start()
    try:
        stdout, stderr = proc.communicate()
    finally:
        stop.set()
        watcher.join(timeout=2.0)

    if vram_killed["value"]:
        raise RuntimeError(vram_killed["reason"] or WATCHDOG_KILL_LEGACY_MARKER)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode or -1, cmd, stdout or "", stderr or "")

    match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", stdout or "")
    if not match:
        match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", stderr or "")

    if not match:
        raise RuntimeError(
            f"llama-cli output did not contain Generation TPS metric: "
            f"{(stdout or '')[:500]} {(stderr or '')[:500]}"
        )

    return float(match.group(1))


def run_llama_perplexity_validation(
    model_path: Path,
    ngl: int = 99,
    threads: int = 8,
    batch_size: int = 512,
    ubatch_size: int = 128,
    flash_attn: str = "on",
    cache_type_k: str = "q4_0",
    cache_type_v: str = "q4_0",
    ctx_size: int = 2048,
    text_file: Path = BASE_DIR / "data/perplexity_val.txt",
    chunks: int = 1,
) -> float:
    """Run llama-perplexity over the validation text and return the resulting float score."""
    llama_ppl = resolve_llama_perplexity()

    cmd = [
        str(llama_ppl),
        "-m",
        str(model_path),
        "-f",
        str(text_file),
        "-t",
        str(threads),
        "-ngl",
        str(ngl),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-fa",
        flash_attn,
        "-ctk",
        cache_type_k,
        "-ctv",
        cache_type_v,
        "-c",
        str(ctx_size),
        "--chunks",
        str(chunks),
    ]

    print(f"  [perplexity] {' '.join(str(a) for a in cmd)}")
    result = subprocess.run(
        **spawn_kwargs(args=cmd, capture_output=True, text=True, encoding="utf-8")
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    import re

    full_output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", full_output)
    if not match:
        # Fallback: parse single chunk perplexity like [1]5.8806,
        chunk_match = re.search(r"\[\d+\]\s*([0-9.]+)", full_output)
        if chunk_match:
            return float(chunk_match.group(1))
        raise RuntimeError(
            f"Could not parse perplexity from output. stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}"
        )

    return float(match.group(1))
