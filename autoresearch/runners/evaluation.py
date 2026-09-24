"""Experiment runner — owns the full trial lifecycle.

Deep module extracted from run.py. One interface (run_trial), typed TrialResult,
hides bench validation, server lifecycle, and metric computation behind the seam.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoresearch.benchmarks.agentic_benchmarks import get_full_tier_tasks, get_quick_tier_tasks
from autoresearch.benchmarks.agentic_coding.runner import run_agentic_coding_eval
from autoresearch.benchmarks.agentic_runner import run_agentic_eval
from autoresearch.benchmarks.benchmark_coding import run_benchmark as run_coding
from autoresearch.benchmarks.mini_swe_agent.runner import run_mini_swe_agent_eval
from autoresearch.core import config as core_config
from autoresearch.core.hardware import (
    detect_gpu_temp_c,
    detect_pid_gpu_shared_mb,
    detect_used_total_vram_mb,
    wait_gpu_near_idle,
)
from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.core.llama_runner import (
    WATCHDOG_KILL_LEGACY_MARKER,
    WATCHDOG_KILL_LEGACY_REASON,
    ConfigError,
    LlamaServerRunner,
    ServerIntent,
    dedicated_vram_kill_ceil,
    is_spec_enabled,
    preflight_host_memory_for_intent,
    preflight_vram_for_intent,
    resolve_llama_cli,
    resolve_llama_perplexity,
    resolve_shared_vram_limit_mb,
    resolve_vram_limit_mb,
    watchdog_kill_reason,
)
from autoresearch.core.model_arch import gguf_block_count, gguf_has_mtp, gguf_is_moe
from autoresearch.core.sglang_runner import SGLangServerRunner, run_sglang_bench_validation

BASE_DIR = Path(__file__).resolve().parent

# ── per-run artifact rotation ───────────────────────────────────────────
LOG_DIR = BASE_DIR / "logs"
LOG_KEEP = 20


def _prune_glob(directory: Path, pattern: str, keep: int) -> None:
    """Delete oldest matching files, keeping the `keep` newest. Missing dir → no-op."""
    if not directory.exists():
        return
    paths = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    for path in paths[:-keep] if keep > 0 else paths:
        try:
            path.unlink()
        except OSError:
            pass


def _new_server_log(model_filename: str) -> Path:
    """Unique per-run llama-server log path under LOG_DIR (keep last LOG_KEEP)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_glob(LOG_DIR, "llama-server-*.log", LOG_KEEP - 1)  # leave room for the new file
    stem = Path(model_filename).stem if model_filename else "unknown"
    return LOG_DIR / f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}-{stem}.log"


# ── llama-bench defaults ────────────────────────────────────────────────
# Fallback only when Baseline omits TPS_FLOOR (legacy local config.py).
BENCH_TPS_THRESHOLD = 20.0
BENCH_N_PROMPT = 512
BENCH_N_GEN = 512
# TPS probe only — full Trial CTX loads on monitored llama-server. Uncapped
# llama-cli previously allocated Trial CTX with no VRAM kill → WDDM shared spill.
BENCH_CTX_CAP = 4096


def _format_arch_line(intent: ServerIntent) -> str:
    """One-line dense/MoE + n-cpu-moe mode before any bench/eval work."""
    path = intent.model_path
    is_moe = False
    block_count: int | None = None
    if path.is_file():
        try:
            is_moe = gguf_is_moe(path)
        except Exception:
            is_moe = False
        try:
            block_count = gguf_block_count(path)
        except Exception:
            block_count = None
    if not is_moe:
        mode = "dense"
    elif intent.n_cpu_moe_auto:
        mode = "auto"
    else:
        mode = "explicit"
    kind = "moe" if is_moe else "dense"
    bc = str(block_count) if block_count is not None else "?"
    n = "None" if intent.n_cpu_moe is None else str(intent.n_cpu_moe)
    return f"  [arch] {kind} block_count={bc} n-cpu-moe={n} ({mode})"


def _moe_vram_reject(intent: ServerIntent, est_mb: float, limit_mb: float) -> str | None:
    """Explicit reject when MoE model exceeds physical VRAM budget."""
    if est_mb <= limit_mb:
        return None
    path = intent.model_path
    if not path.is_file():
        return None
    try:
        if not gguf_is_moe(path):
            return None
    except Exception:
        return None
    if intent.n_cpu_moe == 0:
        return (
            f"MoE full-GPU (N_CPU_MOE=0) est={est_mb:.0f}MB > limit={limit_mb:.0f}MB; "
            "set N_CPU_MOE=None for auto block_count offload"
        )
    return (
        f"MoE offload (N_CPU_MOE={intent.n_cpu_moe}) est={est_mb:.0f}MB > limit={limit_mb:.0f}MB; "
        "exceeds physical VRAM limit"
    )


def resolve_tps_floor(norm: dict[str, Any] | None = None) -> float:
    """TPS Floor from Trial norm / Baseline. User sets ENGINE_DEFAULTS['TPS_FLOOR']."""
    if norm:
        for key in ("bench_tts_threshold", "tps_floor"):
            if key in norm and norm[key] is not None:
                value = float(norm[key])
                return value if math.isfinite(value) else BENCH_TPS_THRESHOLD
    try:
        value = float(core_config.DEFAULTS.get("TPS_FLOOR", BENCH_TPS_THRESHOLD))
        return value if math.isfinite(value) else BENCH_TPS_THRESHOLD
    except Exception:
        return BENCH_TPS_THRESHOLD


class TrialOutcome(str, Enum):
    OK = "OK"
    INVALID_CONFIG = "INVALID_CONFIG"
    MODEL_REJECTED = "MODEL_REJECTED"
    # Runtime VRAM-policy kill (issue #72): the harness watchdog stopped a
    # healthy server on its VRAM budget — never a model OOM rejection.
    WATCHDOG_KILL = "WATCHDOG_KILL"
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
    Sets ``GGML_CUDA_NO_PINNED=1``. Keeps ``NO_MMAP``.
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
        "--no-mmap" if no_mmap else "--mmap",
    ]
    if mlock:
        cmd += ["--mlock"]
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
    text_file: Path = BASE_DIR / "../../data/perplexity_val.txt",
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
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
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


def _watchdog_reason(runner: Any) -> str:
    """Scoped policy-kill reason from a runner, with legacy fallback.

    Live runners carry ``vram_kill_reason`` (``WATCHDOG_KILL scope=...``).
    Anything else (pre-fix messages, bare mocks) maps to the legacy
    NVML device-wide scope so the row still lands honest.
    """
    raw = getattr(runner, "vram_kill_reason", "")
    if isinstance(raw, str) and raw.startswith("WATCHDOG_KILL"):
        return raw
    return WATCHDOG_KILL_LEGACY_REASON


def _apply_watchdog_kill(res: TrialResult, runner: Any) -> None:
    """Record a runtime VRAM-policy kill as WATCHDOG_KILL (issue #72)."""
    reason = _watchdog_reason(runner)
    res.status = f"FAIL: {reason}"
    res.outcome = TrialOutcome.WATCHDOG_KILL
    res.diagnostic = reason
    res.val_score = 0.0
    res.peak_vram_gb = max(getattr(runner, "peak_vram_mb", 0.0), 0.0) / 1024.0


class ExperimentRunner:
    """Deep module: owns the full trial lifecycle.

    run_trial() is the sole external seam. Callers get a typed TrialResult
    without managing:
    - llama-bench validation
    - llama-server lifecycle (start, health check, teardown)
    - benchmark orchestration
    - VRAM tracking
    - metric computation

    Leverage: one interface, N callers (CLI and autoloop).
    Locality: trial orchestration logic and bugs concentrate in one module.
    """

    def __init__(self, models_dir: Path, thermal_wait: bool = True):
        self.models_dir = Path(models_dir)
        self.thermal_wait = thermal_wait
        self._idle_gpu_c = detect_gpu_temp_c()

    def run_trial(
        self,
        config: dict | Any,
        skip_bench: bool = False,
        **overrides,
    ) -> TrialResult:
        """Run one complete trial. Returns typed TrialResult."""
        # Normalize non-dict config to dict (argparse Namespace, MagicMock, etc.)
        if isinstance(config, dict):
            cfg_dict = config
        else:
            try:
                cfg_dict = dict(vars(config))
            except Exception:
                cfg_dict = {}
        res = TrialResult()
        try:
            intent, norm = ServerIntent.from_config(cfg_dict, self.models_dir, **overrides)
        except ConfigError as exc:
            res.status = f"FAIL: {exc}"
            res.outcome = TrialOutcome.INVALID_CONFIG
            res.diagnostic = str(exc)
            return res
        model_filename = intent.model_path.name
        print(_format_arch_line(intent))
        vram_limit_mb = resolve_vram_limit_mb(norm.get("vram_limit_mb"))

        ok_vram, est_vram, vram_reason = preflight_vram_for_intent(
            intent, vram_limit_mb, headroom_mb=norm.get("VRAM_HEADROOM_MB")
        )
        if vram_reason:
            print(f"  [vram-preflight] est={est_vram:.0f}MB ok={ok_vram} — {vram_reason}")
        else:
            print(
                f"  [vram-preflight] est={est_vram:.0f}MB limit={vram_limit_mb:.0f}MB ok={ok_vram}"
            )
        if not ok_vram:
            moe_reject = _moe_vram_reject(intent, est_vram, vram_limit_mb)
            reason = moe_reject or vram_reason
            res.status = f"FAIL: {reason}"
            res.outcome = TrialOutcome.MODEL_REJECTED
            res.diagnostic = reason
            res.peak_vram_gb = est_vram / 1024.0
            return res

        ok_host, est_host, budget_host, host_reason = preflight_host_memory_for_intent(
            intent,
            headroom_mb=norm.get("host_memory_headroom_mb", norm.get("HOST_MEMORY_HEADROOM_MB")),
        )
        print(f"  [host-preflight] est={est_host:.0f}MB budget={budget_host:.0f}MB ok={ok_host}")
        if not ok_host:
            res.status = f"FAIL: {host_reason}"
            res.outcome = TrialOutcome.MODEL_REJECTED
            res.diagnostic = host_reason
            res.peak_vram_gb = est_vram / 1024.0
            return res

        max_tokens = norm.get("max_tokens", 1024)
        include_coding = bool(norm.get("include_coding", False))
        task_limit_val = norm.get("coding_task_limit", 10)
        lcb_limit_val = norm.get("lcb_task_limit", 10)
        bigcode_limit_val = norm.get("bigcode_task_limit", 10)
        bench_tts_threshold = resolve_tps_floor(norm)
        is_validation = norm.get("validation", False)
        # Accept both CLI keys (agentic_*) and bench_config keys (include_agentic_*).
        agentic_quick = bool(
            norm.get("agentic_quick", False) or norm.get("include_agentic_quick", False)
        )
        agentic_full = bool(
            norm.get("agentic_full", False) or norm.get("include_agentic_full", False)
        )
        agentic_coding = bool(
            norm.get("agentic_coding", False) or norm.get("include_agentic_coding", False)
        )
        mini_swe_agent = bool(
            norm.get("mini_swe_agent", False) or norm.get("include_mini_swe_agent", False)
        )
        if is_validation:
            # Validation defaults to Claw quick smoke; respect an explicit
            # --no-agentic-quick so load/TPS-only smoke works without the
            # optional claw-eval vendor tree.
            if not (agentic_quick or agentic_full):
                agentic_quick = True
            agentic_full = False
            agentic_coding = False
            mini_swe_agent = False
            include_coding = (
                False  # validation = smoke gates only; coding-10 is canonical-Trial work
            )
        if mini_swe_agent:
            if agentic_quick or agentic_full or agentic_coding or include_coding:
                print("  [mini-swe-agent] standalone selection: disabling other benchmarks")
            include_coding = False
            agentic_quick = False
            agentic_full = False
            agentic_coding = False
        if include_coding and (task_limit_val, lcb_limit_val, bigcode_limit_val) != (10, 10, 10):
            res.status = "FAIL: Coding preflight requires exactly 10 tasks per dataset"
            res.outcome = TrialOutcome.INVALID_CONFIG
            res.diagnostic = res.status[6:]
            return res
        agentic_tiers: list[tuple[str, list[str]]] = []
        if agentic_quick:
            agentic_tiers.append(("quick", get_quick_tier_tasks()))
        if agentic_full:
            agentic_tiers.append(("full", get_full_tier_tasks()))
        if any(not task_ids for _, task_ids in agentic_tiers):
            missing = next(tier for tier, task_ids in agentic_tiers if not task_ids)
            res.status = f"FAIL: No agentic {missing} tasks found"
            res.outcome = TrialOutcome.INFRA_ERROR
            res.diagnostic = res.status[6:]
            return res

        # ── Pre-check: llama-bench validation ────────────────────────────
        if not skip_bench:
            if intent.model_path.is_dir():
                try:
                    reps = int(
                        cfg_dict.get("TPS_REPS", core_config.ENGINE_DEFAULTS.get("TPS_REPS", 3))
                        or 3
                    )
                    wait_gpu_near_idle(idle_c=self._idle_gpu_c, enabled=self.thermal_wait)
                    bench_tg, rep_vals, gpu_c = median_sglang_bench_validation(
                        model_path=intent.model_path,
                        batch_size=1,  # We use 1 here for bench
                        n_prompt=BENCH_N_PROMPT,
                        n_gen=BENCH_N_GEN,
                        reps=reps,
                        idle_c=self._idle_gpu_c,
                        thermal_wait=self.thermal_wait,
                    )
                    res.bench_tg_tps = bench_tg
                    res.tps_reps = rep_vals
                    res.tps_spread = _tps_spread(rep_vals, bench_tg)
                    res.gpu_temp_c = gpu_c
                    print(f"  [bench] sglang.bench_one_batch tg {BENCH_N_GEN}: {bench_tg:.1f} t/s")

                    if bench_tg < bench_tts_threshold:
                        print(
                            f"  [FAIL] sglang bench tg {bench_tg:.1f} t/s below threshold {bench_tts_threshold:.1f}"
                        )
                        res.status = f"FAIL: sglang bench tg {bench_tg:.1f} < threshold {bench_tts_threshold:.1f}"
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res

                    if is_validation:
                        print(
                            f"  [OK] SGLang bench validation passed: tg {bench_tg:.1f} t/s >= {bench_tts_threshold:.1f}"
                        )
                except Exception as e:
                    print(f"  [FAIL] sglang bench error: {e}")
                    res.status = f"FAIL: sglang bench error: {str(e)[:50]}"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res
            else:
                try:
                    reps = int(
                        cfg_dict.get("TPS_REPS", core_config.ENGINE_DEFAULTS.get("TPS_REPS", 3))
                        or 3
                    )
                    wait_gpu_near_idle(idle_c=self._idle_gpu_c, enabled=self.thermal_wait)
                    bench_tg, rep_vals, gpu_c = median_llama_bench_validation(
                        model_path=intent.model_path,
                        ngl=intent.ngl,
                        threads=intent.threads,
                        batch_size=intent.batch_size,
                        ubatch_size=intent.ubatch_size,
                        flash_attn=intent.flash_attn,
                        cache_type_k=intent.kv_cache_k or intent.kv_cache,
                        cache_type_v=intent.kv_cache_v or intent.kv_cache,
                        ctx_size=intent.ctx_size,
                        threads_batch=intent.threads_batch,
                        no_mmap=intent.no_mmap,
                        mlock=intent.mlock,
                        cont_batching=intent.cont_batching,
                        spec_type=intent.spec_type,
                        spec_draft_n_max=intent.spec_draft_n_max,
                        spec_draft_model=intent.spec_draft_model,
                        n_cpu_moe=intent.n_cpu_moe,
                        n_gen=BENCH_N_GEN,
                        vram_limit_mb=vram_limit_mb,
                        reasoning=intent.reasoning,
                        reps=reps,
                        idle_c=self._idle_gpu_c,
                        thermal_wait=self.thermal_wait,
                    )
                except FileNotFoundError as e:
                    print(f"  [FAIL] llama-cli not found: {e}")
                    res.status = "FAIL: llama-cli not found"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res
                except subprocess.CalledProcessError as e:
                    err_tail = (e.stderr or "").strip()[-800:]
                    print(f"  [FAIL] llama-cli crashed: {e}")
                    if err_tail:
                        print(f"  [stderr] {err_tail}")
                    res.status = "FAIL: llama-cli crashed"
                    res.diagnostic = err_tail or str(e)
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    return res
                except RuntimeError as e:
                    msg = str(e)
                    if "WATCHDOG_KILL" in msg or WATCHDOG_KILL_LEGACY_MARKER in msg:
                        reason = (
                            msg if msg.startswith("WATCHDOG_KILL") else WATCHDOG_KILL_LEGACY_REASON
                        )
                        print(f"  [FAIL] llama-cli {reason}")
                        res.status = f"FAIL: {reason}"
                        res.outcome = TrialOutcome.WATCHDOG_KILL
                        res.diagnostic = reason
                        return res
                    print(f"  [FAIL] llama-cli error: {e}")
                    res.status = f"FAIL: llama-cli error: {str(e)[:50]}"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res
                except Exception as e:
                    print(f"  [FAIL] llama-cli error: {e}")
                    res.status = f"FAIL: llama-cli error: {str(e)[:50]}"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res

                res.bench_tg_tps = bench_tg
                res.tps_reps = rep_vals
                res.tps_spread = _tps_spread(rep_vals, bench_tg)
                if gpu_c is not None:
                    res.gpu_temp_c = gpu_c
                print(f"  [bench] tg {BENCH_N_GEN}: {bench_tg:.1f} t/s")

                if bench_tg < bench_tts_threshold:
                    print(
                        f"  [FAIL] llama-cli tg {bench_tg:.1f} t/s below threshold {bench_tts_threshold:.1f}"
                    )
                    res.status = (
                        f"FAIL: bench tg {bench_tg:.1f} < threshold {bench_tts_threshold:.1f}"
                    )
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    return res

                if is_validation:
                    print(
                        f"  [OK] Bench validation passed: tg {bench_tg:.1f} t/s >= {bench_tts_threshold:.1f}"
                    )
                    # Fall through to the configured agentic smoke validation.

        # ── Perplexity validation ────────────────────────────────────────
        include_perplexity = bool(norm.get("include_perplexity", False))
        if include_perplexity:
            try:
                ppl = run_llama_perplexity_validation(
                    model_path=intent.model_path,
                    ngl=intent.ngl,
                    threads=intent.threads,
                    batch_size=intent.batch_size,
                    ubatch_size=intent.ubatch_size,
                    flash_attn=intent.flash_attn,
                    cache_type_k=intent.kv_cache_k or intent.kv_cache,
                    cache_type_v=intent.kv_cache_v or intent.kv_cache,
                    ctx_size=2048,
                )
                res.bench_ppl = ppl
                print(f"  [perplexity] PPL: {ppl:.4f}")
            except Exception as e:
                print(f"  [FAIL] Perplexity validation failed: {e}")
                res.status = f"FAIL: Perplexity failed: {str(e)[:50]}"
                res.outcome = TrialOutcome.INFRA_ERROR
                res.diagnostic = str(e)
                return res

        # ── Full evaluation ──────────────────────────────────────────────
        # Per-run log (rotated; the old shared llama_server.log got wiped by
        # every Trial, destroying post-mortem evidence like HTTP 400 bodies).
        server_log = _new_server_log(model_filename)
        print(f"  [server-log] {server_log}")

        reasoning_effort = norm.get("reasoning_effort")
        chat_template_kwargs = norm.get("chat_template_kwargs") or norm.get("chat_template_args")
        stop_val = norm.get("stop") or norm.get("stop_tokens")
        model_identifiers = [
            model_filename.lower(),
            str(intent.model_path).lower(),
            str(norm.get("model") or "").lower(),
        ]
        if stop_val is None and any(
            needle in identifier
            for needle in ("k2-horizon", "k2_horizon", "k2horizon")
            for identifier in model_identifiers
        ):
            stop_val = ["<|ifm|im_end|>", "</s>"]
        elif isinstance(stop_val, str):
            stop_val = [stop_val]
        elif isinstance(stop_val, (tuple, set)):
            stop_val = list(stop_val)

        gen_params = GenerationParams(
            temp=norm.get("temp", 0.2),
            top_p=norm.get("top_p"),
            min_p=norm.get("min_p"),
            top_k=norm.get("top_k"),
            repeat_penalty=norm.get("repeat_penalty"),
            presence_penalty=norm.get("presence_penalty"),
            frequency_penalty=norm.get("frequency_penalty"),
            # GenerationParams defaults max_tokens=512; agentic loops need headroom
            # for tool turns + thinking models (reasoning_content) or graders see "".
            # Floor 4096: reasoning models (Ornith-1.5) exhaust 2048 mid-<think>
            # and starve the answer; server log showed n_decoded == 2048 exactly.
            max_tokens=(
                max(int(norm.get("max_tokens", 1024)), 4096)
                if (agentic_quick or agentic_full or agentic_coding or mini_swe_agent)
                else int(norm.get("max_tokens", 1024))
            ),
            stop=stop_val,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
        )

        trial_start = time.time()
        if (
            include_perplexity
            and not include_coding
            and not agentic_quick
            and not agentic_full
            and not agentic_coding
            and not mini_swe_agent
        ):
            res.avg_tps = res.bench_tg_tps
            res.tps_source = "backend-bench"
            if res.avg_tps < bench_tts_threshold:
                res.val_score = 0.0
                res.outcome = TrialOutcome.MODEL_REJECTED
                return res
            if include_perplexity:
                res.val_score = res.avg_tps
            else:
                res.val_score = 0.0

            # Server was not started; preserve the effective preflight decision.
            res.peak_vram_gb = est_vram / 1024.0
            res.elapsed_sec = time.time() - trial_start
            return res

        runner = None
        try:
            gpu_c = wait_gpu_near_idle(idle_c=self._idle_gpu_c, enabled=self.thermal_wait)
            if gpu_c is not None:
                res.gpu_temp_c = gpu_c
            runner_cls = SGLangServerRunner if intent.model_path.is_dir() else LlamaServerRunner
            runner_kwargs: dict[str, Any] = {"log_path": server_log}
            if runner_cls is LlamaServerRunner:
                runner_kwargs["vram_limit_mb"] = vram_limit_mb
            runner = runner_cls(intent, **runner_kwargs)
            with runner as entered_runner:
                runner = entered_runner
                if getattr(runner, "vram_killed", False) is True:
                    _apply_watchdog_kill(res, runner)
                    return res
                raw_turn_timeout = norm.get("turn_timeout") or norm.get("timeout") or 420.0
                turn_timeout = max(float(raw_turn_timeout), 420.0)
                client = LlamaClient(runner.port, timeout=turn_timeout)

                # Coding (HumanEval + MBPP + LCB + BigCode)
                if include_coding:
                    print(
                        f"  [coding] Running (limit={task_limit_val}, "
                        f"lcb_limit={lcb_limit_val}, bigcode_limit={bigcode_limit_val})..."
                    )
                    coding_res = run_coding(
                        client,
                        gen_params=gen_params,
                        is_test=False,
                        model_name=model_filename,
                        task_limit=task_limit_val,
                        lcb_task_limit=lcb_limit_val,
                        bigcode_task_limit=bigcode_limit_val,
                        max_tokens=max_tokens,
                    )
                    res.coding_val = coding_res.val_score
                    res.coding_tps = coding_res.avg_tps
                    # val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode
                    res.lcb_val = getattr(coding_res, "val_pass1", 0.0)
                    res.he_val = getattr(coding_res, "val_pass2", 0.0)
                    res.mbpp_val = getattr(coding_res, "val_pass3", 0.0)
                    res.bigcode_val = getattr(coding_res, "val_pass4", 0.0)
                    res.swe_val = 0.0  # legacy slot, unused
                    if res.coding_val <= 0.0:
                        res.status = "FAIL: Coding preflight failed"
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res

                # Agentic (Claw-Eval quick/full tier)
                if agentic_quick or agentic_full:
                    for tier, task_ids in agentic_tiers:
                        n_tasks = len(task_ids)
                        print(
                            f"  [agentic:{tier}] {n_tasks} tasks selected (rule-based scoring, no LLM judge)"
                        )
                        if n_tasks == 0:
                            raise FileNotFoundError(f"No Claw-Eval {tier} tasks found")
                        agentic_res = run_agentic_eval(
                            client,
                            task_ids,
                            gen_params=gen_params,
                            trials=1,
                            turn_timeout=turn_timeout,
                            stop=stop_val,
                        )
                        res.task_ids = tuple(task_ids)
                        if tier == "full" or not agentic_full:
                            res.agentic_val = agentic_res["score"]
                            res.agentic_task_count = agentic_res["total"]
                            res.agentic_tier = tier
                    # Per-trial agentic sidecar (rotated; JSON is not *.log so the
                    # dir is gitignored separately). Uses the last tier's result —
                    # matches res.agentic_val when full runs after quick.
                    try:
                        LOG_DIR.mkdir(parents=True, exist_ok=True)
                        _prune_glob(LOG_DIR, "agentic-*.json", LOG_KEEP - 1)
                        sidecar = {
                            "model": model_filename,
                            "tier": res.agentic_tier,
                            "score": agentic_res.get("score", 0.0),
                            "passed": agentic_res.get("passed", 0),
                            "total": agentic_res.get("total", 0),
                            "elapsed_sec": agentic_res.get("elapsed_sec", 0.0),
                            "max_tokens": gen_params.max_tokens,
                            "server_log": str(server_log),
                            "tasks": agentic_res.get("task_results", []),
                        }
                        sidecar_path = (
                            LOG_DIR
                            / f"agentic-{time.strftime('%Y%m%d-%H%M%S')}-{Path(model_filename).stem}.json"
                        )
                        with open(sidecar_path, "w", encoding="utf-8") as fh:
                            json.dump(sidecar, fh, indent=2)
                        print(f"  [agentic] wrote {sidecar_path}")
                    except OSError as e:
                        print(f"  [WARNING] could not write agentic sidecar: {e}")
                if agentic_coding:
                    print("  [agentic-coding] SWE-lite issue loop (ADR 0013)")
                    ac = run_agentic_coding_eval(client, gen_params=gen_params)
                    res.agentic_coding_val = float(ac["score"])
                    res.agentic_coding_detail = str(ac.get("detail") or "")
                    print(
                        f"  [agentic-coding] {ac['passed']}/{ac['total']} "
                        f"score={ac['score']:.4f} {res.agentic_coding_detail}"
                    )
                if mini_swe_agent:
                    print("  [mini-swe-agent] DM-Code-Agent scoreboard via mini-swe-agent")
                    base_url = f"http://{runner.intent.host}:{runner.port}/v1"
                    model_filename = intent.model_path.name
                    msa_task_limit = int(norm.get("mini_swe_agent_task_limit", 0) or 0)
                    msa_suite = str(norm.get("mini_swe_agent_suite", "all") or "all")
                    if msa_suite not in ("coding", "maintenance", "all"):
                        print(
                            f"  [mini-swe-agent] unknown suite {msa_suite!r}, falling back to 'all'"
                        )
                        msa_suite = "all"
                    try:
                        msa_task_ids = None
                        if msa_task_limit > 0:
                            from autoresearch.benchmarks.mini_swe_agent import (
                                discover_tasks as _msa_discover,
                            )

                            msa_task_ids = _msa_discover(msa_suite)[:msa_task_limit]
                            print(
                                f"  [mini-swe-agent] limiting to first "
                                f"{len(msa_task_ids)} of the suite"
                            )
                        ms = run_mini_swe_agent_eval(
                            base_url=base_url,
                            api_key="sk-no-auth-required",
                            model_name=model_filename,
                            model_filename=model_filename,
                            task_ids=msa_task_ids,
                            suite=msa_suite,
                            gen_params=gen_params,
                        )
                        res.mini_swe_agent_val = float(ms["score"])
                        suite_sig = str(ms.get("suite_signature") or "")
                        upstream_commit = str(ms.get("upstream_commit") or "")
                        detail = str(ms.get("detail") or "")
                        provenance = []
                        if upstream_commit:
                            provenance.append(f"commit={upstream_commit}")
                        if suite_sig:
                            provenance.append(f"suite={suite_sig}")
                        res.mini_swe_agent_detail = (
                            "; ".join(provenance + [detail]) if provenance else detail
                        )
                        print(
                            f"  [mini-swe-agent] {ms['passed']}/{ms['total']} "
                            f"score={ms['score']:.4f} {res.mini_swe_agent_detail}"
                        )
                        print(f"  [mini-swe-agent] trajectory: {ms.get('trajectory_path', '')}")
                    except FileNotFoundError as exc:
                        # Missing uvx or fixtures — treat as `incomplete`,
                        # never as `rejected`; the column stays None so
                        # downstream Pareto logic does not see a 0.0 score.
                        print(f"  [mini-swe-agent] SKIP: {exc}")
                        res.mini_swe_agent_val = None
                        res.mini_swe_agent_detail = f"skipped: {exc}"
                    except Exception as exc:
                        # Hard infra failure — record as incomplete (None),
                        # never as a measured 0.0, so downstream Pareto
                        # logic does not mistake a crashed harness for
                        # "ran N tasks, passed 0".
                        print(f"  [mini-swe-agent] FAIL: {exc}")
                        res.mini_swe_agent_val = None
                        res.mini_swe_agent_detail = f"error: {str(exc)[:200]}"
                # Compute combined metrics
                tps_list = []
                if include_coding and res.coding_tps > 0:
                    tps_list.append(res.coding_tps)

                if tps_list:
                    avg_tps = sum(tps_list) / len(tps_list)
                    res.avg_tps = avg_tps
                    res.tps_source = "coding-generation"
                    if avg_tps < bench_tts_threshold:
                        print(
                            f"  [WARNING] Combined TPS {avg_tps:.2f} is below "
                            f"{bench_tts_threshold:.1f}! Score set to 0.0."
                        )
                        res.val_score = 0.0
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res
                elif not skip_bench:
                    res.avg_tps = res.bench_tg_tps
                    res.tps_source = "backend-bench"
                    if res.avg_tps < bench_tts_threshold:
                        res.val_score = 0.0
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res
                else:
                    # Bench skipped and no coding TPS — do not invent a zero floor reject.
                    res.avg_tps = 0.0
                    res.tps_source = "skipped"

                # Agentic takes priority as quality gate; falls back to coding; falls back to perplexity-only
                if res.agentic_tier and res.agentic_task_count > 0:
                    res.val_score = res.agentic_val
                elif include_coding:
                    res.val_score = res.coding_val
                elif include_perplexity:
                    res.val_score = res.avg_tps
                else:
                    res.val_score = 0.0

        except Exception as e:
            print(f"  [FAIL] Evaluation failed: {e}")
            res.status = f"FAIL: {str(e)[:50]}"
            res.outcome = (
                TrialOutcome.INFRA_ERROR
                if isinstance(e, (FileNotFoundError, OSError))
                else TrialOutcome.CODE_ERROR
            )
            res.diagnostic = str(e)
            res.val_score = 0.0
        finally:
            if runner is not None:
                res.peak_vram_gb = max(runner.peak_vram_mb, 0.0) / 1024.0
                if getattr(runner, "vram_killed", False) is True:
                    _apply_watchdog_kill(res, runner)

        res.elapsed_sec = time.time() - trial_start
        return res
