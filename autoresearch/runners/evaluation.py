"""Experiment runner — owns the full trial lifecycle.

Deep module extracted from run.py. One interface (run_trial), typed TrialResult,
hides bench validation, server lifecycle, and metric computation behind the seam.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent, resolve_llama_bench
from autoresearch.core.llama_client import LlamaClient
from autoresearch.benchmarks.benchmark_coding import run_benchmark as run_coding

BASE_DIR = Path(__file__).resolve().parent

# ── llama-bench defaults ────────────────────────────────────────────────
BENCH_TPS_THRESHOLD = 20.0  # min tg t/s from llama-bench
BENCH_N_PROMPT = 512
BENCH_N_GEN = 128


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
    avg_tps: float = 0.0
    peak_vram_gb: float = 0.0
    bench_tg_tps: float = 0.0
    bench_pp_tps: float = 0.0


def run_llama_bench_validation(
    model_path: Path,
    ngl: int = 99,
    threads: int = 8,
    batch_size: int = 512,
    ubatch_size: int = 128,
    flash_attn: str = "on",
    cache_type_k: str = "q4_0",
    cache_type_v: str = "q4_0",
    n_prompt: int = BENCH_N_PROMPT,
    n_gen: int = BENCH_N_GEN,
) -> float:
    """Run llama-bench with given config. Returns tg t/s. Raises on failure."""
    llama_bench = resolve_llama_bench()

    cmd = [
        str(llama_bench),
        "-m", str(model_path),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-ngl", str(ngl),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-fa", flash_attn,
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "-o", "json",
        "-r", "3",
    ]

    print(f"  [bench] {' '.join(str(a) for a in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    data = json.loads(result.stdout)
    tg_tps = 0.0
    for entry in data:
        n_g = entry.get("n_gen", 0)
        n_p = entry.get("n_prompt", 0)
        if n_g > 0 and n_p == 0:
            tg_tps = entry.get("avg_ts", 0.0)
            break

    if tg_tps == 0.0:
        raise RuntimeError(f"llama-bench returned no tg result: {result.stdout[:300]}")

    return tg_tps


class ExperimentRunner:
    """Deep module: owns the full trial lifecycle.

    run_trial() is the sole external seam. Callers get a typed TrialResult
    without managing:
    - llama-bench validation
    - llama-server lifecycle (start, health check, teardown)
    - benchmark orchestration
    - VRAM tracking
    - metric computation

    Leverage: one interface, N callers (CLI, autoloop, grid sweep).
    Locality: trial orchestration logic and bugs concentrate in one module.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

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
        intent, norm = ServerIntent.from_config(cfg_dict, self.models_dir, **overrides)
        model_filename = intent.model_path.name

        max_tokens = norm.get("max_tokens", 1024)
        include_coding = norm.get("include_coding", True)
        task_limit_val = norm.get("coding_task_limit", 10)
        lcb_limit_val = norm.get("lcb_task_limit", 10)
        bigcode_limit_val = norm.get("bigcode_task_limit", 10)
        trial_budget = norm.get("trial_budget")
        bench_tts_threshold = norm.get("bench_tts_threshold", BENCH_TPS_THRESHOLD)
        is_validation = norm.get("validation", False)

        res = TrialResult()

        # ── Pre-check: llama-bench validation ────────────────────────────
        if not skip_bench:
            try:
                bench_tg = run_llama_bench_validation(
                    model_path=intent.model_path,
                    ngl=intent.ngl,
                    threads=intent.threads,
                    batch_size=intent.batch_size,
                    ubatch_size=intent.ubatch_size,
                    flash_attn=intent.flash_attn,
                    cache_type_k=intent.kv_cache_k or intent.kv_cache,
                    cache_type_v=intent.kv_cache_v or intent.kv_cache,
                    n_prompt=BENCH_N_PROMPT,
                    n_gen=BENCH_N_GEN,
                )
            except FileNotFoundError as e:
                print(f"  [FAIL] llama-bench not found: {e}")
                res.status = "FAIL: llama-bench not found"
                return res
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] llama-bench crashed: {e}")
                res.status = "FAIL: llama-bench crashed"
                return res
            except Exception as e:
                print(f"  [FAIL] llama-bench error: {e}")
                res.status = f"FAIL: llama-bench error: {str(e)[:50]}"
                return res

            res.bench_tg_tps = bench_tg
            print(f"  [bench] tg {BENCH_N_GEN}: {bench_tg:.1f} t/s")

            if bench_tg < bench_tts_threshold:
                print(f"  [FAIL] llama-bench tg {bench_tg:.1f} t/s below threshold {bench_tts_threshold:.1f}")
                res.status = f"FAIL: bench tg {bench_tg:.1f} < threshold {bench_tts_threshold:.1f}"
                return res

            if is_validation:
                print(f"  [OK] Bench validation passed: tg {bench_tg:.1f} t/s >= {bench_tts_threshold:.1f}")
                # Coerce to quick 2-task coding validation
                task_limit_val = 2
                lcb_limit_val = 2
                bigcode_limit_val = 2
                # fall through to coding eval

        # ── Full evaluation ──────────────────────────────────────────────
        server_log = BASE_DIR / "llama_server.log"
        if server_log.exists():
            try:
                server_log.unlink()
            except OSError:
                pass

        gen_kwargs = {
            "temp": norm.get("temp", 0.2),
            "top_p": norm.get("top_p"),
            "min_p": norm.get("min_p"),
            "top_k": norm.get("top_k"),
            "repeat_penalty": norm.get("repeat_penalty"),
            "presence_penalty": norm.get("presence_penalty"),
            "frequency_penalty": norm.get("frequency_penalty"),
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        trial_start = time.time()
        timeout_at = trial_start + trial_budget if trial_budget else None

        try:
            with LlamaServerRunner(intent, log_path=server_log) as runner:
                client = LlamaClient(runner.port)

                # Coding (HumanEval + MBPP + LCB + BigCode)
                if include_coding:
                    print(
                        f"  [coding] Running (limit={task_limit_val}, "
                        f"lcb_limit={lcb_limit_val}, bigcode_limit={bigcode_limit_val})..."
                    )
                    coding_res = run_coding(
                        client,
                        is_test=False,
                        model_name=model_filename,
                        task_limit=task_limit_val,
                        lcb_task_limit=lcb_limit_val,
                        bigcode_task_limit=bigcode_limit_val,
                        timeout_at=timeout_at,
                        max_tokens=max_tokens,
                        **gen_kwargs,
                    )
                    res.coding_val = coding_res.val_score
                    res.coding_tps = coding_res.avg_tps
                    # val_pass1 = LCB, val_pass2 = HE, val_pass3 = MBPP, val_pass4 = BigCode
                    res.lcb_val = getattr(coding_res, "val_pass1", 0.0)
                    res.he_val = getattr(coding_res, "val_pass2", 0.0)
                    res.mbpp_val = getattr(coding_res, "val_pass3", 0.0)
                    res.bigcode_val = getattr(coding_res, "val_pass4", 0.0)
                    res.swe_val = 0.0  # legacy slot, unused

                # Compute combined metrics
                tps_list = []
                if include_coding and res.coding_tps > 0:
                    tps_list.append(res.coding_tps)

                if tps_list:
                    avg_tps = sum(tps_list) / len(tps_list)
                    res.avg_tps = avg_tps
                    if avg_tps < 20.0:
                        print(f"  [WARNING] Combined TPS {avg_tps:.2f} is below 20.0! Score set to 0.0.")
                        res.val_score = 0.0
                        return res
                else:
                    res.avg_tps = 0.0

                res.peak_vram_gb = max(runner.peak_vram_mb, 0.0) / 1024.0
                res.val_score = res.coding_val

        except Exception as e:
            print(f"  [FAIL] Evaluation failed: {e}")
            res.status = f"FAIL: {str(e)[:50]}"
            res.val_score = 0.0

        return res
