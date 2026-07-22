"""Experiment runner — owns the full trial lifecycle.

Deep module extracted from run.py. One interface (run_trial), typed TrialResult,
hides bench validation, server lifecycle, and metric computation behind the seam.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from autoresearch.core.llama_runner import (
    LlamaServerRunner,
    ServerIntent,
    resolve_llama_bench,
    resolve_llama_cli,
    resolve_llama_perplexity,
    ConfigError,
    estimate_vram_mb,
    preflight_vram_for_intent,
    resolve_vram_limit_mb,
)
from autoresearch.core.sglang_runner import SGLangServerRunner, run_sglang_bench_validation
from autoresearch.core.llama_client import LlamaClient, GenerationParams
from autoresearch.benchmarks.benchmark_coding import run_benchmark as run_coding
from autoresearch.benchmarks.agentic_benchmarks import get_quick_tier_tasks, get_full_tier_tasks
from autoresearch.benchmarks.agentic_runner import run_agentic_eval

BASE_DIR = Path(__file__).resolve().parent

# ── llama-bench defaults ────────────────────────────────────────────────
BENCH_TPS_THRESHOLD = 20.0  # min tg t/s from llama-bench
BENCH_N_PROMPT = 512
BENCH_N_GEN = 512


class TrialOutcome(str, Enum):
    OK = "OK"
    INVALID_CONFIG = "INVALID_CONFIG"
    MODEL_REJECTED = "MODEL_REJECTED"
    INFRA_ERROR = "INFRA_ERROR"
    CODE_ERROR = "CODE_ERROR"


class AgenticBenchmarkAdapter(Protocol):
    def task_ids(self, tier: str) -> list[str]: ...
    def run(self, client: LlamaClient, task_ids: list[str], gen_params: GenerationParams) -> dict: ...


class ClawEvalAdapter:
    """Docker-free Claw-Eval adapter with deterministic local grading."""

    def task_ids(self, tier: str) -> list[str]:
        return get_quick_tier_tasks() if tier == "quick" else get_full_tier_tasks()

    def run(self, client: LlamaClient, task_ids: list[str], gen_params: GenerationParams) -> dict:
        return run_agentic_eval(client, task_ids, gen_params=gen_params, trials=1)


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
    agentic_val: float = 0.0    # Claw-Eval quick/full tier score
    agentic_tier: str = ""       # "quick" or "full"
    agentic_task_count: int = 0  # number of tasks evaluated
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
    cont_batching: bool = False,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
    spec_draft_model: str | None = None,
    n_cpu_moe: int | None = None,
    n_gen: int = BENCH_N_GEN,
) -> float:
    """Run llama-cli with given config. Returns tg t/s. Raises on failure."""
    llama_cli = resolve_llama_cli()

    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", "Write a comprehensive, step-by-step tutorial explaining quantum computing, qubits, superposition, and entanglement, including a detailed Python simulation using NumPy.",
        "-n", str(n_gen),
        "-c", str(ctx_size),
        "-t", str(threads),
        "-ngl", str(ngl),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-fa", flash_attn,
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "--no-mmap" if no_mmap else "--mmap",
        "--no-warmup",
        "--simple-io",
        "--single-turn",
    ]

    if threads_batch is not None:
        cmd += ["-tbd", str(threads_batch)]
    if n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]

    spec_type_val = spec_type
    if spec_type_val is None and "MTP" in model_path.name.upper() and spec_draft_n_max > 0:
        spec_type_val = "draft-mtp"

    if spec_type_val is not None and spec_type_val.lower() != "none" and spec_draft_n_max > 0:
        cmd += [
            "--spec-type", spec_type_val.lower(),
            "--spec-draft-n-max", str(spec_draft_n_max),
            "--spec-draft-type-k", cache_type_k,
            "--spec-draft-type-v", cache_type_v,
            "-ngld", str(ngl),
        ]
        if spec_draft_model:
            draft_path = Path(spec_draft_model)
            if not draft_path.is_absolute():
                draft_path = model_path.parent / draft_path
            cmd += ["--spec-draft-model", str(draft_path)]

    print(f"  [cli-bench] {' '.join(str(a) for a in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    import re
    match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", result.stdout)
    if not match:
        match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", result.stderr)

    if not match:
        raise RuntimeError(f"llama-cli output did not contain Generation TPS metric: {result.stdout[:500]} {result.stderr[:500]}")

    tg_tps = float(match.group(1))
    return tg_tps


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
        "-m", str(model_path),
        "-f", str(text_file),
        "-t", str(threads),
        "-ngl", str(ngl),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-fa", flash_attn,
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "-c", str(ctx_size),
        "--chunks", str(chunks),
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
        raise RuntimeError(f"Could not parse perplexity from output. stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}")
    
    return float(match.group(1))


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

    def __init__(self, models_dir: Path, agentic_adapter: AgenticBenchmarkAdapter | None = None):
        self.models_dir = Path(models_dir)
        self.agentic_adapter = agentic_adapter or ClawEvalAdapter()

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
        vram_limit_mb = resolve_vram_limit_mb(norm.get("vram_limit_mb"))

        ok_vram, est_vram, vram_reason = preflight_vram_for_intent(intent, vram_limit_mb)
        print(f"  [vram-preflight] est={est_vram:.0f}MB limit={vram_limit_mb:.0f}MB ok={ok_vram}")
        if not ok_vram:
            res.status = f"FAIL: {vram_reason}"
            res.outcome = TrialOutcome.MODEL_REJECTED
            res.diagnostic = vram_reason
            res.peak_vram_gb = est_vram / 1024.0
            return res

        max_tokens = norm.get("max_tokens", 1024)
        include_coding = bool(norm.get("include_coding", False))
        task_limit_val = norm.get("coding_task_limit", 10)
        lcb_limit_val = norm.get("lcb_task_limit", 10)
        bigcode_limit_val = norm.get("bigcode_task_limit", 10)
        bench_tts_threshold = norm.get("bench_tts_threshold", BENCH_TPS_THRESHOLD)
        is_validation = norm.get("validation", False)
        # Accept both CLI keys (agentic_*) and bench_config keys (include_agentic_*).
        agentic_quick = bool(
            norm.get("agentic_quick", False) or norm.get("include_agentic_quick", False)
        )
        agentic_full = bool(
            norm.get("agentic_full", False) or norm.get("include_agentic_full", False)
        )
        if is_validation:
            agentic_quick = True
            agentic_full = False
        if include_coding and (task_limit_val, lcb_limit_val, bigcode_limit_val) != (10, 10, 10):
            res.status = "FAIL: Coding preflight requires exactly 10 tasks per dataset"
            res.outcome = TrialOutcome.INVALID_CONFIG
            res.diagnostic = res.status[6:]
            return res
        agentic_tiers: list[tuple[str, list[str]]] = []
        if agentic_quick:
            agentic_tiers.append(("quick", self.agentic_adapter.task_ids("quick")))
        if agentic_full:
            agentic_tiers.append(("full", self.agentic_adapter.task_ids("full")))
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
                    bench_tg = run_sglang_bench_validation(
                        model_path=intent.model_path,
                        batch_size=1,  # We use 1 here for bench
                        n_prompt=BENCH_N_PROMPT,
                        n_gen=BENCH_N_GEN,
                    )
                    res.bench_tg_tps = bench_tg
                    print(f"  [bench] sglang.bench_one_batch tg {BENCH_N_GEN}: {bench_tg:.1f} t/s")

                    if bench_tg < bench_tts_threshold:
                        print(f"  [FAIL] sglang bench tg {bench_tg:.1f} t/s below threshold {bench_tts_threshold:.1f}")
                        res.status = f"FAIL: sglang bench tg {bench_tg:.1f} < threshold {bench_tts_threshold:.1f}"
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res

                    if is_validation:
                        print(f"  [OK] SGLang bench validation passed: tg {bench_tg:.1f} t/s >= {bench_tts_threshold:.1f}")
                except Exception as e:
                    print(f"  [FAIL] sglang bench error: {e}")
                    res.status = f"FAIL: sglang bench error: {str(e)[:50]}"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res
            else:
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
                        ctx_size=intent.ctx_size,
                        threads_batch=intent.threads_batch,
                        no_mmap=intent.no_mmap,
                        cont_batching=intent.cont_batching,
                        spec_type=intent.spec_type,
                        spec_draft_n_max=intent.spec_draft_n_max,
                        spec_draft_model=intent.spec_draft_model,
                        n_cpu_moe=intent.n_cpu_moe,
                        n_gen=BENCH_N_GEN,
                    )
                except FileNotFoundError as e:
                    print(f"  [FAIL] llama-cli not found: {e}")
                    res.status = "FAIL: llama-cli not found"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res
                except subprocess.CalledProcessError as e:
                    print(f"  [FAIL] llama-cli crashed: {e}")
                    res.status = "FAIL: llama-cli crashed"
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    return res
                except Exception as e:
                    print(f"  [FAIL] llama-cli error: {e}")
                    res.status = f"FAIL: llama-cli error: {str(e)[:50]}"
                    res.outcome = TrialOutcome.INFRA_ERROR
                    res.diagnostic = str(e)
                    return res

                res.bench_tg_tps = bench_tg
                print(f"  [bench] tg {BENCH_N_GEN}: {bench_tg:.1f} t/s")

                if bench_tg < bench_tts_threshold:
                    print(f"  [FAIL] llama-cli tg {bench_tg:.1f} t/s below threshold {bench_tts_threshold:.1f}")
                    res.status = f"FAIL: bench tg {bench_tg:.1f} < threshold {bench_tts_threshold:.1f}"
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    return res

                if is_validation:
                    print(f"  [OK] Bench validation passed: tg {bench_tg:.1f} t/s >= {bench_tts_threshold:.1f}")
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
        server_log = BASE_DIR / "llama_server.log"
        if server_log.exists():
            try:
                server_log.unlink()
            except OSError:
                pass

        gen_params = GenerationParams(
            temp=norm.get("temp", 0.2),
            top_p=norm.get("top_p"),
            min_p=norm.get("min_p"),
            top_k=norm.get("top_k"),
            repeat_penalty=norm.get("repeat_penalty"),
            presence_penalty=norm.get("presence_penalty"),
            frequency_penalty=norm.get("frequency_penalty"),
        )

        trial_start = time.time()
        if include_perplexity and not include_coding and not agentic_quick and not agentic_full:
            res.avg_tps = res.bench_tg_tps
            res.tps_source = "backend-bench"
            if res.avg_tps < 20.0:
                res.val_score = 0.0
                res.outcome = TrialOutcome.MODEL_REJECTED
                return res
            if include_perplexity:
                res.val_score = res.avg_tps
            else:
                res.val_score = 0.0
            
            # Estimate peak VRAM from config since server wasn't started
            k_val = intent.kv_cache_k or intent.kv_cache
            v_val = intent.kv_cache_v or intent.kv_cache
            res.peak_vram_gb = estimate_vram_mb(
                intent.model_path,
                intent.ctx_size,
                k_val,
                v_val,
                draft_path=intent.spec_draft_model,
            ) / 1024.0
            res.elapsed_sec = time.time() - trial_start
            return res

        runner = None
        try:
            runner_cls = SGLangServerRunner if intent.model_path.is_dir() else LlamaServerRunner
            runner_kwargs: dict[str, Any] = {"log_path": server_log}
            if runner_cls is LlamaServerRunner:
                runner_kwargs["vram_limit_mb"] = vram_limit_mb
            with runner_cls(intent, **runner_kwargs) as runner:
                if getattr(runner, "vram_killed", False) is True:
                    res.status = "FAIL: VRAM_LIMIT_EXCEEDED"
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    res.diagnostic = "VRAM_LIMIT_EXCEEDED"
                    res.peak_vram_gb = max(getattr(runner, "peak_vram_mb", 0.0), 0.0) / 1024.0
                    return res
                client = LlamaClient(runner.port)

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
                        print(f"  [agentic:{tier}] {n_tasks} tasks selected (rule-based scoring, no LLM judge)")
                        if n_tasks == 0:
                            raise FileNotFoundError(f"No Claw-Eval {tier} tasks found")
                        agentic_res = self.agentic_adapter.run(client, task_ids, gen_params)
                        res.task_ids = tuple(task_ids)
                        if tier == "full" or not agentic_full:
                            res.agentic_val = agentic_res["score"]
                            res.agentic_task_count = agentic_res["total"]
                            res.agentic_tier = tier
                # Compute combined metrics
                tps_list = []
                if include_coding and res.coding_tps > 0:
                    tps_list.append(res.coding_tps)

                if tps_list:
                    avg_tps = sum(tps_list) / len(tps_list)
                    res.avg_tps = avg_tps
                    res.tps_source = "coding-generation"
                    if avg_tps < 20.0:
                        print(f"  [WARNING] Combined TPS {avg_tps:.2f} is below 20.0! Score set to 0.0.")
                        res.val_score = 0.0
                        res.outcome = TrialOutcome.MODEL_REJECTED
                        return res
                elif not skip_bench:
                    res.avg_tps = res.bench_tg_tps
                    res.tps_source = "backend-bench"
                    if res.avg_tps < 20.0:
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
            res.outcome = TrialOutcome.INFRA_ERROR if isinstance(e, (FileNotFoundError, OSError)) else TrialOutcome.CODE_ERROR
            res.diagnostic = str(e)
            res.val_score = 0.0
        finally:
            if runner is not None:
                res.peak_vram_gb = max(runner.peak_vram_mb, 0.0) / 1024.0
                if getattr(runner, "vram_killed", False) is True and res.outcome == TrialOutcome.OK:
                    res.status = "FAIL: VRAM_LIMIT_EXCEEDED"
                    res.outcome = TrialOutcome.MODEL_REJECTED
                    res.diagnostic = "VRAM_LIMIT_EXCEEDED"
                    res.val_score = 0.0

        res.elapsed_sec = time.time() - trial_start
        return res
