#!/usr/bin/env python3
"""
Autonomous Hill-Climbing Evaluation Loop.

Reads Baseline from autoresearch/core/config.py — or starts from a Day/Night
Usage Profile pick off the results-store Pareto front (canonical results.db, legacy TSV fallback) (issue #8) — then runs
active benchmarks → perturbs one flag → if the Neighbor joins or improves the
per-model Pareto Set, writes Baseline back to config.py → loops forever. The
legacy scalar keep rule survives only for incomplete vectors (engine-only /
quality-only modes never measure agentic+coding, so they cannot compete on
the four-axis front; ADR 0006).

Stop with Ctrl+C (SIGINT). Visited memory persists in .autoresearch_state.json;
Baseline persists in config.py; results in the results store (results.db canonical + legacy results.tsv fallback).
"""

import json
import random
import re
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoresearch.core import classify
from autoresearch.core.classify import MORRIS_SCREEN_PROFILE
from autoresearch.core.config import (
    ENGINE_DEFAULTS,
    SAMPLER_DEFAULTS,
)
from autoresearch.core.crash_journal import clear_journal, read_journal, write_journal
from autoresearch.core.hardware import detect_hardware_capabilities
from autoresearch.core.llama_runner import (
    estimate_vram_mb,
    preflight_host_memory,
    resolve_model_path,
    resolve_spec_estimate_args,
)
from autoresearch.core.model_arch import resolve_n_cpu_moe
from autoresearch.core.morris import (
    elementary_effects,
    generate_trajectories,
    pins_from_effects,
    varying_space,
)
from autoresearch.core.search import SearchStrategy
from autoresearch.core.state import SearchState
from autoresearch.runners.evaluation import (
    BENCH_N_GEN,
    ExperimentRunner,
    TrialOutcome,
    median_llama_bench_validation,
)
from autoresearch.runners.run import (
    MODELS_DIR,
    RESULTS_FILE,
    get_git_commit,
    read_rows,
    recompute_statuses,
    tsv_fields_from_cfg,
    write_row,
)

BASE_DIR = Path(__file__).resolve().parent

# ── Search space: param_name → list of candidate values ──────────────────
SEARCH_SPACE = {
    "KV_CACHE_K": ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "KV_CACHE_V": ["q4_0", "q8_0", "turbo2", "turbo3", "turbo4", "f16"],
    "THREADS": [6, 8, 12, 16],
    "THREADS_BATCH": [None, 8, 12, 16, 24],
    "BATCH_SIZE": [256, 512, 1024],
    "UBATCH_SIZE": [64, 128, 256, 512],
    "SPEC_DRAFT_N_MAX": [0, 1, 2, 3, 4],
    "SPEC_TYPE": [None, "ngram-cache"],
    "CONT_BATCHING": [False, True],
    "FLASH_ATTN": ["on"],
    "NO_MMAP": [False, True],
    "TEMP": [0.0, 0.2, 0.4, 0.6, 0.7, 1.0],
    "TOP_P": [None, 0.8, 0.9, 0.95],
    "TOP_K": [None, 20, 40, 64],
    "MIN_P": [None, 0.0, 0.02, 0.05],
    "PRESENCE_PENALTY": [None, 0.0, 1.5],
    "REPEAT_PENALTY": [None, 1.0, 1.1],
    "NUMA": [None, "distribute", "isolate"],
}

# Params not in search space but needed for config persistence
# Core params (in autoresearch.core.config)
CORE_PASSTHROUGH = [
    "KV_CACHE",
    "MODEL",
    "CTX_SIZE",
    "JINJA",
    "REASONING_BUDGET",
    "REASONING_BUDGET_MESSAGE",
    "REASONING",
    "REASONING_PRESERVE",
    "SPEC_DRAFT_MODEL",
    "FREQUENCY_PENALTY",
    "N_CPU_MOE",
    "N_GPU_LAYERS",
    "NUMA",
]
# Bench params (in autoresearch.benchmarks.bench_config)
BENCH_PASSTHROUGH = [
    "INCLUDE_CODING",
    "CODING_TASK_LIMIT",
    "INCLUDE_AGENTIC_QUICK",
    "INCLUDE_AGENTIC_FULL",
    "INCLUDE_AGENTIC_CODING",
]
PASSTHROUGH_PARAMS = CORE_PASSTHROUGH + BENCH_PASSTHROUGH

# GPU-only speculative knobs (no effect with N_GPU_LAYERS==0); dropped from the
# active Search Space on CPU-only hosts (issue #19).
CPU_EXCLUDED_SEARCH_KEYS = {"SPEC_DRAFT_N_MAX"}


def apply_cpu_preflight(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """CPU preflight seed (issue #19).

    Baseline N_GPU_LAYERS Auto (-1) + no detected GPU -> return cfg with
    N_GPU_LAYERS=0 (CPU-only first-class Baseline). A GPU present, or an
    explicit N_GPU_LAYERS, -> None (leave Baseline untouched). Pure: callers
    persist via StateManager.update_baseline (which writes through
    config.write_baseline).
    """
    if cfg.get("N_GPU_LAYERS", -1) != -1:
        return None
    if detect_hardware_capabilities().get("has_gpu"):
        return None
    return {**cfg, "N_GPU_LAYERS": 0}


def filter_search_space_for_cpu(
    search_space: dict[str, list[Any]], n_gpu_layers: int
) -> dict[str, list[Any]]:
    """Drop GPU-only speculative knobs when the active Baseline is CPU-only.

    N_GPU_LAYERS==0 excludes CPU_EXCLUDED_SEARCH_KEYS; Auto (-1) or GPU (N>0)
    keeps the full space. Returns a copy, never mutating the input.
    """
    if n_gpu_layers == 0:
        return {k: v for k, v in search_space.items() if k not in CPU_EXCLUDED_SEARCH_KEYS}
    return dict(search_space)


def apply_pins(search_space: dict[str, list[Any]], pins: dict[str, Any]) -> dict[str, list[Any]]:
    """Collapse pinned keys to a single legal level."""
    out = {k: list(v) for k, v in search_space.items()}
    for key, val in pins.items():
        if key in out and val in out[key]:
            out[key] = [val]
    return out


def consume_crash_journal(state, retry: bool = False, results_file: Path | None = None) -> None:
    """At process start: reject the in-flight Fingerprint or drop the journal."""
    pending = read_journal()
    if not pending:
        return
    if retry:
        clear_journal()
        return
    target = results_file if results_file is not None else RESULTS_FILE
    write_row(
        target,
        get_git_commit(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "rejected",
        "crash journal: started but process died",
        outcome="CRASH",
        diagnostic="crash journal: started but process died",
        model=str(pending.get("model") or ""),
        config_json=str(pending.get("config_json") or ""),
        tps=0.0,
        bench_tg=0.0,
    )
    key = pending.get("config_key")
    if key:
        state.mark_visited(str(key))
    clear_journal()
    recompute_statuses(target)


def _llama_bench_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg.get("MODEL", "")
    model_path = resolve_model_path(MODELS_DIR, model)
    n_cpu_moe, _ = resolve_n_cpu_moe(model_path, cfg.get("N_CPU_MOE"))
    return {
        "model_path": model_path,
        "ngl": int(
            cfg.get("N_GPU_LAYERS", 99) if cfg.get("N_GPU_LAYERS", -1) not in (-1, None) else 99
        ),
        "threads": int(cfg.get("THREADS", 8) or 8),
        "batch_size": int(cfg.get("BATCH_SIZE", 512) or 512),
        "ubatch_size": int(cfg.get("UBATCH_SIZE", 128) or 128),
        "flash_attn": str(cfg.get("FLASH_ATTN", "on") or "on"),
        "cache_type_k": str(cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE") or "q4_0"),
        "cache_type_v": str(cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE") or "q4_0"),
        "ctx_size": int(cfg.get("CTX_SIZE", 131072) or 131072),
        "threads_batch": cfg.get("THREADS_BATCH"),
        "no_mmap": bool(cfg.get("NO_MMAP", False)),
        "mlock": bool(cfg.get("MLOCK", False)),
        "cont_batching": bool(cfg.get("CONT_BATCHING", False)),
        "spec_type": cfg.get("SPEC_TYPE"),
        "spec_draft_n_max": int(cfg.get("SPEC_DRAFT_N_MAX", 0) or 0),
        "spec_draft_model": cfg.get("SPEC_DRAFT_MODEL"),
        "n_cpu_moe": n_cpu_moe,
        "n_gen": BENCH_N_GEN,
        "vram_limit_mb": cfg.get("VRAM_LIMIT_MB"),
    }


def run_morris_screen(
    model_name: str,
    seed_cfg: dict[str, Any],
    engine_space: dict[str, list[Any]],
    runner: ExperimentRunner,
    state_manager: SearchState,
    trajectories: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Measure elementary effects on engine knobs; persist pins. y = llama-cli TPS."""
    space = {k: v for k, v in engine_space.items() if k in ENGINE_DEFAULTS}
    space = varying_space(space)
    trajs = generate_trajectories(space, trajectories, rng, seed_cfg)
    samples: list[tuple[str, float, float]] = []
    measured: list[tuple[dict[str, Any], float]] = []
    vram_limit = seed_cfg.get("VRAM_LIMIT_MB")
    for traj in trajs:
        prev_y = 0.0
        for cfg, param in traj:
            point = {**seed_cfg, **cfg, "MODEL": model_name}
            if not preflight_vram_ok(point, vram_limit):
                y = 0.0
            else:
                try:
                    y, _, _ = median_llama_bench_validation(
                        **_llama_bench_kwargs(point),
                        reps=1,
                        idle_c=getattr(runner, "_idle_gpu_c", None),
                        thermal_wait=getattr(runner, "thermal_wait", True),
                    )
                except Exception:
                    y = 0.0
            measured.append((point, y))
            if param:
                samples.append((param, prev_y, y))
            prev_y = y
            write_row(
                RESULTS_FILE,
                get_git_commit(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "incomplete",
                f"Morris screen {param}",
                category=MORRIS_SCREEN_PROFILE,
                evaluation_profile=MORRIS_SCREEN_PROFILE,
                tps=y,
                bench_tg=y,
                model=model_name,
                config_json=json.dumps(point, sort_keys=True, default=repr),
            )
            recompute_statuses(RESULTS_FILE)
    effects = elementary_effects(samples)
    best_cfg = measured[0][0] if measured else seed_cfg
    best_y = measured[0][1] if measured else 0.0
    for cfg, y in measured:
        if y > best_y:
            best_cfg, best_y = cfg, y
    pins = pins_from_effects(effects, space, best_cfg)
    state_manager.set_morris(model_name, pins, effects)
    return pins


def _journaled_run_trial(runner, cfg, model_name, search_strategy):
    payload = {
        "model": model_name,
        "config_key": search_strategy.get_config_key(cfg),
        "config_json": json.dumps(cfg, sort_keys=True, default=repr),
        "started_at": datetime.now(UTC).isoformat(),
    }
    write_journal(payload)
    try:
        return runner.run_trial(cfg)
    finally:
        clear_journal()


# ── Graceful shutdown ────────────────────────────────────────────────────
_stop_requested = False


def _signal_handler(_sig, _frame):
    global _stop_requested
    _stop_requested = True
    print("\n[AUTOLOOP] Graceful stop requested. Finishing current evaluation...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_config(baseline_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load immutable defaults overlaid by the local Baseline state."""
    from autoresearch.benchmarks import bench_config as _bc

    if baseline_cfg is None:
        from autoresearch.core.config import load_config as _core_load_config

        baseline_cfg = _core_load_config()
    result = dict(baseline_cfg)
    # Merge bench params
    bench_vals = {p: getattr(_bc, p, None) for p in BENCH_PASSTHROUGH}
    result.update({k: v for k, v in bench_vals.items() if v is not None})
    return result


# ── Alias sync (issue #40) ──────────────────────────────────────────────
# Trailing GGUF quant tag (Q4_K_M, IQ4_XS, Q8_0, BF16, ...). Stripped from the
# family slug so a quant change overwrites the same alias (one per family).
_QUANT_TAG_RE = re.compile(
    r"[-_](?:[IQT]?[QK]?\d(?:_[A-Za-z0-9]+)+|BF16|FP\d+|F16|F32|E4M3|E5M2)$",
    re.IGNORECASE,
)


def _family_slug(model_name: str) -> str:
    """Kebab-case family slug from a GGUF basename (quant tag stripped).

    'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf' -> 'qwen3.6-35b-a3b-ud'
    """
    stem = model_name[:-5] if model_name.lower().endswith(".gguf") else model_name
    stem = _QUANT_TAG_RE.sub("", stem)
    slug = re.sub(r"[^a-z0-9.]+", "-", stem.lower()).strip("-")
    return slug or "model"


def _match_alias_dir(aliases_dir: Path, model_name: str) -> Path | None:
    """Existing alias dir whose name is a lowercase prefix of the model basename."""
    if not aliases_dir.exists():
        return None
    model_lower = model_name.lower()
    for d in aliases_dir.iterdir():
        if d.is_dir() and model_lower.startswith(d.name.lower()):
            return d
    return None


def _compile_alias_flags(model_name: str, new_cfg: dict, existing_flags: list) -> list[str]:
    """llama-server flags for the alias recipe from the record Trial Baseline."""
    flags: list[str] = []
    existing_ngl = next(
        (
            flag
            for flag in existing_flags
            if isinstance(flag, str) and flag.startswith(("--n-gpu-layers ", "-ngl "))
        ),
        None,
    )
    if new_cfg.get("JINJA"):
        flags.append("--jinja")
    if new_cfg.get("CTX_SIZE"):
        flags.append(f"--ctx-size {new_cfg['CTX_SIZE']}")
    configured_ngl = new_cfg.get("N_GPU_LAYERS")
    if configured_ngl is not None:
        flags.append(f"--n-gpu-layers {int(configured_ngl)}")
    elif existing_ngl:
        flags.append(existing_ngl)

    model_path = resolve_model_path(MODELS_DIR, model_name)
    n_cpu_moe, _ = resolve_n_cpu_moe(model_path, new_cfg.get("N_CPU_MOE"))
    if n_cpu_moe is not None:
        flags.append(f"--n-cpu-moe {n_cpu_moe}")

    k_val = new_cfg.get("KV_CACHE_K") or new_cfg.get("KV_CACHE")
    if k_val:
        flags.append(f"--cache-type-k {k_val}")

    v_val = new_cfg.get("KV_CACHE_V") or new_cfg.get("KV_CACHE")
    if v_val:
        flags.append(f"--cache-type-v {v_val}")

    if new_cfg.get("FLASH_ATTN"):
        flags.append(f"--flash-attn {new_cfg['FLASH_ATTN']}")
    if new_cfg.get("THREADS"):
        flags.append(f"--threads {new_cfg['THREADS']}")
    if new_cfg.get("THREADS_BATCH"):
        flags.append(f"--threads-batch {new_cfg['THREADS_BATCH']}")
    if new_cfg.get("BATCH_SIZE"):
        flags.append(f"--batch-size {new_cfg['BATCH_SIZE']}")
    if new_cfg.get("UBATCH_SIZE"):
        flags.append(f"--ubatch-size {new_cfg['UBATCH_SIZE']}")
    if new_cfg.get("CONT_BATCHING"):
        flags.append("--cont-batching")
    if new_cfg.get("NO_MMAP"):
        flags.append("--no-mmap")

    spec_type = new_cfg.get("SPEC_TYPE")
    if spec_type and spec_type != "none":
        flags.append(f"--spec-type {spec_type}")
        if new_cfg.get("SPEC_DRAFT_N_MAX", 0) > 0:
            flags.append(f"--spec-draft-n-max {new_cfg['SPEC_DRAFT_N_MAX']}")
        if new_cfg.get("SPEC_DRAFT_MODEL"):
            flags.append(f"--spec-draft-model models/{new_cfg['SPEC_DRAFT_MODEL']}")

    for p in ["TEMP", "TOP_P", "TOP_K", "MIN_P", "REPEAT_PENALTY", "PRESENCE_PENALTY"]:
        val = new_cfg.get(p)
        if val is not None:
            flags.append(f"--{p.lower().replace('_', '-')} {val}")
    return flags


def _touch_metrics(data: dict, tps: float, marker: str) -> None:
    """Stamp metrics on an alias config; `marker` is the autoloop notes tag."""
    import datetime

    if "metrics" not in data or not isinstance(data["metrics"], dict):
        data["metrics"] = {}
    data["metrics"]["tps"] = float(tps)
    data["metrics"]["measured_at"] = datetime.date.today().strftime("%Y-%m-%d")
    data["metrics"]["measured_by"] = "autoloop"
    notes = data["metrics"].get("notes", "")
    if notes and marker not in notes:
        data["metrics"]["notes"] = f"{notes} {marker}"
    elif not notes:
        data["metrics"]["notes"] = marker


def update_model_alias(model_name: str, new_cfg: dict, tps: float, mode: str) -> None:
    """Sync the family alias under models/aliases/ with the record Trial (issue #40).

    Existing alias (lowercase-prefix family match) -> update flags + preferred
    quant (model:) + metrics. Missing -> create the kebab-case family alias.
    Quant changes overwrite the same alias: one preferred quant per family.
    Never writes outside models/; config stays machine-local (gitignored).
    """
    import yaml

    aliases_dir = Path(__file__).resolve().parent / "models" / "aliases"
    try:
        target_dir = _match_alias_dir(aliases_dir, model_name)
        created = target_dir is None
        if target_dir is None:
            target_dir = aliases_dir / _family_slug(model_name)
            target_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = target_dir / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {
                "alias": target_dir.name,
                "port": 18080,
                "host": "127.0.0.1",
                "description": f"Alias for {model_name} (auto-created by autoloop).",
                "status": "ready",
            }

        # Quant change -> preferred quant; flags/metrics follow the record Trial.
        data["model"] = f"models/{model_name}"
        data["flags"] = _compile_alias_flags(model_name, new_cfg, data.get("flags", []))
        _touch_metrics(
            data, tps, "Auto-created by autoloop" if created else "(Auto-updated by autoloop)"
        )

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

        verb = "created" if created else "updated"
        print(f"  [ALIAS] Auto-{verb} alias config at {yaml_path}")
    except Exception as e:
        print(f"  [WARNING] Failed to auto-update alias config: {e}")


def write_climb_fingerprint(
    model_name: str,
    cfg: dict[str, Any],
    *,
    outcome: Any = None,
    status: str = "on_front",
    is_tps_climb: bool = True,
    directory: Path | str | None = None,
) -> Path | None:
    """Write the ADR 0014 Fingerprint file after a kept TPS+PPL Neighbor (issue #51).

    Engine only — sampler is user/card choice, never a TPS Neighbor, so the
    file carries no sampler section. Skips (None, existing file untouched) when
    this was not a TPS/PPL climb, the Trial outcome is not OK, or the Trial was
    rejected — a failed climb never overwrites a good file. Write failures warn;
    the Baseline keep stands.
    """
    from autoresearch.core.fingerprint import dump, path_for

    if not is_tps_climb:
        return None
    if outcome is not None and outcome != "OK":
        return None
    if status == "rejected":
        return None
    try:
        engine = {k: v for k, v in cfg.items() if k in ENGINE_DEFAULTS}
        return dump(path_for(model_name, directory), model=model_name, engine=engine)
    except Exception as e:
        print(f"  [WARNING] Failed to write Fingerprint file: {e}")
        return None


def _objective_vector(cfg: dict[str, Any], res) -> classify.ObjectiveVector:
    """Objective Vector of a Trial; blank axis = not measured (ADR 0006)."""
    return classify.ObjectiveVector(
        ctx=cfg.get("CTX_SIZE"),
        tps=getattr(res, "avg_tps", None) or None,
        agentic=getattr(res, "agentic_val", None)
        if getattr(res, "agentic_tier", "") == "full"
        else None,
        coding=getattr(res, "coding_val", None) if bool(cfg.get("INCLUDE_CODING", False)) else None,
    )


def _seed_known_vectors(model_name: str, bucket_gb: int | None) -> list[classify.ObjectiveVector]:
    """Complete known vectors for one model, scoped to a hardware+budget bucket.

    ADR 0017 Known-Set shape via ``classify.known_vectors`` — one filter/merge
    implementation shared with the store's classification, not a mirror:
    same basename, same bucket, rejected rows never compete, Morris screen
    rows are diagnostic-only, and repeated measurements merge per-axis.
    """
    return classify.known_vectors(read_rows(RESULTS_FILE), model=model_name, bucket_gb=bucket_gb)


def _classify(cfg: dict[str, Any], res) -> tuple[str, dict[str, str], classify.ObjectiveVector]:
    """Classify an AutoLoop Trial vs the known Set (issue #4)."""
    vector = _objective_vector(cfg, res)
    status, flips = classify.plan_write(
        read_rows(RESULTS_FILE),
        fp=classify.fp_from_baseline(cfg),
        vector=vector,
        bucket_gb=classify.bucket(getattr(res, "peak_vram_gb", 0.0)),
        failed=getattr(res, "outcome", TrialOutcome.OK) != TrialOutcome.OK,
        model=str(cfg.get("MODEL") or cfg.get("model") or "").strip() or None,
    )
    return status, flips, vector


def _write_trial(
    cfg: dict[str, Any],
    res,
    description: str,
    model_name: str,
    tsv_category: str,
) -> str:
    """Classify, persist, and apply merge flips for one Trial (issue #4).

    Returns the ADR 0006 status. Every non-OK outcome (including
    INFRA_ERROR / CODE_ERROR) lands as `rejected` — no Trial disappears.
    """
    status, _, vector = _classify(cfg, res)
    write_row(
        RESULTS_FILE,
        get_git_commit(),
        res.val_score,
        res.swe_val,
        res.he_val,
        res.mbpp_val,
        res.peak_vram_gb,
        status,
        description,
        lcb_score=res.lcb_val,
        bigcode_score=res.bigcode_val,
        agentic=vector.agentic,
        coding=vector.coding,
        agentic_coding=getattr(res, "agentic_coding_val", None),
        category=tsv_category,
        tps=res.avg_tps,
        bench_tg=getattr(res, "bench_tg_tps", None),
        outcome=getattr(getattr(res, "outcome", TrialOutcome.OK), "value", "OK"),
        diagnostic=getattr(res, "diagnostic", ""),
        evaluation_profile=tsv_category,
        scoring_benchmark="claw-eval",
        task_ids=",".join(getattr(res, "task_ids", ())),
        tps_source=getattr(res, "tps_source", ""),
        gpu_temp_c=getattr(res, "gpu_temp_c", None),
        tps_reps=",".join(f"{x:.4g}" for x in getattr(res, "tps_reps", ()) or ()),
        tps_spread=getattr(res, "tps_spread", None),
        **{
            **tsv_fields_from_cfg(cfg),
            "model": model_name,
            "config_json": json.dumps(cfg, sort_keys=True, default=repr),
        },
    )
    recompute_statuses(RESULTS_FILE)
    return status


def trial_config(
    cfg: dict[str, Any], defaults: dict[str, Any], include_ppl: bool = False
) -> dict[str, Any]:
    """Map bench_config INCLUDE_* flags onto evaluation.py agentic_*/include_coding keys."""
    res_cfg = {**defaults, **cfg}
    if include_ppl:
        for k in [
            "INCLUDE_CODING",
            "INCLUDE_AGENTIC_QUICK",
            "INCLUDE_AGENTIC_FULL",
            "INCLUDE_AGENTIC_CODING",
            "include_coding",
            "agentic_quick",
            "agentic_full",
            "agentic_coding",
            "include_agentic_quick",
            "include_agentic_full",
            "include_agentic_coding",
        ]:
            res_cfg[k] = False
        res_cfg["include_perplexity"] = True
    else:
        res_cfg["include_coding"] = bool(cfg.get("INCLUDE_CODING", False))
        res_cfg["agentic_quick"] = bool(cfg.get("INCLUDE_AGENTIC_QUICK", False))
        res_cfg["agentic_full"] = bool(cfg.get("INCLUDE_AGENTIC_FULL", False))
        res_cfg["agentic_coding"] = bool(cfg.get("INCLUDE_AGENTIC_CODING", False))
        res_cfg["include_perplexity"] = False
    return res_cfg


def preflight_vram_ok(cfg: dict[str, Any], vram_limit: float | None) -> bool:
    """Estimate VRAM + host memory; return True if both gates pass."""
    model = cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")
    ctx = cfg.get("CTX_SIZE", 131072)
    kv_k = cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE", "q4_0")
    kv_v = cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE", "q4_0")
    if not kv_k:
        kv_k = "q4_0"
    if not kv_v:
        kv_v = "q4_0"
    draft = cfg.get("SPEC_DRAFT_MODEL")
    model_path = resolve_model_path(MODELS_DIR, model)
    spec_type, _, draft_path = resolve_spec_estimate_args(
        model_path,
        cfg.get("SPEC_TYPE"),
        int(cfg.get("SPEC_DRAFT_N_MAX", 0) or 0),
        resolve_model_path(MODELS_DIR, draft) if draft else None,
    )

    if vram_limit is not None:
        n_cpu_moe, _ = resolve_n_cpu_moe(model_path, cfg.get("N_CPU_MOE"))
        # Prefer module-level estimate_vram_mb so tests can patch autoloop.estimate_vram_mb.
        est = estimate_vram_mb(
            model_path,
            ctx,
            kv_k,
            kv_v,
            draft_path=draft_path,
            n_cpu_moe=n_cpu_moe,
            spec_type=spec_type,
            spec_draft_n_max=int(cfg.get("SPEC_DRAFT_N_MAX", 0) or 0),
        )
        if est > vram_limit:
            return False
    return preflight_host_ok(cfg)


def preflight_host_ok(cfg: dict[str, Any]) -> bool:
    """Host-memory gate (full GGUF, no MoE shrink). Fail closed on unified if RAM unknown."""
    model = cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")
    ctx = cfg.get("CTX_SIZE", 131072)
    kv_k = cfg.get("KV_CACHE_K") or cfg.get("KV_CACHE", "q4_0")
    kv_v = cfg.get("KV_CACHE_V") or cfg.get("KV_CACHE", "q4_0")
    if not kv_k:
        kv_k = "q4_0"
    if not kv_v:
        kv_v = "q4_0"
    draft = cfg.get("SPEC_DRAFT_MODEL")
    draft_path = resolve_model_path(MODELS_DIR, draft) if draft else None
    model_path = resolve_model_path(MODELS_DIR, model)
    headroom = cfg.get("HOST_MEMORY_HEADROOM_MB", ENGINE_DEFAULTS.get("HOST_MEMORY_HEADROOM_MB"))
    ok, _, _, _ = preflight_host_memory(
        model_path,
        ctx,
        kv_cache_k=kv_k,
        kv_cache_v=kv_v,
        draft_path=draft_path,
        headroom_mb=headroom,
    )
    return ok


def _available_gguf_names(models_dir: Path) -> list[str]:
    """Basenames of main GGUFs under models/ (nested OK). Skips draft/vision/aliases/cache."""
    skip_roots = {".cache", "aliases", "huggingface", "draft", "vision"}
    names: set[str] = set()
    for path in models_dir.rglob("*.gguf"):
        rel = path.relative_to(models_dir)
        if rel.parts and rel.parts[0] in skip_roots:
            continue
        if any(part in {".cache", "aliases", "huggingface"} for part in rel.parts):
            continue
        names.add(path.name)
    return sorted(names)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Hill-Climbing Evaluation Loop")
    parser.add_argument("--max-rounds", type=int, default=0, help="Max rounds (0=infinite)")
    parser.add_argument(
        "--reset-visited",
        action="store_true",
        help="Clear visited memory only (Baseline stays in config.py)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Space-separated list of model filenames to optimize (1 or more)",
    )
    parser.add_argument(
        "--perplexity-val",
        action="store_true",
        help="Enable perplexity validation to act as a quality ceiling constraint while optimizing for TPS",
    )
    parser.add_argument(
        "--mode",
        choices=["tps", "quality", "both"],
        default="both",
        help="Optimization mode: 'tps' (speed), 'quality' (accuracy), 'both' (everything)",
    )
    parser.add_argument("--no-screen", action="store_true", help="Skip Morris engine-knob screen")
    parser.add_argument(
        "--rescreen", action="store_true", help="Ignore stored Morris pins and screen again"
    )
    parser.add_argument(
        "--screen-trajectories",
        type=int,
        default=4,
        help="Morris trajectories R (default 4)",
    )
    parser.add_argument(
        "--no-thermal-wait", action="store_true", help="Skip GPU cooldown before benches"
    )
    parser.add_argument(
        "--tps-reps",
        type=int,
        default=int(ENGINE_DEFAULTS.get("TPS_REPS", 3) or 3),
        help="Median-of-reps for llama-cli / SGLang bench TPS",
    )
    parser.add_argument(
        "--retry-crashed",
        action="store_true",
        help="Do not consume crash journal as a rejected Trial",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke path: print the plan (baseline, neighbors) without running benchmarks",
    )
    cli_args = parser.parse_args()

    max_rounds = cli_args.max_rounds

    state_manager = SearchState()
    consume_crash_journal(state_manager, retry=cli_args.retry_crashed)

    if cli_args.reset_visited:
        state_manager.reset()
        print("[AUTOLOOP] Cleared visited memory. Baseline unchanged in config.py.")

    # 1. Resolve selected models
    available_models = _available_gguf_names(MODELS_DIR)
    if not available_models:
        print("[AUTOLOOP] Error: No GGUF models found in models/ directory!")
        sys.exit(1)

    selected_models = []
    if cli_args.models:
        for m in cli_args.models:
            if m not in available_models:
                matches = [am for am in available_models if m.lower() in am.lower()]
                if matches:
                    selected_models.append(matches[0])
                else:
                    print(f"[AUTOLOOP] Error: Model '{m}' not found in models/.")
                    sys.exit(1)
            else:
                selected_models.append(m)
    elif sys.stdin.isatty():
        print("\nAvailable models in models/:")
        for idx, m in enumerate(available_models, 1):
            print(f"  {idx}) {m}")
        print(
            "\nChoose 1 or more models to run the loop (comma-separated numbers, e.g. 1,3 or 'all'):"
        )
        while True:
            choice = input("Choice: ").strip()
            if not choice:
                continue
            if choice.lower() == "all":
                selected_models = available_models
                break
            try:
                indices = [int(i.strip()) for i in choice.split(",")]
                selected_models = [
                    available_models[i - 1] for i in indices if 1 <= i <= len(available_models)
                ]
                if selected_models:
                    break
            except Exception:
                pass
            print("Invalid choice, try again.")
    else:
        baseline_cfg = load_config(state_manager.get_baseline())
        selected_models = [baseline_cfg.get("MODEL", "g4-opt-it-Q4_K_M.gguf")]

    # ── CPU preflight (issue #19) ──────────────────────────────────────
    # Decide the seed up-front (pure); persist after the dry-run gate below
    # so a --dry-run stays side-effect-free. CPU-only hosts never keep
    # N_GPU_LAYERS Auto (-1) burning Trials.
    preflight_cfg = load_config(state_manager.get_baseline())
    cpu_seed = apply_cpu_preflight(preflight_cfg)

    print("=" * 60)
    print("  AUTONOMOUS HILL-CLIMBING LOOP")
    print(f"  Target models: {', '.join(selected_models)}")
    print("  Trial budget: none (runs to completion)")
    print(
        "  Stop with Ctrl+C. State persists in .autoresearch_state.json + the results store (results.db + legacy results.tsv)"
    )
    print("=" * 60)

    print(f"[AUTOLOOP] Loaded {len(state_manager.visited)} previously visited configs.")

    _defaults = {
        "port": 18080,
        "host": "127.0.0.1",
        "parallel": 1,
        "N_GPU_LAYERS": load_config().get("N_GPU_LAYERS", -1),
        "max_tokens": 1024,
        "context_tokens": 131072,
    }
    ENGINE_KEYS = set(ENGINE_DEFAULTS.keys())
    SAMPLER_KEYS = set(SAMPLER_DEFAULTS.keys())

    active_search_space = dict(SEARCH_SPACE)
    if cli_args.mode == "tps":
        active_search_space = {k: v for k, v in SEARCH_SPACE.items() if k in ENGINE_KEYS}
    elif cli_args.mode == "quality":
        active_search_space = {k: v for k, v in SEARCH_SPACE.items() if k in SAMPLER_KEYS}
    active_search_space = filter_search_space_for_cpu(
        active_search_space, (cpu_seed or preflight_cfg).get("N_GPU_LAYERS", -1)
    )

    search_strategy = SearchStrategy(active_search_space, use_pareto_tiebreaker=True)

    if cli_args.dry_run:
        print("\n[DRY-RUN] No benchmarks will be executed.")
        for model_name in selected_models:
            cfg = load_config(state_manager.get_baseline())
            cfg["MODEL"] = model_name
            print(f"  model: {model_name}")
            print(f"  baseline: {search_strategy.format_config_summary(cfg)}")
            neighbors = search_strategy.get_neighbors(cfg)
            print(f"  neighbor candidates: {len(neighbors)}")
            for n in neighbors[:10]:
                print(f"    - {n.changed}: {n.old} -> {n.new}")
        print("[DRY-RUN] Done.")
        return

    if cpu_seed is not None:
        state_manager.update_baseline(cpu_seed)
        print("[AUTOLOOP] CPU-only host: N_GPU_LAYERS Auto (-1) -> 0 (CPU Baseline seeded).")

    runner = ExperimentRunner(MODELS_DIR, thermal_wait=not cli_args.no_thermal_wait)
    for model_name in selected_models:
        if _stop_requested:
            break

        # Seed the per-model front from the results store, scoped to this
        # hardware+budget bucket (via classify.known_vectors, ADR 0017) so Neighbor
        # acceptance sees prior sessions' points, not an empty session-local front.
        baseline_now = state_manager.get_baseline()
        vram_limit = baseline_now.get("VRAM_LIMIT_MB")
        bucket_gb = classify.bucket(float(vram_limit) / 1024.0) if vram_limit else None
        model_space = dict(active_search_space)
        do_screen = (not cli_args.no_screen) and cli_args.mode != "quality"
        stored_pins = state_manager.morris_pins_for(model_name)
        if do_screen and (cli_args.rescreen or not stored_pins):
            seed_cfg = load_config(state_manager.get_baseline())
            seed_cfg["MODEL"] = model_name
            seed_cfg["TPS_REPS"] = cli_args.tps_reps
            pins = run_morris_screen(
                model_name,
                seed_cfg,
                model_space,
                runner,
                state_manager,
                cli_args.screen_trajectories,
                random.Random(),
            )
            model_space = apply_pins(model_space, pins)
        elif stored_pins:
            model_space = apply_pins(model_space, stored_pins)
        search_strategy = SearchStrategy(
            model_space,
            use_pareto_tiebreaker=True,
            known=_seed_known_vectors(model_name, bucket_gb),
        )

        print(f"\n{'#' * 60}")
        print(f"  OPTIMIZING MODEL: {model_name}")
        print(f"{'#' * 60}")

        # Load config and update MODEL
        cfg = load_config(state_manager.get_baseline())
        cfg["MODEL"] = model_name
        state_manager.update_baseline(cfg)

        round_num = 0
        while not _stop_requested:
            round_num += 1
            if max_rounds > 0 and round_num > max_rounds:
                print(f"\n[AUTOLOOP] Reached max rounds ({max_rounds}) for {model_name}. Stopping.")
                break

            print(f"\n{'=' * 60}")
            print(f"  ROUND {round_num} ({model_name})")
            print(f"{'=' * 60}")

            # ── Step 1: Load current baseline from config.py ─────────────
            baseline_cfg = load_config(state_manager.get_baseline())
            vram_limit = baseline_cfg.get("VRAM_LIMIT_MB")
            baseline_key = search_strategy.get_config_key(baseline_cfg)
            if not state_manager.is_visited(baseline_key):
                state_manager.mark_visited(baseline_key)

            print(f"[BASELINE] {search_strategy.format_config_summary(baseline_cfg)}")

            # ── Step 2: Evaluate baseline ────────────────────────────────
            print("\n[EVAL] Running baseline benchmarks...")
            is_tps_mode = cli_args.mode == "tps"
            baseline_trial_cfg = trial_config(
                baseline_cfg, _defaults, include_ppl=(is_tps_mode or cli_args.perplexity_val)
            )
            baseline_trial_cfg["TPS_REPS"] = cli_args.tps_reps
            baseline_res = _journaled_run_trial(
                runner, baseline_trial_cfg, model_name, search_strategy
            )
            baseline_score = baseline_res.val_score
            baseline_tps = baseline_res.avg_tps
            baseline_vram = baseline_res.peak_vram_gb
            baseline_vector = _objective_vector(baseline_cfg, baseline_res)
            search_strategy.record(baseline_vector)

            if cli_args.mode == "tps":
                tsv_category = "engine-tps"
            elif cli_args.mode == "quality":
                tsv_category = "sampler-quality"
            else:
                tsv_category = "agentic-full"

            baseline_desc = (
                f"AutoLoop R{round_num} baseline for {model_name}: "
                f"{search_strategy.format_config_summary(baseline_cfg)} "
                f"TPS={baseline_tps:.1f} PPL={getattr(baseline_res, 'bench_ppl', 0.0):.4f}"
            )
            baseline_status = _write_trial(
                baseline_cfg, baseline_res, baseline_desc, model_name, tsv_category
            )

            baseline_outcome = getattr(baseline_res, "outcome", TrialOutcome.OK)
            if baseline_outcome in (TrialOutcome.INFRA_ERROR, TrialOutcome.CODE_ERROR):
                raise RuntimeError(f"Search stopped: {baseline_res.status}")

            ppl_str = (
                f" PPL={getattr(baseline_res, 'bench_ppl', 0.0):.4f}"
                if (is_tps_mode or cli_args.perplexity_val)
                else ""
            )
            print(
                f"[BASELINE] Score={baseline_score:.6f} TPS={baseline_tps:.1f}{ppl_str} VRAM={baseline_vram:.1f}GB"
            )

            if baseline_status == "rejected":
                print(f"[BASELINE] Rejected ({baseline_res.status}); attempting Random Restart.")
                new_baseline = search_strategy.random_restart(state_manager.visited, baseline_cfg)
                if new_baseline and preflight_vram_ok(new_baseline, vram_limit):
                    state_manager.update_baseline(new_baseline)
                    continue
                print("[AUTOLOOP] No unvisited VRAM-safe restart available. Stopping.")
                break

            if _stop_requested:
                break

            # ── Step 3: Generate and evaluate neighbors ──────────────────
            neighbors = search_strategy.get_neighbors(baseline_cfg)
            improved = False

            for neighbor in neighbors:
                if _stop_requested:
                    break

                n_key = search_strategy.get_config_key(neighbor.config)
                if state_manager.is_visited(n_key):
                    continue
                state_manager.mark_visited(n_key)

                changed = neighbor.changed
                old_val = neighbor.old
                new_val = neighbor.new

                # Pre-flight VRAM check
                if not preflight_vram_ok(neighbor.config, vram_limit):
                    print(f"  [SKIP] {changed}: {old_val} -> {new_val} (VRAM over budget)")
                    continue

                print(f"\n  [EVAL] Trying {changed}: {old_val} -> {new_val}")
                neighbor_trial_cfg = trial_config(
                    neighbor.config,
                    _defaults,
                    include_ppl=(is_tps_mode or cli_args.perplexity_val),
                )
                neighbor_trial_cfg["TPS_REPS"] = cli_args.tps_reps
                res = _journaled_run_trial(runner, neighbor_trial_cfg, model_name, search_strategy)
                score = res.val_score
                tps = res.avg_tps
                vram = res.peak_vram_gb

                delta = score - baseline_score

                neighbor_vector = _objective_vector(neighbor.config, res)
                search_strategy.record(neighbor_vector)

                # Search truth (issue #7/#8): Neighbor improves iff it joins or
                # improves the per-model Pareto Set. Incomplete vectors never
                # join the front (ADR 0006), so engine-only / quality-only
                # modes keep the legacy scalar rule — they measure no
                # agentic/coding axes to compete on.
                if neighbor_vector.complete:
                    is_improvement = search_strategy.improves_set(neighbor_vector)
                    reason = (
                        "joins/improves per-model Pareto Set"
                        if is_improvement
                        else "dominated by per-model front"
                    )
                else:
                    is_improvement, reason = search_strategy.is_improvement(
                        baseline_score, baseline_tps, baseline_vram, score, tps, vram
                    )

                # Apply Perplexity Quality Ceiling Constraint
                if is_tps_mode or cli_args.perplexity_val:
                    n_ppl = getattr(res, "bench_ppl", 0.0)
                    b_ppl = getattr(baseline_res, "bench_ppl", 0.0)
                    if n_ppl > b_ppl * 1.01:
                        is_improvement = False
                        reason = (
                            f"Perplexity degraded too much (PPL={n_ppl:.4f} vs base={b_ppl:.4f})"
                        )

                neighbor_desc = (
                    f"AutoLoop R{round_num} {changed}={new_val}: "
                    f"{search_strategy.format_config_summary(neighbor.config)} TPS={tps:.1f} "
                    f"PPL={getattr(res, 'bench_ppl', 0.0):.4f} Δ={delta:+.6f}"
                )
                neighbor_status = _write_trial(
                    neighbor.config, res, neighbor_desc, model_name, tsv_category
                )

                if getattr(res, "outcome", TrialOutcome.OK) in (
                    TrialOutcome.INFRA_ERROR,
                    TrialOutcome.CODE_ERROR,
                ):
                    raise RuntimeError(f"Search stopped: {res.status}")

                if is_improvement:
                    print(f"  >>> IMPROVEMENT! {changed}: {old_val} -> {new_val} ({reason})")
                    # Persist new baseline to config.py
                    state_manager.update_baseline(neighbor.config)
                    # Automatically update model alias config
                    update_model_alias(model_name, neighbor.config, tps, cli_args.mode)
                    # TPS+PPL keep writes the Fingerprint file (issue #51, ADR 0014).
                    # TPS mode always measures PPL, so is_tps_mode alone is the
                    # spec climb; sampler keeps (quality/both) never touch the bus.
                    write_climb_fingerprint(
                        model_name,
                        neighbor.config,
                        outcome=getattr(res, "outcome", TrialOutcome.OK),
                        status=neighbor_status,
                        is_tps_climb=is_tps_mode,
                    )
                    improved = True
                    break
                else:
                    print(
                        f"  [DISCARD] {changed}: {old_val} -> {new_val} "
                        f"(Score={score:.6f}, Δ={delta:+.6f})"
                    )

            if not improved and not _stop_requested:
                print(f"\n[AUTOLOOP] Local maxima reached in round {round_num}.")
                print("[AUTOLOOP] Attempting Random Restart...")
                new_baseline = None
                for _ in range(50):
                    candidate = search_strategy.random_restart(state_manager.visited, baseline_cfg)
                    if not candidate:
                        break
                    # Pre-flight VRAM check
                    if preflight_vram_ok(candidate, vram_limit):
                        new_baseline = candidate
                        break
                    else:
                        # Mark as visited in memory so we don't try it again in this round, but do not write to disk
                        state_manager.mark_visited(
                            search_strategy.get_config_key(candidate), persist=False
                        )

                if new_baseline:
                    print(
                        "[AUTOLOOP] Found unvisited VRAM-safe random configuration. Restarting search."
                    )
                    state_manager.update_baseline(new_baseline)
                else:
                    print(
                        "[AUTOLOOP] Exhausted random search space or cannot find VRAM-safe config. Stopping."
                    )
                    break

    # ── Shutdown summary ─────────────────────────────────────────────
    final_cfg = load_config(state_manager.get_baseline())
    print(f"\n{'=' * 60}")
    print("  AUTOLOOP STOPPED")
    print(f"{'=' * 60}")
    print(f"  Final config: {search_strategy.format_config_summary(final_cfg)}")
    print(f"  Results logged to: {RESULTS_FILE} (.db canonical + .tsv fallback)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
