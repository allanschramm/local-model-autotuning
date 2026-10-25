"""Bounded TPS x configured-context Search, independent of quality benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from autoresearch.core import config, fingerprint
from autoresearch.core.hardware import detect_gpu_temp_c, wait_gpu_near_idle
from autoresearch.core.llama_client import GenerationParams
from autoresearch.core.llama_runner import (
    ConfigError,
    FlagDriftError,
    LlamaServerRunner,
    NativeModelProcess,
    ServerIntent,
    assert_flags_supported,
    engine_version_tag,
    load_mode_flag,
    preflight_host_memory_for_intent,
    preflight_vram_for_intent,
    resolve_llama_perplexity,
)
from autoresearch.core.model_arch import default_models_dir
from autoresearch.core.search import SearchStrategy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "tuning-runtime-v1"
PPL_CONTEXT = 2048
GENERATION_TOKENS = 512
PROMPT = "Continue this sequence with many more integers: 1, 2, 3, 4, 5, 6, 7, 8,"


def measure_perplexity(intent: ServerIntent, *, log_path: Path) -> float:
    """Fixed-text PPL, with the same model-process protection as llama-server."""
    text_file = ROOT / "data/perplexity_val.txt"
    if not text_file.is_file():
        raise FileNotFoundError(f"Missing fixed PPL text: {text_file}")
    ppl_intent = replace(
        intent,
        ctx_size=PPL_CONTEXT,
        parallel=1,
        spec_type=None,
        spec_draft_n_max=0,
        spec_draft_model=None,
    )
    binary = resolve_llama_perplexity()
    cmd = [
        str(binary),
        "-m",
        str(intent.model_path),
        "-f",
        str(text_file),
        "-c",
        str(PPL_CONTEXT),
        "--chunks",
        "1",
        "-ngl",
        str(intent.ngl),
        "-t",
        str(intent.threads),
        "-b",
        str(intent.batch_size),
        "-ub",
        str(intent.ubatch_size),
        "-fa",
        intent.flash_attn,
        "-ctk",
        intent.kv_cache_k or intent.kv_cache,
        "-ctv",
        intent.kv_cache_v or intent.kv_cache,
    ]
    cmd += load_mode_flag(intent.no_mmap, intent.mlock)
    cmd += ["--no-warmup"]
    if intent.n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(intent.n_cpu_moe)]
    if intent.threads_batch is not None:
        cmd += ["--threads-batch", str(intent.threads_batch)]
    if intent.numa is not None:
        cmd += ["--numa", intent.numa]
    assert_flags_supported(cmd, binary)
    with NativeModelProcess(ppl_intent, cmd, log_path) as child:
        code = child.wait()
        output = child.output()
        if code:
            raise RuntimeError(f"perplexity exited {code}; see {log_path}")
    final = re.search(r"Final estimate:\s*PPL\s*=\s*([^\s,]+)", output)
    chunks = re.findall(r"\[\d+\]\s*([^\s,]+)", output)
    if not final and not chunks:
        raise ValueError(f"Missing PPL estimate; see {log_path}")
    value = float(final.group(1) if final else chunks[-1])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid PPL measurement")
    return value


class TuningRunner:
    """One configuration: guarded PPL, then guarded full-context generation."""

    def __init__(
        self,
        models_dir: Path,
        *,
        logs_dir: Path | None = None,
        server_factory=LlamaServerRunner,
        perplexity=measure_perplexity,
        thermal_wait: bool = True,
    ):
        self.models_dir = Path(models_dir).resolve()
        self.logs_dir = logs_dir or ROOT / "tuning-artifacts"
        self.server_factory = server_factory
        self.perplexity = perplexity
        self.thermal_wait = thermal_wait
        self.idle = detect_gpu_temp_c() if thermal_wait else None

    def run_trial(self, cfg: dict[str, Any], *, minimum_context: int) -> dict[str, Any]:
        started = time.monotonic()
        result = {
            "protocol": PROTOCOL,
            "outcome": "INFRA_ERROR",
            "tps": None,
            "ppl": None,
            "ctx": cfg.get("CTX_SIZE"),
            "observed_context": None,
            "diagnostic": "",
            "samples": [],
            "config": cfg,
        }
        server = None
        try:
            intent, norm = ServerIntent.from_config(cfg, self.models_dir)
            if intent.ctx_size < minimum_context or intent.parallel != 1:
                raise ConfigError("Tuning requires CTX_SIZE >= minimum_context and PARALLEL=1")
            if intent.model_path.is_dir():
                raise ConfigError("This tuning protocol requires a GGUF/llama.cpp runtime")
            reps = int(cfg.get("TPS_REPS", 3))
            if reps < 1:
                raise ConfigError("TPS_REPS must be positive")
            ok, _, reason = preflight_vram_for_intent(intent, norm.get("vram_limit_mb"))
            if not ok:
                result.update(outcome="MODEL_REJECTED", diagnostic=reason)
                return result
            ok, _, _, reason = preflight_host_memory_for_intent(intent)
            if not ok:
                result.update(outcome="MODEL_REJECTED", diagnostic=reason)
                return result
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            tag = uuid.uuid4().hex
            wait_gpu_near_idle(idle_c=self.idle, enabled=self.thermal_wait)
            result["ppl"] = self.perplexity(intent, log_path=self.logs_dir / f"{tag}-ppl.log")
            if not math.isfinite(result["ppl"]) or result["ppl"] <= 0:
                raise ValueError("Missing or invalid PPL measurement")
            result["ppl_context"] = PPL_CONTEXT
            result["ppl_text_sha256"] = hashlib.sha256(
                (ROOT / "data/perplexity_val.txt").read_bytes()
            ).hexdigest()
            with self.server_factory(
                intent,
                log_path=self.logs_dir / f"{tag}-server.log",
                vram_limit_mb=norm.get("vram_limit_mb"),
            ) as server:
                if getattr(server, "llama_server", None):
                    result["engine_build"] = engine_version_tag(server.llama_server)
                base = f"http://{intent.host}:{server.port}"

                def request(path, payload=None):
                    data = json.dumps(payload).encode() if payload is not None else None
                    req = urllib.request.Request(
                        base + path, data=data, headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=420) as response:
                        return json.loads(response.read())

                props = request("/props")
                observed = props.get("default_generation_settings", {}).get("n_ctx")
                if observed != intent.ctx_size or props.get("total_slots") != 1:
                    raise ValueError("Server reported a different per-slot context or slot count")
                result["observed_context"] = observed
                generation = GenerationParams(
                    temp=norm.get("temp", 0.1),
                    top_p=norm.get("top_p"),
                    top_k=norm.get("top_k"),
                    min_p=norm.get("min_p"),
                    repeat_penalty=norm.get("repeat_penalty"),
                    presence_penalty=norm.get("presence_penalty"),
                    frequency_penalty=norm.get("frequency_penalty"),
                ).to_payload()
                generation.pop("max_tokens")
                for _ in range(reps):
                    wait_gpu_near_idle(idle_c=self.idle, enabled=self.thermal_wait)
                    sample = request(
                        "/completion",
                        {
                            **generation,
                            "prompt": PROMPT,
                            "n_predict": GENERATION_TOKENS,
                            "ignore_eos": True,
                            "seed": 1234,
                            "cache_prompt": False,
                            "stream": False,
                        },
                    )
                    rate = sample.get("timings", {}).get("predicted_per_second")
                    if (
                        not isinstance(rate, (int, float))
                        or not math.isfinite(rate)
                        or rate <= 0
                        or sample.get("timings", {}).get("predicted_n") != GENERATION_TOKENS
                        or sample.get("generation_settings", {}).get("n_ctx") != intent.ctx_size
                        or sample.get("truncated") is not False
                    ):
                        raise ValueError(
                            "Incomplete generation, invalid timing, or context mismatch"
                        )
                    result["samples"].append(sample)
                result["tps"] = statistics.median(
                    s["timings"]["predicted_per_second"] for s in result["samples"]
                )
                result["outcome"] = "OK"
        except ConfigError as exc:
            result.update(outcome="INVALID_CONFIG", diagnostic=str(exc), tps=None)
        except (OSError, RuntimeError, ValueError, FlagDriftError) as exc:
            outcome = (
                "WATCHDOG_KILL"
                if "WATCHDOG_KILL" in str(exc) or "RAM_CIRCUIT_BREAKER" in str(exc)
                else "INFRA_ERROR"
            )
            result.update(outcome=outcome, diagnostic=str(exc), tps=None)
        finally:
            if server is not None and (server.vram_killed or server.ram_killed):
                result.update(
                    outcome="WATCHDOG_KILL",
                    diagnostic=server.vram_kill_reason or "RAM_CIRCUIT_BREAKER",
                    tps=None,
                )
            result["elapsed_sec"] = time.monotonic() - started
        return result


def run_search(
    baseline: dict[str, Any], plan: dict[str, Any], runner: TuningRunner, database: Path
) -> dict[str, Any]:
    """Finite Neighbor Search; every admitted point uses one fixed PPL reference."""
    minimum = plan.get("minimum_context")
    maximum = plan.get("max_trials")
    space = plan.get("search_space")
    if (
        type(minimum) is not int
        or minimum < 1
        or type(maximum) is not int
        or maximum < 1
        or not isinstance(space, dict)
        or not space
    ):
        raise ValueError("Plan requires positive minimum_context/max_trials and a search_space")
    floor = max(minimum, int(baseline["CTX_SIZE"]))
    for name, values in space.items():
        if (
            name not in config.ENGINE_DEFAULTS
            or name not in fingerprint.SERVER_ENGINE_KEYS
            or not isinstance(values, list)
            or not values
        ):
            raise ValueError(f"Invalid engine Search Space: {name}")
        if name == "CTX_SIZE" and any(type(v) is not int or v < floor for v in values):
            raise ValueError("Search Space cannot reduce the Baseline or minimum context")
    strategy = SearchStrategy(space)
    pending = deque([dict(baseline)])
    visited = set()
    front = []
    reference = None
    run_id = uuid.uuid4().hex
    directory = Path(database).resolve().parent / "tuning-artifacts" / run_id
    directory.mkdir(parents=True)
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS tuning_runs (run_id TEXT PRIMARY KEY, protocol TEXT NOT NULL, plan_json TEXT NOT NULL, status TEXT NOT NULL, reference_ppl REAL, frontier_json TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS tuning_trials (trial_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, config_json TEXT NOT NULL, ctx INTEGER, tps REAL, ppl REAL, outcome TEXT NOT NULL, accepted INTEGER NOT NULL, artifact TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO tuning_runs VALUES (?,?,?,?,?,?)",
            (run_id, PROTOCOL, json.dumps(plan), "running", None, None),
        )
        db.commit()
        while pending and len(visited) < maximum:
            cfg = pending.popleft()
            key = strategy.get_config_key(cfg)
            if key in visited:
                continue
            visited.add(key)
            trial = runner.run_trial(cfg, minimum_context=floor)
            trial_id = uuid.uuid4().hex
            if reference is None and trial["outcome"] == "OK":
                reference = trial["ppl"]
            eligible = (
                trial["outcome"] == "OK"
                and reference is not None
                and trial["ppl"] <= reference * 1.01
            )
            entry = {
                "trial_id": trial_id,
                "config": cfg,
                "ctx": cfg["CTX_SIZE"],
                "tps": trial["tps"],
                "ppl": trial["ppl"],
            }

            def dominates(a, b):
                return (
                    a["ctx"] >= b["ctx"]
                    and a["tps"] >= b["tps"]
                    and (a["ctx"] > b["ctx"] or a["tps"] > b["tps"])
                )

            joins = eligible and not any(
                dominates(p, entry) or (p["ctx"], p["tps"]) == (entry["ctx"], entry["tps"])
                for p in front
            )
            if joins:
                front = [p for p in front if not dominates(entry, p)] + [entry]
                pending.extend(
                    n.config
                    for n in sorted(
                        strategy.get_neighbors(cfg), key=lambda n: (n.changed, repr(n.new))
                    )
                )
            trial["ppl_reference"] = reference
            trial["quality_admitted"] = eligible
            artifact = directory / f"{trial_id}.json"
            artifact.write_text(json.dumps(trial, indent=2), encoding="utf-8")
            db.execute(
                "INSERT INTO tuning_trials VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trial_id,
                    run_id,
                    json.dumps(cfg),
                    cfg["CTX_SIZE"],
                    trial["tps"],
                    trial["ppl"],
                    trial["outcome"],
                    int(eligible),
                    str(artifact),
                ),
            )
            db.execute(
                "UPDATE tuning_runs SET reference_ppl=?,frontier_json=? WHERE run_id=?",
                (reference, json.dumps(front), run_id),
            )
            db.commit()
            if trial["outcome"] in ("INFRA_ERROR", "INVALID_CONFIG") or reference is None:
                raise RuntimeError(
                    f"Tuning stopped: {trial['diagnostic'] or trial['outcome']}; evidence: {artifact}"
                )
        for entry in front:
            selected = entry["config"]
            target = directory / f"fingerprint-{entry['trial_id']}.json"
            fingerprint.dump(
                target,
                model=selected["MODEL"],
                engine={k: v for k, v in selected.items() if k in config.ENGINE_DEFAULTS},
                sampler={k: v for k, v in selected.items() if k in config.SAMPLER_DEFAULTS},
            )
            entry["fingerprint"] = str(target)
        db.execute(
            "UPDATE tuning_runs SET status='complete',frontier_json=? WHERE run_id=?",
            (json.dumps(front), run_id),
        )
    return {
        "run_id": run_id,
        "reference_ppl": reference,
        "frontier": sorted(front, key=lambda p: p["ctx"]),
        "trials": len(visited),
        "termination": "trial_budget"
        if any(strategy.get_config_key(c) not in visited for c in pending)
        else "neighbors_exhausted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, required=True, help="JSON Search Space and finite Trial budget"
    )
    parser.add_argument("--db", type=Path, default=ROOT / "results.db")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    runner = TuningRunner(default_models_dir())
    print(json.dumps(run_search(config.load_config(), plan, runner, args.db), indent=2))


if __name__ == "__main__":
    main()
