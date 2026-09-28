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
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LLAMA_SERVER_HELP_CACHE = None

import autoresearch.core.config as config
from autoresearch.core import circuit_breaker
from autoresearch.core.config import ConfigError, is_dense_model, validate_config
from autoresearch.core.hardware import (
    detect_free_vram_mb,
    detect_pid_gpu_shared_mb,
    detect_total_vram_mb,
    detect_used_total_vram_mb,
)
from autoresearch.core.memory_budget import MemoryBudget
from autoresearch.core.model_arch import (
    gguf_block_count,
    gguf_expert_bytes_mb,
    gguf_has_mtp,
    gguf_is_moe,
    gguf_kv_f16_mb,
    resolve_model_file,
    resolve_n_cpu_moe,
)
from autoresearch.core.process_guard import ProcessGuard, cleanup_leftover_processes
from autoresearch.core.single_load import enforce_single_load, resolve_allow_multi


def resolve_model_path(models_dir: Path, ref: str | Path) -> Path:
    """Resolve a model ref under models_dir (flat or nested LM Studio layout).

    Order: absolute → models_dir/ref if present → rglob basename (skip junk dirs).
    Missing refs return models_dir/ref for the caller to fail later.
    """
    found = resolve_model_file(ref, models_dir=models_dir, allow_dirs=True)
    if found is not None:
        return found
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return Path(models_dir) / ref_path


ROOT_DIR = Path(__file__).resolve().parent
LLAMA_CPP_ROOT = Path(os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", "./llama.cpp"))
IS_WINDOWS = os.name == "nt"


def _exe(name: str) -> str:
    return f"{name}.exe" if IS_WINDOWS else name


def _parse_bool_env(var_name: str, default: bool = False) -> bool:
    val = os.environ.get(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _is_gpu_working(binary_name: str, probe_flag: str) -> bool:
    tool_path = shutil.which(binary_name)
    if not tool_path:
        return False
    try:
        res = subprocess.run(
            [tool_path, probe_flag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        return res.returncode == 0
    except Exception:
        return False


def should_prefer_gpu_build() -> bool:
    if _parse_bool_env("AUTORESEARCH_PREFER_CPU"):
        return False
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        or os.environ.get("ROCR_VISIBLE_DEVICES") == "-1"
    ):
        return False
    if sys.platform == "darwin":
        return platform.machine() == "arm64"
    if _is_gpu_working("nvidia-smi", "-L") or _is_gpu_working("rocm-smi", "-i"):
        return True
    try:
        from autoresearch.core.hardware import has_discrete_amd, has_discrete_nvidia

        return has_discrete_nvidia() or has_discrete_amd()
    except Exception:
        return False


def _build_dir_candidates(root: Path, build_dir: str, exe: str) -> tuple[Path, ...]:
    base = root / build_dir / "bin"
    return (base / exe, base / "Release" / exe, base / "Debug" / exe)


def _candidate_binary(root: Path, name: str) -> tuple[Path, ...]:
    exe = _exe(name)
    cuda_paths = _build_dir_candidates(root, "build-cuda", exe)
    rocm_paths = _build_dir_candidates(root, "build-rocm", exe)
    cpu_paths = _build_dir_candidates(root, "build-cpu", exe)
    generic_paths = _build_dir_candidates(root, "build", exe)

    if should_prefer_gpu_build():
        try:
            from autoresearch.core.hardware import has_discrete_amd

            if has_discrete_amd():
                return rocm_paths + cuda_paths + generic_paths + cpu_paths
        except Exception:
            pass
        return cuda_paths + rocm_paths + generic_paths + cpu_paths
    return cpu_paths + generic_paths + cuda_paths + rocm_paths


def _binary_candidates(name: str) -> tuple[Path, ...]:
    roots = (
        LLAMA_CPP_ROOT,
        Path.cwd() / "llama.cpp",
        ROOT_DIR.parent.parent / "llama.cpp",
        ROOT_DIR.parent.parent.parent / "llama.cpp",
    )
    seen: list[Path] = []
    for root in roots:
        for candidate in _candidate_binary(root, name):
            if candidate not in seen:
                seen.append(candidate)
    on_path = shutil.which(_exe(name)) or shutil.which(name)
    if on_path:
        seen.append(Path(on_path))
    return tuple(seen)


# ── load-mode passthrough (upstream #26934) ──────────────────────────────
# Upstream folded `--mmap` / `--no-mmap` / `--mlock` into one `--load-mode MODE`
# enum. The harness keeps `NO_MMAP` / `MLOCK` as Baseline knobs (Fingerprint files,
# results.db rows and config.py all key on them) and translates here, at the CLI
# boundary, so an engine bump never touches the store schema.
#
# `NO_MMAP x MLOCK` is a bijection onto the four meaningful modes:
#     (False, False) -> mmap         (False, True) -> mmap+mlock
#     (True,  False) -> none         (True,  True) -> mlock
# No information is lost. The default is emitted explicitly (`mmap`, not `auto`)
# so a Trial command line is reproducible across devices instead of inheriting
# a device-dependent upstream default.
_LOAD_MODES = {
    (False, False): "mmap",
    (False, True): "mmap+mlock",
    (True, False): "none",
    (True, True): "mlock",
}

# Fork-only passthrough: codacus expert-cache knobs are not upstream common
# args, so `--help` never lists them. Exempt by design (opt-in Baseline keys).
FORK_ONLY_FLAGS = frozenset({"--moe-cache-profile", "--moe-cache-slots"})


def load_mode_flag(no_mmap: bool, mlock: bool) -> list[str]:
    """Baseline `NO_MMAP`/`MLOCK` -> the `--load-mode` argv pair.

    Single source for every emitter (server runner, llama-cli bench, model_up,
    serve-config). The legacy `--mmap`/`--no-mmap`/`--mlock` trio is gone from
    b11149 and from the pinned v0.5.0 `common/arg.cpp`.
    """
    return ["--load-mode", _LOAD_MODES[(bool(no_mmap), bool(mlock))]]


class FlagDriftError(RuntimeError):
    """The harness emitted a flag the resolved engine build does not accept.

    Engine bumps rename and remove flags; a stale passthrough otherwise fails
    deep inside a Trial with a cryptic `error: invalid argument` line. Raise at
    build time instead, naming the flag and the build.
    """


# One help line of a llama.cpp `--help` dump is an alias list followed by a
# value placeholder and the description, padded to a column. Aliases appear in
# any order and several per line, and short ones are multi-character:
#   `-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM`
#   `-ncmoe, --n-cpu-moe N                   keep the Mixture of Experts ...`
#   `--spec-draft-model, -md, --model-draft FNAME`
#   `--fim-qwen-1.5b-default                 use default Qwen 2.5 Coder 1.5B`
# The alias list is tokenized from the left and stops at the first description
# word. Only a token directly after a flag counts as a value placeholder (`N`,
# `MODE`, `FNAME`), which is what keeps enum continuation lines
# (`- auto: mmap, ...`) and negative numbers out of the flag set. Long flags
# may contain `.` (upstream ships `--fim-qwen-1.5b-default`), so a narrower
# charset here would report a working flag as drift.
_HELP_FLAG_TOKEN_RE = re.compile(r"^(?:--[\w.-]+|-[A-Za-z][\w-]*)$")
_HELP_PLACEHOLDER_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_HELP_PROMPT_RE = re.compile(r"^[>|\s]*")
_LONG_FLAG_RE = re.compile(r"^--[\w.-]+$")

_HELP_FLAGS_CACHE: dict[str, frozenset[str]] = {}


def _parse_help_flags(help_text: str) -> set[str]:
    """Long flags declared by a llama.cpp `--help` dump."""
    flags: set[str] = set()
    for line in help_text.splitlines():
        tokens = _HELP_PROMPT_RE.sub("", line).split()
        for index, token in enumerate(tokens):
            token = token.rstrip(",")
            if _HELP_FLAG_TOKEN_RE.match(token):
                if token.startswith("--"):
                    flags.add(token)
                continue  # next alias in the list
            if index and _HELP_PLACEHOLDER_RE.match(token) and tokens[index - 1].startswith("-"):
                continue  # value placeholder for the previous flag
            break  # description column reached
    return flags


def supported_flags(binary: Path) -> frozenset[str] | None:
    """Long flags accepted by `binary --help`, cached per binary path.

    Returns None when the probe cannot answer (binary missing, `--help` erroring,
    no flag parsed) so an unprobeable build skips the drift check instead of
    failing a Trial on a guess. The resolvers only ever return existing paths, so
    a None here means the probe was the problem, never a valid build.
    """
    key = str(binary)
    if key in _HELP_FLAGS_CACHE:
        return _HELP_FLAGS_CACHE[key]
    if not Path(binary).is_file():
        return None
    try:
        raw = subprocess.check_output(
            [key, "--help"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
    except Exception:
        return None
    flags = _parse_help_flags(raw)
    if not flags:
        return None
    frozen = frozenset(flags)
    _HELP_FLAGS_CACHE[key] = frozen
    return frozen


def assert_flags_supported(cmd: list[str], binary: Path) -> None:
    """Fail closed when `cmd` carries long flags this engine build dropped.

    Fork-only passthrough (`FORK_ONLY_FLAGS`) is exempt; an unprobeable build
    skips the check. This is the guard that turns the next engine bump from a
    silent Trial failure into an actionable message at the first command build.
    """
    accepted = supported_flags(binary)
    if accepted is None:
        return
    unknown = sorted(
        {str(tok) for tok in cmd[1:] if _LONG_FLAG_RE.match(str(tok)) and str(tok) not in accepted}
        - FORK_ONLY_FLAGS
    )
    if not unknown:
        return
    tag = engine_version_tag(Path(binary))
    where = f"{tag} ({binary})" if tag else str(binary)
    raise FlagDriftError(
        f"flag drift vs {where}: {', '.join(unknown)} not accepted by this build. "
        f"Upstream renamed or removed these. Check the passthrough that built this "
        f"command (load_mode_flag, or the call site that forwarded the flag) against "
        f"llama.cpp/common/arg.cpp add_opt; {binary.name} --help lists what this build "
        f"accepts. Do not migrate Baseline keys for a flag rename."
    )


LLAMA_SERVER_CANDIDATES = _binary_candidates("llama-server")


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
    numa: str | None = None
    kv_cache_k: str | None = None
    kv_cache_v: str | None = None
    threads_batch: int | None = None
    spec_draft_n_max: int = 1
    no_mmap: bool = False
    mlock: bool = False
    jinja: bool = False
    reasoning_budget: int | None = None
    reasoning_budget_message: str | None = None
    reasoning: str | None = None
    reasoning_preserve: bool | None = None
    reasoning_effort: str | None = None
    cont_batching: bool = False
    cache_reuse: int = 256
    host: str = "127.0.0.1"
    spec_type: str | None = None
    spec_draft_model: str | None = None
    n_cpu_moe: int | None = None
    n_cpu_moe_auto: bool = False
    moe_cache_profile: str | None = None
    moe_cache_slots: int | None = None

    @classmethod
    def from_config(cls, cfg: dict, models_dir: Path, **overrides) -> tuple["ServerIntent", dict]:
        """Build ServerIntent from config dict. Caller converts non-dict to dict first.

        Returns (intent, norm_dict) where norm_dict holds all config fields
        (server + non-server) for callers that need remaining params.
        """
        merged = dict(cfg)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        merged = validate_config(merged)
        norm = {
            str(k).lower(): v
            for k, v in merged.items()
            if v is not None and isinstance(k, (str, bytes))
        }

        model_fn = norm.get("model", "g4-opt-it-Q4_K_M.gguf")
        kv_cache = norm.get("kv_cache") or norm.get("kv") or "q4_0"
        k_val = norm.get("kv_cache_k") or norm.get("kv_k") or kv_cache
        v_val = norm.get("kv_cache_v") or norm.get("kv_v") or kv_cache
        draft_ref = (
            norm.get("spec_draft_model")
            or norm.get("draft_model")
            or norm.get("spec_draft_model_file")
        )
        draft_path = str(resolve_model_path(models_dir, draft_ref)) if draft_ref else None
        model_path = resolve_model_path(models_dir, model_fn)
        # Baseline None is dropped from norm; read raw merged for explicit 0 vs auto.
        raw_n_cpu_moe = merged.get("N_CPU_MOE", merged.get("n_cpu_moe"))
        try:
            resolved_n_cpu_moe, n_cpu_moe_auto = resolve_n_cpu_moe(model_path, raw_n_cpu_moe)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if n_cpu_moe_auto:
            print(
                f"  [VITRIOL] auto n-cpu-moe={resolved_n_cpu_moe} "
                f"from GGUF block_count for {model_path.name}"
            )

        intent = cls(
            model_path=model_path,
            ctx_size=norm.get("ctx_size", 131072),
            kv_cache=kv_cache,
            flash_attn=norm.get("flash_attn", "on"),
            port=norm.get("port", 18080),
            host=norm.get("host", "127.0.0.1"),
            ngl=norm.get("n_gpu_layers", norm.get("ngl", 999)),
            numa=norm.get("numa"),
            batch_size=norm.get("batch_size", 512),
            ubatch_size=norm.get("ubatch_size", 128),
            threads=norm.get("threads", 12),
            parallel=norm.get("parallel", 1),
            kv_cache_k=k_val,
            kv_cache_v=v_val,
            threads_batch=norm.get("threads_batch"),
            spec_draft_n_max=norm.get("spec_draft_n_max", 1),
            no_mmap=norm.get("no_mmap", False),
            mlock=norm.get("mlock", False),
            jinja=norm.get("jinja", False),
            reasoning_budget=norm.get("reasoning_budget"),
            reasoning_budget_message=norm.get("reasoning_budget_message"),
            reasoning=norm.get("reasoning"),
            reasoning_preserve=norm.get("reasoning_preserve"),
            reasoning_effort=norm.get("reasoning_effort"),
            cont_batching=norm.get("cont_batching", False),
            cache_reuse=int(norm.get("cache_reuse", 256))
            if norm.get("cache_reuse") is not None
            else 256,
            spec_type=norm.get("spec_type"),
            spec_draft_model=draft_path,
            n_cpu_moe=resolved_n_cpu_moe,
            n_cpu_moe_auto=n_cpu_moe_auto,
            moe_cache_profile=norm.get("moe_cache_profile"),
            moe_cache_slots=norm.get("moe_cache_slots"),
        )

        return intent, norm


# VRAM Estimation Constants (calibrated for f16 KV cache and typical systems)
VRAM_KB_PER_TOKEN_F16 = 80.0
"""Calibrated memory consumption per context token at f16 precision (in kilobytes)."""

VRAM_OVERHEAD_MB = 300.0
"""Typical baseline VRAM overhead for CUDA runtime and system operations (in megabytes)."""

# Residual risk: estimator can under-read peak. Measured dense 65k coding overshoot:
# est 7584 MB → peak 7931 MB (qwen3.5-9b, 2026-07-27). Runtime VRAM sampler is the
# final kill guard; keep the preflight margin conservative for new model arches.

VRAM_SPECULATIVE_BASE_MB = 512.0
VRAM_SPECULATIVE_PER_DRAFT_TOKEN_MB = 256.0
"""Conservative speculative-decoding runtime/workspace allowance."""

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
    # Issue #58: measured ~0.137 on tqp-v0.3.0 dense 27B @64k
    # (7720 MB peak − 6300 weights − 300 overhead) / 8192 f16; was 0.10.
    "turbo2": 0.137,
}
"""KV cache quantization type memory usage scaling factors relative to f16."""

# VITRIOL preflight: MoE expert tensors dominate file size; --n-cpu-moe keeps them on CPU.
VRAM_MOE_NON_EXPERT_FRAC = 0.28
VRAM_MOE_OFFLOAD_LAYER_REF = 32.0

DEFAULT_VRAM_LIMIT_MB = 7900.0
# Safety margin subtracted from free VRAM at Trial start (issue #10).
DEFAULT_VRAM_HEADROOM_MB = 512.0
# Keep this much dedicated VRAM free so WDDM CUDA Sysmem Fallback never arms.
# 256 MiB measured safe on the 8 GB rig (dense @65k peaks 7.4-7.8 GB, no
# Shared spill — autoresearch/AGENTS.md); 512 false-killed loads that fit.
DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB = 256.0
# Task Manager "Shared GPU" kill — WDDM/PCI-e host maps (MoE+NO_MMAP), not
# dedicated-full CUDA Sysmem Fallback. Dedicated can stay ~4–5 GB while Shared
# climbs to 10+ GB and pagefile/SSD freezes the PC. Absolute ceil; do not wait
# for dedicated to fill.
DEFAULT_SHARED_VRAM_LIMIT_MB = 2048.0

# Startup health-poll deadline: a wedged-but-alive llama-server (never returns
# 200 on /health) must not hang a Trial forever. Generous enough for slow GGUF
# loads (HDD / 9p bridges); the loop still returns early on child exit.
SERVER_HEALTH_TIMEOUT_SECONDS = 300.0


def dedicated_vram_kill_ceil(
    limit_mb: float,
    total_mb: float | None,
    keepout_mb: float | None = None,
) -> float:
    """Kill ceiling: ``min(limit, physical − keepout)`` when total is known.

    keepout defaults to env AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB, else
    DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB — single source so the preflight clamp,
    runtime monitor, and cli-bench monitor all agree on the ceiling.
    """
    if keepout_mb is None:
        env_keep = os.environ.get("AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB")
        keepout_mb = float(env_keep) if env_keep else DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB
    ceil = float(limit_mb)
    if total_mb is not None and total_mb > 0:
        return min(ceil, max(0.0, float(total_mb) - keepout_mb))
    return ceil


def resolve_shared_vram_limit_mb(limit: float | int | None = None) -> float:
    """Resolve Shared-GPU kill ceiling (env AUTORESEARCH_SHARED_VRAM_LIMIT_MB)."""
    if limit is not None:
        return float(limit)
    env = os.environ.get("AUTORESEARCH_SHARED_VRAM_LIMIT_MB")
    if env:
        return float(env)
    return float(config.DEFAULTS.get("SHARED_VRAM_LIMIT_MB", DEFAULT_SHARED_VRAM_LIMIT_MB))


DEFAULT_CUDA_FREE_FLOOR_MB = 256.0


def resolve_cuda_free_floor_mb(limit: float | int | None = None) -> float:
    """Resolve CUDA-free kill floor (env AUTORESEARCH_CUDA_FREE_FLOOR_MB).

    When the CUDA-free probe is available, this is the ONLY dedicated-VRAM kill
    condition: kill when CUDA itself reports less free memory than the floor.
    NVML ``used`` counts committed-but-evictable desktop memory on WDDM and is
    kept for peak reporting / fallback only (operator decision 2026-09-04).
    """
    if limit is not None:
        return float(limit)
    env = os.environ.get("AUTORESEARCH_CUDA_FREE_FLOOR_MB")
    if env:
        return float(env)
    return float(config.DEFAULTS.get("CUDA_FREE_FLOOR_MB", DEFAULT_CUDA_FREE_FLOOR_MB))


def resolve_vram_limit_mb(limit: float | int | None = None) -> float:
    """Resolve VRAM budget: explicit arg > env AUTORESEARCH_VRAM_LIMIT_MB > config default.

    Hard clamp: never above ``physical − keepout`` so Trials stay below the
    WDDM CUDA Sysmem Fallback zone (Shared GPU / pagefile thrash).
    """
    if limit is not None:
        configured = float(limit)
    else:
        env = os.environ.get("AUTORESEARCH_VRAM_LIMIT_MB")
        if env:
            configured = float(env)
        else:
            configured = float(config.DEFAULTS.get("VRAM_LIMIT_MB", DEFAULT_VRAM_LIMIT_MB))
    total = detect_total_vram_mb()
    if total is None or total <= 0:
        return configured
    keepout = float(
        config.DEFAULTS.get("PHYSICAL_VRAM_KEEPOUT_MB", DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB)
    )
    env_keep = os.environ.get("AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB")
    if env_keep:
        keepout = float(env_keep)
    safe_ceil = dedicated_vram_kill_ceil(configured, total, keepout)
    # When total known, kill_ceil always applies keepout — clamp only if over.
    if configured > safe_ceil:
        print(
            f"  [vram] clamp VRAM_LIMIT_MB {configured:.0f} -> {safe_ceil:.0f}MB "
            f"(physical {total:.0f} - keepout {keepout:.0f}; stay out of Shared spill zone)",
            flush=True,
        )
        return float(safe_ceil)
    return configured


def resolve_vram_headroom_mb(headroom_mb: float | int | None = None) -> float:
    """Resolve safety margin: explicit arg > env AUTORESEARCH_VRAM_HEADROOM_MB > config > default."""
    if headroom_mb is not None:
        return float(headroom_mb)
    env = os.environ.get("AUTORESEARCH_VRAM_HEADROOM_MB")
    if env:
        return float(env)
    val = config.DEFAULTS.get("VRAM_HEADROOM_MB")
    if val is not None:
        return float(val)
    return float(DEFAULT_VRAM_HEADROOM_MB)


def free_vram_clamp_enabled() -> bool:
    """Dense free-at-start clamp: OFF by default (operator decision 2026-09-04).

    WDDM/desktop reservations made free−headroom false-reject loads that fit
    physical VRAM. AUTORESEARCH_VRAM_FREE_CLAMP=1 restores the old
    min(free−headroom) budget. The runtime VRAM monitor remains the kill guard
    in both modes (device-wide NVML sampling + physical−keepout ceil).
    """
    raw = (os.environ.get("AUTORESEARCH_VRAM_FREE_CLAMP") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def effective_vram_limit_mb(
    configured_mb: float,
    free_vram_mb: float | None = None,
    headroom_mb: float | int | None = None,
) -> float:
    """Effective Trial budget: min(configured, free-at-start - headroom).

    Free VRAM is measured at Trial start; the headroom absorbs measurement noise
    and concurrent-process drift so dirty-GPU Trials fail early instead of
    spuriously. Unknown free -> configured unchanged.
    """
    if free_vram_mb is None or free_vram_mb <= 0:
        return float(configured_mb)
    headroom = resolve_vram_headroom_mb(headroom_mb)
    return min(float(configured_mb), max(0.0, float(free_vram_mb) - headroom))


def is_ngram_spec_type(spec_type: str | None) -> bool:
    """Return True if spec_type specifies ngram-based statistical decoding."""
    if not spec_type:
        return False
    return any(t.strip().startswith("ngram-") for t in str(spec_type).lower().split(","))


def is_pure_ngram_spec_type(spec_type: str | None) -> bool:
    """Return True if spec_type specifies pure ngram decoding without neural drafters."""
    if not is_ngram_spec_type(spec_type):
        return False
    return not any(t.strip().startswith("draft-") for t in str(spec_type).lower().split(","))


def is_spec_enabled(spec_type: str | None, spec_draft_n_max: int = 0) -> bool:
    """Return True if speculative decoding is active via neural drafter or ngram."""
    if not spec_type or str(spec_type).lower() == "none":
        return False
    return int(spec_draft_n_max) > 0 or is_ngram_spec_type(spec_type)


def estimate_vram_mb(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    draft_path: Path | str | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
) -> float:
    try:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
    except Exception:
        model_size_mb = 4000.0

    if n_cpu_moe is not None and int(n_cpu_moe) > 0:
        layer_ref = VRAM_MOE_OFFLOAD_LAYER_REF
        try:
            if model_path.is_file() and gguf_is_moe(model_path):
                layer_ref = float(gguf_block_count(model_path))
        except Exception:
            pass
        offload = min(1.0, float(n_cpu_moe) / layer_ref)
        # GGUF-derived charge (operator decision 2026-09-04): the flat 0.28
        # non-expert fraction over-read 3x on MoVA-class archs (experts = 91%
        # of file). Non-expert = file − expert-class bytes; on-GPU experts for
        # partial offload = total expert bytes − offloaded expert bytes.
        expert_total_mb = None
        expert_off_mb = None
        try:
            expert_total_mb = gguf_expert_bytes_mb(model_path)
            expert_off_mb = gguf_expert_bytes_mb(model_path, layer_limit=int(n_cpu_moe))
        except Exception:
            expert_total_mb = None
        if (
            expert_total_mb is not None
            and expert_off_mb is not None
            and expert_total_mb <= model_size_mb
        ):
            model_size_mb = (model_size_mb - expert_total_mb) + (expert_total_mb - expert_off_mb)
        else:
            expert_frac = 1.0 - VRAM_MOE_NON_EXPERT_FRAC
            model_size_mb = model_size_mb * (
                VRAM_MOE_NON_EXPERT_FRAC + expert_frac * (1.0 - offload)
            )

    draft_mb = 0.0
    if draft_path:
        try:
            draft_mb = Path(draft_path).stat().st_size / (1024 * 1024)
        except Exception:
            draft_mb = 0.0

    try:
        c_size = int(ctx_size)
    except Exception:
        c_size = 16384

    kv_est_mb = _kv_est_mb(
        c_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        base_kv_cache=base_kv_cache,
        model_path=model_path,
    )

    spec_enabled = is_spec_enabled(spec_type, spec_draft_n_max)
    # MoE expert-CPU offload: speculative workspace is already covered by the
    # offload shrink — charging flat 512+256*n false-rejects embedded-MTP
    # (measured 131k n=4 = 4.2GB actual vs 9104 est). Pure ngram speculation
    # has 0 MB VRAM cost (universal statistical lookup). Keep workspace for dense neural spec.
    is_pure_ngram = is_pure_ngram_spec_type(spec_type)
    moe_spec = n_cpu_moe is not None and int(n_cpu_moe) > 0 and spec_enabled
    spec_workspace_mb = (
        0.0
        if (not spec_enabled) or moe_spec or is_pure_ngram
        else VRAM_SPECULATIVE_BASE_MB + VRAM_SPECULATIVE_PER_DRAFT_TOKEN_MB * spec_draft_n_max
    )

    # Baseline system/CUDA overhead (+ draft weights and speculative workspace).
    return model_size_mb + draft_mb + kv_est_mb + VRAM_OVERHEAD_MB + spec_workspace_mb


def preflight_vram(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    vram_limit_mb: float | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
) -> tuple[bool, float, str]:
    """Return (ok, estimate_mb, reason). reason non-empty when rejected."""
    limit = resolve_vram_limit_mb(vram_limit_mb)
    est = estimate_vram_mb(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
        n_cpu_moe=n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
    )
    if est > limit:
        return False, est, f"VRAM_PREFLIGHT est={est:.0f}MB > limit={limit:.0f}MB"
    return True, est, ""


def _build_memory_budget(
    vram_limit_mb: float | int | None = None,
    headroom_mb: float | int | None = None,
    apply_physical_clamp: bool = True,
) -> MemoryBudget:
    """Fold the current env/config/probe state into one MemoryBudget.

    This is the single impure boundary: it is the only place in the decision
    path that reads env, consults config.DEFAULTS, and asks the hardware for
    the physical VRAM total. Everything downstream is pure policy on the
    resulting object.

    ``apply_physical_clamp=False`` is for the hill-climb search screen, which
    compares an estimate against the operator's *configured* budget and must not
    silently narrow its own search space against the device ceiling. The serving
    path keeps the clamp, because that is where the WDDM Sysmem Fallback guard
    applies.
    """
    keepout_env = os.environ.get("AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB")
    keepout = float(keepout_env) if keepout_env else DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB
    # config.DEFAULTS wins when the key exists; the env var is the operator
    # override on top of it. Mirrors resolve_vram_limit_mb's precedence.
    keepout = float(config.DEFAULTS.get("PHYSICAL_VRAM_KEEPOUT_MB", keepout))
    # resolve_vram_limit_mb applies the physical clamp itself. When the caller
    # opted out of that clamp we must therefore skip that resolver and pass the
    # configured value straight through, or the clamp would happen twice.
    resolved_limit = (
        float(vram_limit_mb)
        if (vram_limit_mb is not None and not apply_physical_clamp)
        else resolve_vram_limit_mb(vram_limit_mb)
    )
    return MemoryBudget.resolve(
        vram_limit_mb=resolved_limit,
        vram_headroom_mb=resolve_vram_headroom_mb(headroom_mb),
        shared_vram_limit_mb=resolve_shared_vram_limit_mb(),
        cuda_free_floor_mb=resolve_cuda_free_floor_mb(),
        physical_keepout_mb=keepout,
        free_clamp_enabled=free_vram_clamp_enabled(),
        total_vram_mb=_safe_detect_total_vram_mb() if apply_physical_clamp else None,
    )


def _safe_detect_total_vram_mb() -> float | None:
    """Physical VRAM total, or None when the probe is unavailable.

    An unknown device is not a reason to refuse a Trial, so probe failure
    degrades to None (no clamp) rather than to zero (clamp everything away).
    """
    try:
        total = detect_total_vram_mb()
    except Exception:
        return None
    if total is None or total <= 0:
        return None
    return float(total)


def preflight_vram_effective(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    vram_limit_mb: float | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
    headroom_mb: float | int | None = None,
    free_vram_mb: float | None = None,
) -> tuple[bool, float, str]:
    """Headroom wrapper around preflight_vram (issue #10).

    Effective budget = min(configured, free VRAM at Trial start - headroom).
    Exception: MoE with `n_cpu_moe > 0` uses configured only — OS-reserved VRAM
    otherwise false-rejects expert-CPU offload (measured peaks far below free).
    Runtime VRAM monitoring remains the OOM kill guard.
    The reject reason records both configured and effective budgets.

    The budget arithmetic is delegated to ``MemoryBudget``; the seven legacy
    resolvers remain exported below for backwards compatibility, but they are no
    longer the decision path.
    """
    budget = _build_memory_budget(vram_limit_mb, headroom_mb)
    configured = budget.vram_limit_mb
    moe_offload = n_cpu_moe is not None and int(n_cpu_moe) > 0
    clamp_on = budget.free_clamp_enabled
    if moe_offload or not clamp_on:
        effective = float(configured)
        if not moe_offload:
            print(
                f"  [vram-preflight] free-at-start clamp off (default; "
                "AUTORESEARCH_VRAM_FREE_CLAMP=1 restores) — budget="
                f"configured={configured:.0f}MB; runtime monitor remains kill guard",
                flush=True,
            )
    else:
        if free_vram_mb is None:
            free_vram_mb = detect_free_vram_mb()
        effective = budget.effective_vram_limit_mb(
            n_cpu_moe=n_cpu_moe,
            free_vram_mb=free_vram_mb,
            headroom_mb=headroom_mb,
        )
    ok, est, reason = preflight_vram(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
        vram_limit_mb=effective,
        n_cpu_moe=n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
    )
    if ok or effective == configured:
        return ok, est, reason
    enriched = budget.reject_reason(
        est,
        n_cpu_moe=n_cpu_moe,
        free_vram_mb=free_vram_mb,
        headroom_mb=headroom_mb,
    )
    return (False, est, enriched or reason)


def resolve_spec_estimate_args(
    model_path: Path | str,
    spec_type: str | None,
    spec_draft_n_max: int,
    draft_path: Path | str | None,
) -> tuple[str | None, bool, Path | str | None]:
    """Resolve effective speculative args for VRAM estimation (MTP-aware).

    Mirrors ServerIntent MTP inference: embedded-MTP GGUFs (declared via
    ``<arch>.nextn_predict_layers`` in the GGUF metadata) with no explicit
    SPEC_TYPE estimate with speculative workspace; external drafts only count
    when speculation is enabled. Metadata is authoritative, not the filename.
    """
    if spec_type is None and gguf_has_mtp(Path(model_path)):
        spec_type = "mtp"
    spec_enabled = is_spec_enabled(spec_type, spec_draft_n_max)
    return spec_type, spec_enabled, draft_path if spec_enabled else None


def preflight_vram_for_intent(
    intent: "ServerIntent",
    vram_limit_mb: float | None = None,
    headroom_mb: float | int | None = None,
) -> tuple[bool, float, str]:
    spec_type, _, draft_path = resolve_spec_estimate_args(
        intent.model_path,
        intent.spec_type,
        intent.spec_draft_n_max,
        intent.spec_draft_model,
    )
    return preflight_vram_effective(
        intent.model_path,
        intent.ctx_size,
        kv_cache_k=intent.kv_cache_k or intent.kv_cache,
        kv_cache_v=intent.kv_cache_v or intent.kv_cache,
        draft_path=draft_path,
        vram_limit_mb=vram_limit_mb,
        n_cpu_moe=intent.n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=intent.spec_draft_n_max,
        headroom_mb=headroom_mb,
    )


def _kv_est_mb(
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    model_path: Path | None = None,
) -> float:
    """KV cache estimate in MiB.

    Prefers GGUF-derived total f16 MiB (`gguf_kv_f16_mb`) — sliding-window
    arches (gemma4) only charge `min(ctx, window)` with SWA dims. Falls back to
    flat VRAM_KB_PER_TOKEN_F16. head_count_kv=0 (recurrent) → 0.
    """
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

    kv_base_mb = c_size * VRAM_KB_PER_TOKEN_F16 / 1024.0
    if model_path is not None:
        try:
            swa_mb = gguf_kv_f16_mb(model_path, c_size)
            if swa_mb is not None:
                kv_base_mb = float(swa_mb)
        except Exception:
            pass
    return (kv_base_mb / 2.0) * kf + (kv_base_mb / 2.0) * vf


# RAM charge fallback when the GGUF header cannot be read but n-cpu-moe is on:
# the expert share of the file (complement of the VRAM non-expert fraction).
MOE_EXPERT_RAM_FRAC_FALLBACK = 1.0 - VRAM_MOE_NON_EXPERT_FRAC


def ram_resident_bytes_mb(
    model_path: Path,
    n_cpu_moe: int | None = None,
    unified: bool | None = None,
) -> tuple[float, bool]:
    """System-RAM resident charge for the model (MiB) + whether it was shrunk.

    Discrete GPU + n-cpu-moe>0: CPU buffers hold expert-class tensors only
    (GGUF-derived bytes; fallback = file x MOE_EXPERT_RAM_FRAC_FALLBACK when
    the header cannot be read) — non-expert weights live in VRAM, the file
    itself stays mmap'd. Full file otherwise (unified hosts, dense, CPU-only).
    Operator decision 2026-09-04: the full-file charge false-rejected offloaded
    MoE loads that fit physical RAM with room to spare.
    """
    try:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
    except Exception:
        model_size_mb = 4000.0
    if n_cpu_moe is None or int(n_cpu_moe) <= 0:
        return model_size_mb, False
    if unified is None:
        try:
            from autoresearch.core.hardware import is_unified_memory_host

            unified = is_unified_memory_host()
        except Exception:
            unified = False
    if unified:
        return model_size_mb, False
    expert_mb = None
    try:
        expert_mb = gguf_expert_bytes_mb(model_path, layer_limit=int(n_cpu_moe))
    except Exception:
        expert_mb = None
    if expert_mb is None:
        return model_size_mb * MOE_EXPERT_RAM_FRAC_FALLBACK, True
    return expert_mb, True


def estimate_host_memory_mb(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    draft_path: Path | str | None = None,
    n_cpu_moe: int | None = None,
    unified: bool | None = None,
) -> float:
    """Host-RAM charge (MiB).

    Discrete + n-cpu-moe>0: expert bytes + overhead only — the offloaded
    experts fill CPU buffers while draft weights and KV stay on the GPU.
    Otherwise full GGUF + draft + KV + overhead."""
    ram_mb, shrunk = ram_resident_bytes_mb(model_path, n_cpu_moe, unified)
    if shrunk:
        return ram_mb + VRAM_OVERHEAD_MB

    draft_mb = 0.0
    if draft_path:
        try:
            draft_mb = Path(draft_path).stat().st_size / (1024 * 1024)
        except Exception:
            draft_mb = 0.0

    kv_est_mb = _kv_est_mb(
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        base_kv_cache=base_kv_cache,
        model_path=model_path,
    )
    return ram_mb + draft_mb + kv_est_mb + VRAM_OVERHEAD_MB


def preflight_host_memory(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    n_cpu_moe: int | None = None,
    headroom_mb: float | int | None = None,
    ram_mb: float | None = None,
    unified: bool | None = None,
) -> tuple[bool, float, float, str]:
    """Return (ok, estimate_mb, budget_mb, reason).

    Fail closed on unified hosts when RAM cannot be detected.
    On discrete NVIDIA with unknown RAM: warn via empty reason and pass (VRAM gate remains).
    """
    from autoresearch.core.hardware import (
        MEMORY_CLASS_DISCRETE,
        MEMORY_CLASS_UNIFIED,
        detect_host_ram_mb,
        is_unified_memory_host,
    )

    if unified is None:
        unified = is_unified_memory_host()
    mem_class = MEMORY_CLASS_UNIFIED if unified else MEMORY_CLASS_DISCRETE

    if ram_mb is None:
        ram_mb = detect_host_ram_mb()

    est = estimate_host_memory_mb(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
        n_cpu_moe=n_cpu_moe,
        unified=unified,
    )

    if ram_mb is None or ram_mb <= 0:
        if unified:
            return (
                False,
                est,
                0.0,
                f"HOST_MEMORY_PREFLIGHT ram_unknown class={mem_class} est={est:.0f}MB",
            )
        return True, est, 0.0, ""

    # The budget arithmetic (usable figure + rejection message) is owned by
    # MemoryBudget. Headroom resolution stays here because it reads env/config
    # and MemoryBudget is pure by contract.
    from autoresearch.core.hardware import resolve_host_headroom_mb

    resolved_headroom = resolve_host_headroom_mb(ram_mb, unified=unified, override_mb=headroom_mb)
    host = MemoryBudget.for_host(
        ram_mb,
        unified=unified,
        headroom_mb=resolved_headroom,
        memory_class=mem_class,
    )
    budget = host.host_budget_mb
    if budget is None:
        if unified:
            return (
                False,
                est,
                0.0,
                f"HOST_MEMORY_PREFLIGHT ram_unknown class={mem_class} est={est:.0f}MB",
            )
        return True, est, 0.0, ""

    reason = host.host_reject_reason(est)
    if reason is not None:
        return False, est, budget, reason
    return True, est, budget, ""


def preflight_host_memory_for_intent(
    intent: "ServerIntent",
    headroom_mb: float | int | None = None,
) -> tuple[bool, float, float, str]:
    if headroom_mb is None:
        headroom_mb = config.DEFAULTS.get("HOST_MEMORY_HEADROOM_MB")
    return preflight_host_memory(
        intent.model_path,
        intent.ctx_size,
        kv_cache_k=intent.kv_cache_k or intent.kv_cache,
        kv_cache_v=intent.kv_cache_v or intent.kv_cache,
        draft_path=intent.spec_draft_model,
        n_cpu_moe=intent.n_cpu_moe,
        headroom_mb=headroom_mb,
    )


LLAMA_BENCH_CANDIDATES = _binary_candidates("llama-bench")


def resolve_llama_server() -> Path:
    for candidate in LLAMA_SERVER_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-server not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_SERVER_CANDIDATES)
    )


def engine_version_tag(server: Path) -> str:
    """Engine identity for Trial evidence: `engine@tag` for versioned fork
    releases (`llama.cpp-releases/<engine>/<tag>/`), `""` for the stock
    submodule.

    AGENTS.md runtime policy: alternate engines land as versioned prebuilt
    releases under `llama.cpp-releases/<engine>/<tag>/`, selected via
    `AUTORESEARCH_LLAMA_CPP_ROOT`. Trials must preserve engine/tag so rows
    from different forks stay distinguishable.
    """
    parts = server.parts
    for i, part in enumerate(parts):
        if part == "llama.cpp-releases" and i + 2 < len(parts):
            return f"{parts[i + 1]}@{parts[i + 2]}"
    return ""


def resolve_llama_bench() -> Path:
    for candidate in LLAMA_BENCH_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-bench not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_BENCH_CANDIDATES)
    )


LLAMA_CLI_CANDIDATES = _binary_candidates("llama-cli")


def resolve_llama_cli() -> Path:
    for candidate in LLAMA_CLI_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-cli not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_CLI_CANDIDATES)
    )


LLAMA_PERPLEXITY_CANDIDATES = _binary_candidates("llama-perplexity")


def resolve_llama_perplexity() -> Path:
    for candidate in LLAMA_PERPLEXITY_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-perplexity not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_PERPLEXITY_CANDIDATES)
    )


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))


def sweep_leftover_processes() -> None:
    """Best-effort pre-flight orphan sweep (ADR 0010 decision 2).

    Kills leftover harness processes holding a harness port before a server
    binds. Never blocks startup: a sweep failure (e.g. subprocess tooling
    unavailable or mocked out) logs and moves on — the port bind would surface
    EADDRINUSE on its own.
    """
    try:
        killed = cleanup_leftover_processes()
    except Exception as exc:
        print(f"  [process-guard] pre-flight orphan sweep skipped: {exc}")
        return
    if killed:
        print(f"  [process-guard] pre-flight killed leftover harness procs: {sorted(killed)}")


WATCHDOG_KILL_LEGACY_MARKER = "VRAM_LIMIT_EXCEEDED"
"""Pre-fix watchdog kill text: always a policy kill, never a model OOM."""

WATCHDOG_KILL_LEGACY_REASON = (
    f"WATCHDOG_KILL (legacy {WATCHDOG_KILL_LEGACY_MARKER}, scope=nvml-device-wide)"
)
"""Scope-less kills predate the CUDA-free guard: all were NVML device-wide (issue #72)."""


def watchdog_kill_reason(scope: str, detail: str, where: str) -> str:
    """One formatter for every watchdog policy-kill reason (issue #72).

    Keeps the ``WATCHDOG_KILL scope=<scope> <detail> (<where>)`` shape a single
    contract shared by the server sampler and the llama-cli bench watcher.
    """
    return f"WATCHDOG_KILL scope={scope} {detail} ({where})"


class LlamaServerRunner:
    def __init__(
        self,
        intent: ServerIntent,
        log_path: Path | None = None,
        vram_limit_mb: float | None = None,
    ):
        self.intent = intent
        self.log_path = log_path
        self.vram_limit_mb = resolve_vram_limit_mb(vram_limit_mb)
        self.shared_vram_limit_mb = resolve_shared_vram_limit_mb()

        self.port: int | None = None
        self.peak_vram_mb: float = 0.0
        self.peak_shared_mb: float = 0.0
        self.vram_killed: bool = False
        self.vram_kill_reason: str = ""
        self.ram_killed: bool = False

        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        self._vram_thread: threading.Thread | None = None
        self._ram_thread: threading.Thread | None = None
        self._guard: ProcessGuard | None = None

        self.llama_server = resolve_llama_server()

    def _build_cmd(self, target_port: int) -> list[str]:
        cache_type_k = (
            self.intent.kv_cache_k if self.intent.kv_cache_k is not None else self.intent.kv_cache
        )
        cache_type_v = (
            self.intent.kv_cache_v if self.intent.kv_cache_v is not None else self.intent.kv_cache
        )

        cmd = [
            str(self.llama_server),
            "--model",
            str(self.intent.model_path),
            "--host",
            str(self.intent.host),
            "--port",
            str(target_port),
            "--ctx-size",
            str(self.intent.ctx_size),
            "--batch-size",
            str(self.intent.batch_size),
            "--ubatch-size",
            str(self.intent.ubatch_size),
            "--threads",
            str(self.intent.threads),
            "--parallel",
            str(self.intent.parallel),
            "--n-gpu-layers",
            str(self.intent.ngl),
        ]

        if self.intent.numa:
            cmd += ["--numa", self.intent.numa]

        cmd += [
            "--cache-type-k",
            cache_type_k,
            "--cache-type-v",
            cache_type_v,
            "--flash-attn",
            self.intent.flash_attn,
        ]

        if self.intent.threads_batch is not None:
            cmd += ["--threads-batch", str(self.intent.threads_batch)]

        cmd += load_mode_flag(self.intent.no_mmap, self.intent.mlock)
        if self.intent.jinja:
            cmd += ["--jinja"]
        if self.intent.reasoning_budget is not None:
            cmd += ["--reasoning-budget", str(self.intent.reasoning_budget)]
        if self.intent.reasoning_budget_message is not None:
            cmd += ["--reasoning-budget-message", self.intent.reasoning_budget_message]
        if self.intent.reasoning is not None:
            cmd += ["--reasoning", str(self.intent.reasoning)]
        if self.intent.reasoning_effort is not None:
            cmd += ["--reasoning-effort", str(self.intent.reasoning_effort)]
        if self.intent.reasoning_preserve is True:
            cmd += ["--reasoning-preserve"]
        elif self.intent.reasoning_preserve is False:
            cmd += ["--no-reasoning-preserve"]
        if self.intent.cont_batching:
            cmd += ["--cont-batching"]
        if self.intent.cache_reuse is not None and int(self.intent.cache_reuse) > 0:
            cmd += ["--cache-reuse", str(int(self.intent.cache_reuse))]
        spec_type_val = self.intent.spec_type
        if (
            spec_type_val is None
            and gguf_has_mtp(self.intent.model_path)
            and self.intent.spec_draft_n_max > 0
        ):
            global _LLAMA_SERVER_HELP_CACHE
            if _LLAMA_SERVER_HELP_CACHE is None:
                try:
                    _LLAMA_SERVER_HELP_CACHE = subprocess.check_output(
                        [str(self.llama_server), "--help"], stderr=subprocess.STDOUT, text=True
                    )
                except Exception:
                    _LLAMA_SERVER_HELP_CACHE = "mtp"
            if "mtp" in _LLAMA_SERVER_HELP_CACHE:
                spec_type_val = "mtp"
            else:
                spec_type_val = "draft-mtp"
            print(
                f"  [MTP] Multi-Token Prediction detected for {self.intent.model_path.name}. Auto-selected spec-type: {spec_type_val}"
            )

        if is_spec_enabled(spec_type_val, self.intent.spec_draft_n_max):
            cmd += [
                "--spec-type",
                spec_type_val,
            ]
            if self.intent.spec_draft_n_max > 0:
                cmd += [
                    "--spec-draft-n-max",
                    str(self.intent.spec_draft_n_max),
                    "--spec-draft-type-k",
                    cache_type_k,
                    "--spec-draft-type-v",
                    cache_type_v,
                ]
            if self.intent.spec_draft_model:
                draft_path = Path(self.intent.spec_draft_model)
                if not draft_path.is_absolute():
                    draft_path = self.intent.model_path.parent / draft_path
                cmd += ["--spec-draft-model", str(draft_path)]

        # codacus-fork expert-cache passthrough (not present in upstream common args).
        if self.intent.moe_cache_profile:
            profile = Path(self.intent.moe_cache_profile)
            if not profile.is_absolute():
                profile = Path.cwd() / profile
            cmd += ["--moe-cache-profile", str(profile)]
            if self.intent.moe_cache_slots:
                cmd += ["--moe-cache-slots", str(int(self.intent.moe_cache_slots))]

        # VITRIOL: MoE expert offload only (resolved in from_config). Dense: no flag.
        if self.intent.n_cpu_moe is not None:
            print(
                f"  [VITRIOL] MoE Expert Streaming: --n-cpu-moe {self.intent.n_cpu_moe} "
                f"for {self.intent.model_path.name}."
            )
            cmd += ["--n-cpu-moe", str(self.intent.n_cpu_moe)]

        assert_flags_supported(cmd, self.llama_server)
        return cmd

    def __enter__(self):
        try:
            return self._start()
        except BaseException:
            # `with` never entered, so __exit__ never runs: release the VRAM
            # sampler, any spawned process and the ProcessGuard here. Every step
            # is None-guarded and idempotent, so the explicit _cleanup_all()
            # calls on the port-failover paths in _start() stay correct.
            self._cleanup_all()
            raise

    def _start(self):
        self._start_vram_sampler()

        server_env = os.environ.copy()
        llama_lib_dir = str(self.llama_server.parent)
        lib_path_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
        existing = server_env.get(lib_path_var, "")
        server_env[lib_path_var] = (
            f"{llama_lib_dir}{os.pathsep}{existing}" if existing else llama_lib_dir
        )
        # Avoid CUDA pinned host maps that WDDM counts as Shared GPU (pagefile freeze).
        server_env["GGML_CUDA_NO_PINNED"] = "1"

        # Single-load gate (#41): refuse a second full server while one is
        # live. The pre-flight orphan sweep would kill a live sibling on a
        # harness port, so it only runs when the gate passes and allow-multi
        # is off.
        allow_multi = resolve_allow_multi()
        enforce_single_load(allow_multi=allow_multi)
        if not allow_multi:
            sweep_leftover_processes()
        self._guard = ProcessGuard()

        # Permanent RAM circuit breaker: refuse a launch that cannot fit in RAM.
        # Small launch margin; the runtime watchdog enforces the real floor.
        margin = float(
            config.DEFAULTS.get(
                "RAM_PREFLIGHT_MARGIN_MB", circuit_breaker.DEFAULT_PREFLIGHT_MARGIN_MB
            )
        )
        try:
            resident_mb, _ = ram_resident_bytes_mb(self.intent.model_path, self.intent.n_cpu_moe)
            circuit_breaker.preflight_ram(int(resident_mb * 1024 * 1024), margin_mb=margin)
        except circuit_breaker.CircuitBreakerError as exc:
            print(f"  [RAM] {exc}", flush=True)
            raise

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

            self._server_proc = self._guard.spawn(
                cmd,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                env=server_env,
                text=True,
            )

            if self._server_proc.pid is not None:
                self._ram_thread = circuit_breaker.start_ram_watchdog(
                    self._server_proc.pid,
                    floor_mb=float(
                        config.DEFAULTS.get("FREE_RAM_FLOOR_MB", circuit_breaker.DEFAULT_FLOOR_MB)
                    ),
                    poll_s=float(
                        config.DEFAULTS.get("RAM_WATCHDOG_POLL_S", circuit_breaker.DEFAULT_POLL_S)
                    ),
                    reserve_mb=float(
                        config.DEFAULTS.get(
                            "RAM_WATCHDOG_RESERVE_MB", circuit_breaker.DEFAULT_RESERVE_MB
                        )
                    ),
                    on_kill=self._maybe_kill_ram,
                )

            self.port = port
            if self._wait_for_server(port):
                return self

            if self.vram_killed:
                self._cleanup_all()
                raise RuntimeError(self.vram_kill_reason or WATCHDOG_KILL_LEGACY_REASON)
            if self.ram_killed:
                self._cleanup_all()
                raise RuntimeError("RAM_CIRCUIT_BREAKER")

            # If wait failed, grab the tail before cleaning up
            self._server_log.flush()
            if hasattr(self._server_log, "name"):
                log_content = Path(self._server_log.name).read_text(
                    encoding="utf-8", errors="replace"
                )
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

    def _wait_for_server(self, port: int) -> bool:
        delay = 0.05
        deadline = time.monotonic() + SERVER_HEALTH_TIMEOUT_SECONDS
        while True:
            if self._server_proc is None or self._server_proc.poll() is not None:
                return False
            if time.monotonic() >= deadline:
                print(
                    f"  [server] /health did not return 200 within "
                    f"{SERVER_HEALTH_TIMEOUT_SECONDS:.0f}s on port {port}; failing over."
                )
                return False
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                # A listening but wedged server must not block startup forever.
                with urllib.request.urlopen(req, timeout=2.0) as response:
                    if response.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 2, 0.4)

    @property
    def is_cpu_mode(self) -> bool:
        return (
            "build-cpu" in str(self.llama_server).lower()
            or (self.intent is not None and self.intent.ngl == 0)
            or not should_prefer_gpu_build()
        )

    def _start_vram_sampler(self) -> None:
        if self.is_cpu_mode:
            return

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
            nvml_name = "nvml.dll" if IS_WINDOWS else "libnvidia-ml.so.1"
            nvml = ctypes.CDLL(nvml_name)
            nvml.nvmlInit_v2()
            device = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device))
            print("  [VRAM] NVML initialized successfully. High-frequency 20ms sampling enabled.")
        except Exception:
            nvml = None
            print(
                "  [VRAM] NVML initialization failed. Falling back to subprocess nvidia-smi (200ms)."
            )

        def _maybe_kill_shared() -> None:
            """Kill absolute Shared GPU overshoot (dedicated can stay ~4–5 GB)."""
            if self.vram_killed or self._server_proc is None or self._server_proc.pid is None:
                return
            shared = detect_pid_gpu_shared_mb(int(self._server_proc.pid))
            if shared is None:
                return
            if shared > self.peak_shared_mb:
                self.peak_shared_mb = shared
            if shared > shared_limit and not self.vram_killed:
                self.vram_killed = True
                kind = "dense" if dense else "moe"
                self.vram_kill_reason = watchdog_kill_reason(
                    "shared",
                    f"shared={shared:.0f}MB>limit={shared_limit:.0f}MB",
                    f"{kind}={self.intent.model_path.name}",
                )
                print(
                    f"  [VRAM] SHARED GPU EXCEEDED shared={shared:.0f}MB > "
                    f"limit={shared_limit:.0f}MB ({kind}={self.intent.model_path.name}) — killing "
                    "(WDDM/PCI-e Shared→pagefile freeze)",
                    flush=True,
                )
                self._cleanup_process()
                self._stop_event.set()

        cuda_guard = True
        try:
            from autoresearch.core.hardware import cuda_free_mb

            probe = cuda_free_mb()
        except Exception:
            probe = None
        if probe is None:
            cuda_guard = False
            print(
                "  [VRAM] CUDA-free probe unavailable — NVML used>ceil kill remains active.",
                flush=True,
            )
        else:
            print(
                f"  [VRAM] CUDA-free guard active (floor={resolve_cuda_free_floor_mb():.0f}MB; "
                "NVML used is peak-metric only — desktop committed memory is evictable).",
                flush=True,
            )

        dense = is_dense_model(self.intent.model_path)
        limit = self.vram_limit_mb
        shared_limit = self.shared_vram_limit_mb
        shared_tick = 0

        def _maybe_kill(
            current: float, total_mb: float | None = None, cuda_free_now: float | None = None
        ) -> None:
            if current > self.peak_vram_mb:
                self.peak_vram_mb = current
            if self.vram_killed:
                return
            kind = "dense" if dense else "moe"
            if cuda_guard:
                if cuda_free_now is None:
                    return  # transient probe miss: no kill decision this tick
                floor = resolve_cuda_free_floor_mb()
                if cuda_free_now < floor:
                    self.vram_killed = True
                    self.vram_kill_reason = watchdog_kill_reason(
                        "cuda_free",
                        f"cuda_free={cuda_free_now:.0f}MB<floor={floor:.0f}MB",
                        f"{kind}={self.intent.model_path.name}",
                    )
                    print(
                        f"  [VRAM] CUDA FREE EXHAUSTED cuda_free={cuda_free_now:.0f}MB < "
                        f"floor={floor:.0f}MB ({kind}={self.intent.model_path.name}) — killing "
                        "(allocation failure imminent)",
                        flush=True,
                    )
                    self._cleanup_process()
                    self._stop_event.set()
                return
            ceil = dedicated_vram_kill_ceil(limit, total_mb)
            if current > ceil:
                self.vram_killed = True
                self.vram_kill_reason = watchdog_kill_reason(
                    "nvml-used",
                    f"used={current:.0f}MB>ceil={ceil:.0f}MB",
                    f"{kind}={self.intent.model_path.name}",
                )
                print(
                    f"  [VRAM] LIMIT EXCEEDED used={current:.0f}MB > limit={ceil:.0f}MB "
                    f"({kind}={self.intent.model_path.name}) — killing "
                    "(no Windows shared-GPU / pagefile spill)",
                    flush=True,
                )

        probe_fail_streak = 0

        def sampler() -> None:
            nonlocal nvml, shared_tick, cuda_guard, probe_fail_streak
            probe_fail_streak = 0
            while not self._stop_event.is_set():
                if nvml is not None and device is not None:
                    try:
                        mem_info = nvmlMemory_t()
                        nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(mem_info))
                        current = float(mem_info.used) / (1024.0 * 1024.0)
                        total_mb = float(mem_info.total) / (1024.0 * 1024.0)
                        cuda_cf = cuda_free_mb() if cuda_guard else None
                        if cuda_cf is None and cuda_guard:
                            probe_fail_streak += 1
                            if probe_fail_streak >= 5:
                                cuda_guard = False
                                from autoresearch.core.hardware import cuda_probe_last_error

                                last_err = cuda_probe_last_error()
                                print(
                                    "  [VRAM] CUDA-free probe unstable — NVML used>ceil fallback engaged"
                                    + (f" (last cuMemGetInfo ret={last_err})" if last_err else "")
                                    + ".",
                                    flush=True,
                                )
                        else:
                            probe_fail_streak = 0
                        _maybe_kill(current, total_mb, cuda_free_now=cuda_cf)
                        shared_tick += 1
                        # typeperf is slow — sample Shared ~1/s while NVML is 20ms.
                        if shared_tick % 50 == 0:
                            _maybe_kill_shared()
                        self._stop_event.wait(0.02)
                    except Exception:
                        nvml = None
                try:
                    current, total_mb = detect_used_total_vram_mb()
                    _maybe_kill(
                        current, total_mb, cuda_free_now=cuda_free_mb() if cuda_guard else None
                    )
                    _maybe_kill_shared()
                except FileNotFoundError:
                    # VRAM sampling unavailable on non-GPU host
                    break
                except (subprocess.CalledProcessError, ValueError):
                    pass
                self._stop_event.wait(0.2)

        self._vram_thread = threading.Thread(target=sampler, daemon=True)
        self._vram_thread.start()

    def _maybe_kill_ram(self, reason: str) -> None:
        """RAM circuit-breaker callback: kill the server before swap spill."""
        if self.ram_killed:
            return
        self.ram_killed = True
        print(
            f"  [RAM] CIRCUIT BREAKER: {reason} ({self.intent.model_path.name}) — "
            "killing to prevent pagefile swap spill",
            flush=True,
        )
        self._cleanup_process()
        self._stop_event.set()

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
        self._cleanup_process()
        if self._guard:
            self._guard.teardown()
            self._guard = None
