"""Architecture class from GGUF metadata (not filename heuristics).

Model cards under docs/models/ must mirror this GGUF truth. Runtime decisions
(VITRIOL / N_CPU_MOE / dense VRAM kill) read the local GGUF only.

MoE Baseline `N_CPU_MOE`:
  None → auto `--n-cpu-moe {block_count}` from GGUF
  0    → full GPU (`--n-cpu-moe 0`)
  N>0  → manual offload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class GGUFMetadataError(RuntimeError):
    """GGUF architecture could not be read, so MoE-vs-dense is unknown.

    Never degrade this to "dense": dense means "no expert offload", so an
    unreadable MoE GGUF silently puts every expert on the GPU and WDDM spills
    the overflow into host RAM (measured on an 8 GB-class card: ~14 GB GGUF
    on 8 GB VRAM -> ~9 GB Shared GPU). Raising keeps the offload decision,
    the dense VRAM kill and the host-memory preflight from all reading a wrong
    answer.
    """


# Skip junk while resolving basename under models/
_MODEL_SEARCH_SKIP = frozenset({".cache", "aliases", "huggingface", "vision"})

# (resolved_path, mtime_ns) → (is_moe, block_count|None)
_ARCH_CACHE: dict[tuple[str, int], tuple[bool, int | None]] = {}

# (resolved_path, mtime_ns) → bool (embedded MTP draft heads present)
_MTP_CACHE: dict[tuple[str, int], bool] = {}


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def resolve_model_file(
    ref: str | Path,
    models_dir: Path | None = None,
    *,
    allow_dirs: bool = False,
) -> Path | None:
    """Resolve a model ref under models_dir (flat or nested LM Studio layout).

    Order: absolute → models_dir/ref if present → rglob basename (skip junk dirs).
    Returns an existing path, or None if missing. Set allow_dirs=True for SGLang
    directory checkpoints (resolve_model_path uses that).
    """
    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    ref_path = Path(ref)

    def _ok(path: Path) -> bool:
        return path.is_file() or (allow_dirs and path.is_dir())

    if ref_path.is_absolute():
        return ref_path if _ok(ref_path) else None

    direct = models_dir / ref_path
    if _ok(direct):
        return direct

    name = ref_path.name
    matches: list[Path] = []
    if models_dir.is_dir():
        for path in models_dir.rglob(name):
            if any(part in _MODEL_SEARCH_SKIP for part in path.parts):
                continue
            if _ok(path):
                matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: (len(p.relative_to(models_dir).parts), str(p).lower()))
    return matches[0]


def _field_contents(field: Any) -> Any:
    try:
        return field.contents()
    except Exception:
        return None


def _gguf_arch_info(path: Path) -> tuple[bool, int | None]:
    """One GGUF open: (is_moe, block_count). Cached by path+mtime."""
    resolved = path.resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    cache_key = (str(resolved), mtime_ns)
    hit = _ARCH_CACHE.get(cache_key)
    if hit is not None:
        return hit

    try:
        from gguf import GGUFReader
    except ImportError as exc:
        # A wrong interpreter (system python without requirements.txt) used to
        # fall through to "dense" here - the exact silent spill this repo N_CPU_MOE
        # design exists to prevent.
        raise GGUFMetadataError(
            f"GGUF architecture unreadable: the 'gguf' package is not importable by "
            f"{sys.executable!r}. Use the repo venv (requirements.txt installs gguf); "
            "without the reader a MoE GGUF is misclassified as dense and loses --n-cpu-moe."
        ) from exc

    try:
        reader = GGUFReader(str(path))
        is_moe = False
        block_count: int | None = None

        for key, field in reader.fields.items():
            kl = str(key).lower()
            if kl.endswith(".expert_count") or kl == "expert_count":
                raw = _field_contents(field)
                try:
                    if int(raw) > 1:
                        is_moe = True
                except (TypeError, ValueError):
                    pass
            if block_count is None and (kl.endswith(".block_count") or kl == "block_count"):
                raw = _field_contents(field)
                try:
                    block_count = int(raw)
                except (TypeError, ValueError):
                    pass

        if not is_moe:
            arch_field = reader.fields.get("general.architecture")
            if arch_field is not None:
                arch = str(_field_contents(arch_field) or "").lower()
                if "moe" in arch:
                    is_moe = True

    except GGUFMetadataError:
        raise
    except Exception as exc:
        raise GGUFMetadataError(f"cannot read GGUF metadata from {path}: {exc}") from exc

    _ARCH_CACHE[cache_key] = (is_moe, block_count)
    return is_moe, block_count


def gguf_has_mtp(path: Path) -> bool:
    """True when the GGUF declares embedded MTP draft heads.

    Reads any ``<arch>.nextn_predict_layers`` key (arch-agnostic) and returns
    True when its value is > 0. This replaces the old filename "MTP" heuristic:
    file naming alone is not proof (see data/perplexity_val.txt §MTP audit).
    Missing/unreadable file or absent key → False.
    """
    resolved = path.resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    cache_key = (str(resolved), mtime_ns)
    if cache_key in _MTP_CACHE:
        return _MTP_CACHE[cache_key]

    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        for key, field in reader.fields.items():
            if str(key).lower().endswith(".nextn_predict_layers"):
                raw = _field_contents(field)
                try:
                    if int(raw) > 0:
                        _MTP_CACHE[cache_key] = True
                        return True
                except (TypeError, ValueError):
                    pass
        _MTP_CACHE[cache_key] = False
        return False
    except Exception:
        _MTP_CACHE[cache_key] = False
        return False


def gguf_is_moe(path: Path) -> bool:
    """True when local GGUF metadata shows routed experts (MoE / hybrid-MoE)."""
    is_moe, _ = _gguf_arch_info(path)
    return is_moe


def gguf_block_count(path: Path) -> int:
    """Return GGUF `*.block_count`. Raises ValueError if missing/invalid."""
    _, block_count = _gguf_arch_info(path)
    if block_count is None or block_count < 1:
        raise ValueError(f"GGUF missing valid block_count: {path}")
    return block_count


def gguf_expert_bytes_mb(path: Path, layer_limit: int | None = None) -> float | None:
    """Sum expert-class tensor bytes in MiB from GGUF tensor metadata (header-only).

    Expert-class = the tensor families a ``--n-cpu-moe`` style buffer-type
    override routes to CPU: routed FFN experts (``ffn_*(ch|)exps``) and
    attention value experts (``attn_v_exps``, MoVA-style). Returns None when the
    file cannot be read or carries no expert tensors (dense GGUFs).
    ``layer_limit`` (first N blocks, as ``--n-cpu-moe N``) scales the sum.
    """
    import re

    try:
        from gguf import GGML_QUANT_SIZES, GGUFReader
    except ImportError:
        return None
    pattern = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps(\.|$)|\.attn_v_exps(\.|$)")
    try:
        reader = GGUFReader(str(path))
        total_bytes = 0
        for tensor in reader.tensors:
            if not pattern.search(tensor.name):
                continue
            n_elem = 1
            for dim in tensor.shape:
                n_elem *= dim
            blck_size, type_size = GGML_QUANT_SIZES[tensor.tensor_type]
            total_bytes += ((n_elem + blck_size - 1) // blck_size) * type_size
        if total_bytes <= 0:
            return None
    except Exception:
        return None
    total_mb = total_bytes / (1024.0 * 1024.0)
    if layer_limit is not None:
        try:
            blocks = gguf_block_count(path)
            total_mb *= min(1.0, float(layer_limit) / float(blocks))
        except Exception:
            pass
    return total_mb


def gguf_kv_bytes_per_token_f16(path: Path) -> float | None:
    """KV cache bytes per token at f16 from GGUF metadata, or None if unknown.

    Mirrors llama.cpp: `n_head_kv` defaults to `head_count` when the key is
    absent; an explicit `head_count_kv = 0` marks recurrent layers -> no KV
    cache (e.g. LFM2.5-8B-A1B, head_count_kv=0, measured KV ~0). Bytes per
    token = n_layer * n_head_kv * (key_length + value_length), key/value
    lengths defaulting to head_dim = embedding_length / head_count.

    For sliding-window arches (e.g. gemma4 SWA), prefer
    `gguf_kv_f16_mb(path, ctx_size)` — per-token scaling with full ctx is wrong.
    """
    meta = _gguf_kv_meta(path)
    if meta is None:
        return None
    if meta.get("has_swa"):
        # Ambiguous under full-ctx scaling; callers must use gguf_kv_f16_mb.
        return None
    total_heads = 0
    k_len = meta["k_len"]
    v_len = meta["v_len"]
    for n_kv in meta["per_layer_kv"]:
        if n_kv > 0:
            total_heads += n_kv
    return float(total_heads) * (k_len + v_len)


def gguf_kv_f16_mb(path: Path, ctx_size: int) -> float | None:
    """Total f16 KV cache size in MiB for ``ctx_size`` from GGUF metadata.

    Honors per-layer ``head_count_kv``, sliding-window pattern/window, and
    SWA key/value lengths (gemma4). Non-SWA arches = bytes/token × ctx.

    Quant scaling (calibrated 2026-08-20, b10375, 8 GB-class): the f16
    estimate matches measured peaks at ×1; scale by bits/16 for the actual
    cache type — q8_0 ×0.5, q4_0/tq4 ×0.25, tq2_0 ×0.125, tq1_0 ×0.0625.
    Total VRAM ≈ resident weights + scaled KV + residual (~1.0 GiB on
    8 GB-class: CUDA ctx + compute buffers + allocator slack; the residual
    grows with ctx prefill scratch — budget margin accordingly).
    """
    meta = _gguf_kv_meta(path)
    if meta is None:
        return None
    try:
        ctx = int(ctx_size)
    except (TypeError, ValueError):
        ctx = 16384
    if ctx <= 0:
        return 0.0

    pattern = meta.get("swa_pattern")
    window = meta.get("swa_window")
    k_len = meta["k_len"]
    v_len = meta["v_len"]
    k_swa = meta.get("k_len_swa") or k_len
    v_swa = meta.get("v_len_swa") or v_len
    per_layer = meta["per_layer_kv"]

    total_cells = 0.0
    for i, n_kv in enumerate(per_layer):
        if n_kv <= 0:
            continue
        is_swa = bool(pattern[i]) if pattern is not None and i < len(pattern) else False
        if is_swa and window is not None and window > 0:
            tok = min(ctx, int(window))
            kl, vl = int(k_swa), int(v_swa)
        else:
            tok = ctx
            kl, vl = int(k_len), int(v_len)
        # Element-cells (k+v dims × heads × tokens). Unit matches legacy
        # gguf_kv_bytes_per_token_f16 (name says bytes; value is dim-cells —
        # calibrated to measured llama.cpp peaks without an extra ×2).
        total_cells += float(n_kv) * float(kl + vl) * float(tok)
    return total_cells / (1024.0 * 1024.0)


def _gguf_kv_meta(path: Path) -> dict[str, Any] | None:
    """Shared GGUF attention metadata for KV sizing. None if unreadable/incomplete."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        reader = GGUFReader(str(path))
        arch = str(_field_contents(reader.fields.get("general.architecture")) or "").lower()
        if not arch:
            return None

        def _num(key: str) -> int | None:
            raw = _field_contents(reader.fields.get(key))
            try:
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None

        def _bools(key: str) -> list[bool] | None:
            raw = _field_contents(reader.fields.get(key))
            if raw is None:
                return None
            vals = raw if isinstance(raw, (list, tuple)) else [raw]
            out: list[bool] = []
            for v in vals:
                if isinstance(v, str):
                    out.append(v.strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(v, (bool, int, float)):
                    out.append(bool(v))
                else:
                    return None
            return out or None

        n_layer = _num(f"{arch}.block_count")
        n_embd = _num(f"{arch}.embedding_length")
        n_head = _num(f"{arch}.attention.head_count")
        if n_layer is None or n_embd is None or n_head is None or n_head == 0:
            return None

        kv_raw = _field_contents(reader.fields.get(f"{arch}.attention.head_count_kv"))
        if kv_raw is None:
            per_layer = [n_head] * n_layer
        else:
            vals = kv_raw if isinstance(kv_raw, (list, tuple)) else [kv_raw]
            try:
                parsed = [int(v) for v in vals]
            except (TypeError, ValueError):
                parsed = []
            if not parsed:
                per_layer = [n_head] * n_layer
            elif len(parsed) == 1:
                per_layer = [parsed[0]] * n_layer
            elif len(parsed) >= n_layer:
                per_layer = parsed[:n_layer]
            else:
                # Pad short arrays with 0 (unknown trailing layers)
                per_layer = parsed + [0] * (n_layer - len(parsed))

        head_dim = n_embd / n_head
        k_len = _num(f"{arch}.attention.key_length")
        v_len = _num(f"{arch}.attention.value_length")
        if k_len is None:
            k_len = int(head_dim)
        if v_len is None:
            v_len = int(head_dim)

        swa_pattern = _bools(f"{arch}.attention.sliding_window_pattern")
        swa_window = _num(f"{arch}.attention.sliding_window")
        k_swa = _num(f"{arch}.attention.key_length_swa")
        v_swa = _num(f"{arch}.attention.value_length_swa")
        has_swa = bool(swa_pattern) and swa_window is not None and swa_window > 0

        return {
            "per_layer_kv": per_layer,
            "k_len": k_len,
            "v_len": v_len,
            "k_len_swa": k_swa,
            "v_len_swa": v_swa,
            "swa_pattern": swa_pattern if has_swa else None,
            "swa_window": swa_window if has_swa else None,
            "has_swa": has_swa,
        }
    except Exception:
        return None


def is_moe_model(ref: str | Path, *, models_dir: Path | None = None) -> bool:
    """Classify MoE from GGUF metadata. Missing file → not MoE (no MoE without a file).

    An existing but unreadable GGUF raises GGUFMetadataError instead of
    reporting "dense" - dense means "no expert offload", the dangerous default.
    """
    path = resolve_model_file(ref, models_dir=models_dir)
    if path is None:
        return False
    return gguf_is_moe(path)


def is_dense_model(ref: str | Path, *, models_dir: Path | None = None) -> bool:
    return not is_moe_model(ref, models_dir=models_dir)


def resolve_n_cpu_moe(path: Path, n_cpu_moe: int | None) -> tuple[int | None, bool]:
    """Resolve effective `--n-cpu-moe` for a local GGUF.

    Returns (resolved_n, auto) where auto=True when Baseline None was replaced
    by GGUF block_count. Dense → (None, False). MoE+None → (block_count, True).
    MoE+0/N → (N, False).

    Missing GGUF: keep explicit N; with None return (None, False)
    (dense-safe — no MoE without a file). Existing but unreadable GGUF
    with None → ValueError (auto needs readable metadata).
    """
    if not path.is_file():
        if n_cpu_moe is not None:
            return int(n_cpu_moe), False
        return None, False

    try:
        is_moe, _ = _gguf_arch_info(path)
    except Exception as exc:
        if n_cpu_moe is not None:
            return int(n_cpu_moe), False
        raise ValueError(f"cannot read GGUF architecture for auto N_CPU_MOE: {path}") from exc

    if not is_moe:
        return None, False
    if n_cpu_moe is not None:
        return int(n_cpu_moe), False
    try:
        return gguf_block_count(path), True
    except ValueError as exc:
        raise ValueError(
            f"MoE GGUF {path.name!r} needs block_count for auto N_CPU_MOE; "
            "set N_CPU_MOE explicitly or fix the GGUF metadata"
        ) from exc
