"""Architecture class from GGUF metadata (not filename heuristics).

Model cards under docs/models/ must mirror this GGUF truth. Runtime decisions
(VITRIOL / N_CPU_MOE / dense VRAM kill) read the local GGUF only.

MoE Baseline `N_CPU_MOE`:
  None → auto `--n-cpu-moe {block_count}` from GGUF
  0    → full GPU (`--n-cpu-moe 0`)
  N>0  → manual offload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Skip junk while resolving basename under models/
_MODEL_SEARCH_SKIP = frozenset({".cache", "aliases", "huggingface", "vision"})

# (resolved_path, mtime_ns) → (is_moe, block_count|None)
_ARCH_CACHE: dict[tuple[str, int], tuple[bool, int | None]] = {}


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def resolve_model_file(ref: str | Path, models_dir: Path | None = None) -> Path | None:
    """Resolve a model ref to an existing GGUF path, or None if missing."""
    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path if ref_path.is_file() else None

    direct = models_dir / ref_path
    if direct.is_file():
        return direct

    name = ref_path.name
    matches: list[Path] = []
    if models_dir.is_dir():
        for path in models_dir.rglob(name):
            if any(part in _MODEL_SEARCH_SKIP for part in path.parts):
                continue
            if path.is_file():
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
        raise RuntimeError("gguf package required to classify model architecture") from exc

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

    _ARCH_CACHE[cache_key] = (is_moe, block_count)
    return is_moe, block_count


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


def is_moe_model(ref: str | Path, *, models_dir: Path | None = None) -> bool:
    """Classify MoE from GGUF metadata. Missing/unreadable file → not MoE (dense-safe)."""
    path = resolve_model_file(ref, models_dir=models_dir)
    if path is None:
        return False
    try:
        return gguf_is_moe(path)
    except Exception:
        return False


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
        raise ValueError(
            f"cannot read GGUF architecture for auto N_CPU_MOE: {path}"
        ) from exc

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
