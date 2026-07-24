"""Architecture class from GGUF metadata (not filename heuristics).

Model cards under docs/models/ must mirror this GGUF truth. Runtime decisions
(VITRIOL / N_CPU_MOE / dense VRAM kill) read the local GGUF only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Skip junk while resolving basename under models/
_MODEL_SEARCH_SKIP = frozenset({".cache", "aliases", "huggingface", "vision"})


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


def gguf_is_moe(path: Path) -> bool:
    """True when local GGUF metadata shows routed experts (MoE / hybrid-MoE)."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError("gguf package required to classify model architecture") from exc

    reader = GGUFReader(str(path))
    for key, field in reader.fields.items():
        kl = str(key).lower()
        if kl.endswith(".expert_count") or kl == "expert_count":
            raw = _field_contents(field)
            try:
                if int(raw) > 1:
                    return True
            except (TypeError, ValueError):
                continue

    arch_field = reader.fields.get("general.architecture")
    if arch_field is not None:
        arch = str(_field_contents(arch_field) or "").lower()
        if "moe" in arch:
            return True
    return False


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
