"""Smoke test for the autoresearch_core PyO3 bindings (Phase 0 Sprint 0.4).

Verifies, against the Rust kernel in `autoresearch_core.{pareto,fingerprint}`,
that the zero-ruptura contract holds:

1. Every public PyO3-exposed symbol imports cleanly.
2. `ObjectiveVector` round-trips through its four `#[getter]` axes and the
   `complete` boolean property (not a callable).
3. `Trial(fp, vector)` accepts the canonical 2-arg form.
4. `dominates`, `pareto_set`, `merge`, `fingerprint` produce the same
   answers as the hand-rolled Python reference (canonical JSON via
   `hashlib.sha256`, no dependency on `autoresearch.core.*` shims).
5. A 100-config parity sweep over varying `CTX_SIZE`/`BATCH_SIZE`/
   `KV_CACHE`/`FLASH_ATTN` + `top_k`/`temperature` returns identical
   hex digests.

Run with the venv python (system-global Python breaks the AGENTS.md contract):

    .venv\\Scripts\\python.exe scripts\\smoke_bindings.py
"""

from __future__ import annotations

import hashlib
import json

import autoresearch_core as arc
from autoresearch_core.pareto import (
    ObjectiveVector,
    Trial,
    dominates,
    fingerprint,
    merge,
    pareto_set,
)


def _canonical_payload(engine: dict, sampler: dict) -> str:
    """Match the Rust canonical JSON: dict shape ``{"engine": …, "sampler": …}``
    with sorted keys and compact separators."""
    return json.dumps(
        {"engine": engine, "sampler": sampler},
        sort_keys=True,
        separators=(",", ":"),
    )


def _python_hash(engine: dict, sampler: dict) -> str:
    return hashlib.sha256(_canonical_payload(engine, sampler).encode("utf-8")).hexdigest()


def main() -> int:
    print(f"arc module: {arc.__file__ if hasattr(arc, '__file__') else arc!r}")
    print(f"ObjectiveVector from Rust: {ObjectiveVector}")

    v = ObjectiveVector(2048, 100.0, 0.6, 0.6)
    assert isinstance(v.complete, bool), "`complete` must be a bool property"
    assert v.complete is True, "all 4 axes populated -> complete must be True"
    assert v.ctx == 2048.0 and v.tps == 100.0 and v.agentic == 0.6 and v.coding == 0.6
    print(f"  ObjectiveVector OK (complete={v.complete}, ctx={v.ctx}, tps={v.tps})")

    h = fingerprint({"threads": 4, "ctx": 2048}, {"top_k": 40})
    assert len(h) == 64, "expected 64-char hex"
    expected = _python_hash({"threads": 4, "ctx": 2048}, {"top_k": 40})
    assert h == expected, f"hash mismatch: rust={h} python={expected}"
    print(f"  fingerprint OK ({h[:16]}…)")

    a = ObjectiveVector(2048, 100.0, 0.6, 0.6)
    b = ObjectiveVector(1024, 50.0, 0.3, 0.3)
    assert dominates(a, b)
    assert not dominates(b, a)
    print("  dominates OK")

    big = pareto_set([a, b, ObjectiveVector(0, 0, 0, 0)])
    assert len(big) == 1
    print(f"  pareto_set OK (3 in -> {len(big)} complete)")

    merged = merge([Trial("m", ObjectiveVector(1024, 50.0, 0.5, 0.5))])
    assert len(merged) == 1
    print(f"  merge OK (1 in -> {len(merged)})")

    # 100-config parity sweep.
    swept_ok = 0
    for i in range(100):
        engine = {
            "CTX_SIZE": 1024 + (i % 8) * 512,
            "BATCH_SIZE": 64 + (i % 4) * 64,
            "KV_CACHE": ("q4_0", "q8_0", "f16")[i % 3],
            "FLASH_ATTN": i % 2 == 0,
        }
        sampler = {"top_k": 10 + i, "temperature": 0.1 * (i % 11)}
        if fingerprint(engine, sampler) != _python_hash(engine, sampler):
            raise SystemExit(f"HASH MISMATCH at iteration {i}")
        swept_ok += 1
    print(f"  parity sweep: {swept_ok}/100 hashes identical")

    print("smoke_bindings.py: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
