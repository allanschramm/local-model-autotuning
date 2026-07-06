from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class BenchmarkResult:
    val_score: float
    val_pass1: float
    val_pass2: float
    val_pass3: float = 0.0
    val_pass4: float = 0.0
    avg_tps: float = 0.0
    total_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)