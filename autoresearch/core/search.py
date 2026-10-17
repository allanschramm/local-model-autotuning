"""Hill-Climbing Search Strategy over the per-model Pareto Set (ADR 0006).

Neighbor generation and Random Restarts live here; Trial acceptance no longer
uses scalar Val Score — a Neighbor improves when it joins or improves the
per-model Pareto Set (issue #7), and autoloop drives the Set (issue #8). The
legacy scalar keep rules stay as a compat shim for incomplete-vector modes
(engine-only / quality-only, which measure no agentic/coding axes).
"""

import itertools
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from autoresearch.core.pareto import ObjectiveVector, dominates, pareto_set

Config = dict[str, Any]


@dataclass
class Neighbor:
    config: Config
    changed: str
    old: Any
    new: Any


class SearchStrategy:
    """
    Deep module encapsulating the Hill-Climbing Search logic.
    Provides leverage by standardising Neighbor generation, per-model Pareto Set
    updates, and Random Restarts across different search spaces.
    """

    def __init__(
        self,
        search_space: dict[str, list[Any]],
        *,
        known: Iterable[ObjectiveVector] = (),
        use_pareto_tiebreaker: bool = False,
    ):
        self.search_space = search_space
        # Legacy flag consumed only by the scalar keep fallback for
        # incomplete-vector modes (autoloop --mode tps|quality; issue #8).
        self.use_pareto_tiebreaker = use_pareto_tiebreaker
        # Per-model Pareto Set state: every Trial outcome recorded, merged per
        # Fingerprint by the caller. The front is derived and cached.
        self.known = list(known)
        self._pareto_cache: list[ObjectiveVector] | None = None
        self._pareto_cache_key: tuple[ObjectiveVector, ...] | None = None

    def get_config_key(self, cfg: Config) -> str:
        """Deterministically serialize the search-space parameters of a config."""
        return str(sorted((k, v) for k, v in cfg.items() if k in self.search_space or k == "MODEL"))

    @staticmethod
    def is_batch_consistent(cfg: Config) -> bool:
        """Reject neighbors where UBATCH_SIZE > BATCH_SIZE."""
        batch = cfg.get("BATCH_SIZE", cfg.get("batch_size"))
        ubatch = cfg.get("UBATCH_SIZE", cfg.get("ubatch_size"))
        if batch is None or ubatch is None:
            return True
        try:
            return int(ubatch) <= int(batch)
        except (TypeError, ValueError):
            return True

    def get_neighbors(self, current_cfg: Config) -> list[Neighbor]:
        """Generate single-parameter perturbations of current config."""
        neighbors = []
        for param, candidates in self.search_space.items():
            current = current_cfg.get(param)
            try:
                idx = candidates.index(current)
            except ValueError:
                # Current value not in search space; try all candidates
                for val in candidates:
                    if val != current:
                        n = current_cfg.copy()
                        n[param] = val
                        if self.is_batch_consistent(n):
                            neighbors.append(
                                Neighbor(config=n, changed=param, old=current, new=val)
                            )
                continue

            # Adjacent neighbors in the ordered list
            if idx > 0:
                n = current_cfg.copy()
                n[param] = candidates[idx - 1]
                if self.is_batch_consistent(n):
                    neighbors.append(
                        Neighbor(config=n, changed=param, old=current, new=candidates[idx - 1])
                    )
            if idx < len(candidates) - 1:
                n = current_cfg.copy()
                n[param] = candidates[idx + 1]
                if self.is_batch_consistent(n):
                    neighbors.append(
                        Neighbor(config=n, changed=param, old=current, new=candidates[idx + 1])
                    )

        random.shuffle(neighbors)
        return neighbors

    def random_restart(
        self, visited: set[str], current_cfg: Config, max_attempts: int = 100
    ) -> Config | None:
        """Generate a random valid configuration not in visited to escape local maxima."""
        for _ in range(max_attempts):
            new_cfg = current_cfg.copy()
            for param, values in self.search_space.items():
                new_cfg[param] = random.choice(values)
            if not self.is_batch_consistent(new_cfg):
                continue
            n_key = self.get_config_key(new_cfg)
            if n_key not in visited:
                return new_cfg

        # Random sampling can miss a valid point in a sparse space. Probe a
        # bounded deterministic prefix, avoiding an explosive full Cartesian
        # scan when the configured search space is large.
        params = list(self.search_space)
        values = [self.search_space[param] for param in params]
        for choices in itertools.islice(itertools.product(*values), max_attempts * 10):
            new_cfg = current_cfg.copy()
            new_cfg.update(zip(params, choices, strict=True))
            if self.is_batch_consistent(new_cfg) and self.get_config_key(new_cfg) not in visited:
                return new_cfg
        return None

    @property
    def pareto_set(self) -> list[ObjectiveVector]:
        """Current per-model front, cached until known vectors change."""
        cache_key = tuple(self.known)
        if self._pareto_cache is None or self._pareto_cache_key != cache_key:
            self._pareto_cache = pareto_set(self.known)
            self._pareto_cache_key = cache_key
        # Do not expose the mutable cache to callers.
        return list(self._pareto_cache)

    def record(self, vector: ObjectiveVector) -> None:
        """Record a Trial outcome into the per-model Set.

        Every Trial is kept (results-store semantics); statuses derive from the
        front instead of being decided at write time.
        """
        self.known.append(vector)

    def improves_set(self, vector: ObjectiveVector) -> bool:
        """Neighbor acceptance: joins or improves the per-model Pareto Set.

        Replaces the scalar Val Score / Pareto Tie-Breaker keep rules: a
        Neighbor improves the Set iff its Objective Vector is non-dominated by
        the current front. Joining (incomparable) counts, and so does improving
        (dominating older members shrinks the front). Incomplete vectors never
        compete (ADR 0006: incomplete merges, does not dominate). Caller merges
        repeated measurements per Fingerprint before recording, so an exact
        duplicate never reaches the Set (equal vectors would judge each other as
        joining, not dominating).
        """
        return vector.complete and not any(dominates(other, vector) for other in self.pareto_set)

    def is_local_maximum(self, candidates: Iterable[ObjectiveVector]) -> bool:
        """Local Maxima = no Neighbor joins or improves the per-model Set."""
        return not any(self.improves_set(candidate) for candidate in candidates)

    def is_improvement(
        self,
        baseline_score: float,
        baseline_tps: float,
        baseline_vram: float,
        new_score: float,
        new_tps: float,
        new_vram: float,
    ) -> tuple[bool, str]:
        """
        LEGACY scalar keep rules — kept for incomplete-vector modes (autoloop
        `--mode tps|quality`, which measure no agentic/coding axes and so can
        never join the four-axis front; ADR 0006). The Search keep truth is
        `improves_set` / the per-model Set (issue #7, wired by issue #8).
        Rules (legacy matrix):
          Score+  Speed+  → KEEP
          Score+  Speed-  → KEEP
          Score+  Speed=  → KEEP
          Score-  Speed+  → DISCARD
          Score-  Speed-  → DISCARD
          Score=  Speed+  → KEEP
          Score=  Speed=  → DISCARD
        Returns (is_improvement, reason_string).
        """
        delta = new_score - baseline_score

        # Score improved → always KEEP
        if new_score > baseline_score + 0.0001:
            return True, f"Score improved (Δ={delta:+.6f})"

        # Score tied → KEEP only if speed improved
        if self.use_pareto_tiebreaker and abs(new_score - baseline_score) <= 0.0001:
            if new_tps > baseline_tps * 1.05:
                return True, f"Score tied, TPS improved (+{new_tps - baseline_tps:.1f})"
            tps_tied = abs(new_tps - baseline_tps) <= abs(baseline_tps) * 0.05
            if tps_tied and baseline_vram > 0 and new_vram < baseline_vram * 0.95:
                return (
                    True,
                    f"Score and TPS tied, VRAM improved (-{baseline_vram - new_vram:.2f}GB)",
                )

        return False, ""

    def format_config_summary(self, cfg: Config) -> str:
        """One-line summary of tunable params for logging."""
        parts = []
        for p in self.search_space:
            v = cfg.get(p)
            if v is not None:
                parts.append(f"{p}={v}")
        return " ".join(parts)
