"""Finite tuning Search Space: stable keys and single-parameter Neighbors."""

import random
from dataclasses import dataclass
from typing import Any

Config = dict[str, Any]


@dataclass
class Neighbor:
    config: Config
    changed: str
    old: Any
    new: Any


class SearchStrategy:
    """Generate adjacent engine configurations without owning scores or runtime."""

    def __init__(self, search_space: dict[str, list[Any]]):
        self.search_space = search_space

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
