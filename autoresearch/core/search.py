import random
from typing import Dict, Any, List, Set, Tuple

Config = Dict[str, Any]

class SearchStrategy:
    """
    Deep module encapsulating the Hill-Climbing Search logic.
    Provides leverage by standardising Neighbor generation, the Pareto Tie-Breaker, 
    and Random Restarts across different search spaces.
    """
    def __init__(self, search_space: Dict[str, List[Any]], use_pareto_tiebreaker: bool = False):
        self.search_space = search_space
        self.use_pareto_tiebreaker = use_pareto_tiebreaker

    def get_config_key(self, cfg: Config) -> str:
        """Deterministically serialize the search-space parameters of a config."""
        return str(sorted((k, v) for k, v in cfg.items() if k in self.search_space))

    def get_neighbors(self, current_cfg: Config) -> List[Config]:
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
                        n["_changed"] = param
                        n["_old"] = current
                        n["_new"] = val
                        neighbors.append(n)
                continue

            # Adjacent neighbors in the ordered list
            if idx > 0:
                n = current_cfg.copy()
                n[param] = candidates[idx - 1]
                n["_changed"] = param
                n["_old"] = current
                n["_new"] = candidates[idx - 1]
                neighbors.append(n)
            if idx < len(candidates) - 1:
                n = current_cfg.copy()
                n[param] = candidates[idx + 1]
                n["_changed"] = param
                n["_old"] = current
                n["_new"] = candidates[idx + 1]
                neighbors.append(n)

        random.shuffle(neighbors)
        return neighbors

    def random_restart(self, visited: Set[str], current_cfg: Config, max_attempts: int = 100) -> Config | None:
        """Generate a random valid configuration not in visited to escape local maxima."""
        for _ in range(max_attempts):
            new_cfg = current_cfg.copy()
            for param, values in self.search_space.items():
                new_cfg[param] = random.choice(values)
            
            n_key = self.get_config_key(new_cfg)
            if n_key not in visited:
                return new_cfg
        return None

    def is_improvement(
        self,
        baseline_score: float, baseline_tps: float, baseline_vram: float,
        new_score: float, new_tps: float, new_vram: float
    ) -> Tuple[bool, str]:
        """
        Evaluate if the new trial beats the baseline.
        Returns (is_improvement, reason_string).
        """
        delta = new_score - baseline_score
        
        if new_score > baseline_score + 0.0001:
            return True, f"Score improved (Δ={delta:+.6f})"
            
        if self.use_pareto_tiebreaker and abs(new_score - baseline_score) <= 0.0001:
            if new_tps > baseline_tps * 1.05:
                return True, f"Score tied, TPS improved (+{new_tps - baseline_tps:.1f})"
            elif new_tps >= baseline_tps * 0.95 and new_vram < baseline_vram * 0.95:
                return True, f"Score/TPS tied, VRAM improved (-{baseline_vram - new_vram:.1f}GB)"
                
        return False, ""

    def format_config_summary(self, cfg: Config) -> str:
        """One-line summary of tunable params for logging."""
        parts = []
        for p in self.search_space:
            v = cfg.get(p)
            if v is not None:
                parts.append(f"{p}={v}")
        return " ".join(parts)
