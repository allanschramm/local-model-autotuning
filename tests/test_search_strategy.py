import unittest
from autoresearch.core.search import SearchStrategy

class TestSearchStrategy(unittest.TestCase):

    def test_is_improvement_simple_improvement(self):
        # Score improved
        strategy = SearchStrategy({}, use_pareto_tiebreaker=False)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70, baseline_tps=30.0, baseline_vram=4.0,
            new_score=0.75, new_tps=30.0, new_vram=4.0
        )
        self.assertTrue(is_imp)
        self.assertIn("Score improved", reason)

    def test_is_improvement_no_improvement(self):
        # Score regressed/same without tiebreaker
        strategy = SearchStrategy({}, use_pareto_tiebreaker=False)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70, baseline_tps=30.0, baseline_vram=4.0,
            new_score=0.70, new_tps=50.0, new_vram=2.0
        )
        self.assertFalse(is_imp)

    def test_is_improvement_pareto_tps(self):
        # Score tied, TPS improved (> 5%)
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70, baseline_tps=30.0, baseline_vram=4.0,
            new_score=0.70, new_tps=32.0, new_vram=4.0
        )
        self.assertTrue(is_imp)
        self.assertIn("TPS improved", reason)

    def test_is_improvement_pareto_vram(self):
        # Score tied, TPS same, VRAM improved — but VRAM is no longer a tie-breaker
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70, baseline_tps=30.0, baseline_vram=4.0,
            new_score=0.70, new_tps=29.0, new_vram=3.5
        )
        self.assertFalse(is_imp)  # VRAM no longer used as tie-breaker

    def test_is_improvement_pareto_no_tps_no_vram(self):
        # Score tied, TPS regressed heavily, VRAM improved (not enough for TPS drop)
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70, baseline_tps=30.0, baseline_vram=4.0,
            new_score=0.70, new_tps=20.0, new_vram=3.5
        )
        self.assertFalse(is_imp)

    def test_random_restart(self):
        search_space = {
            "param1": [1, 2],
            "param2": [10]
        }
        strategy = SearchStrategy(search_space)
        current = {"param1": 1, "param2": 10}
        
        # If we visit the current config, it should pick the other option
        visited = {strategy.get_config_key(current)}
        new_cfg = strategy.random_restart(visited, current)
        self.assertIsNotNone(new_cfg)
        self.assertEqual(new_cfg["param1"], 2)
        self.assertEqual(new_cfg["param2"], 10)

        # If all configurations are visited, it should return None
        visited.add(strategy.get_config_key(new_cfg))
        final_cfg = strategy.random_restart(visited, current, max_attempts=50)
        self.assertIsNone(final_cfg)

if __name__ == "__main__":
    unittest.main()
