import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import autoloop
from autoresearch.core.search import SearchStrategy

class TestAutoLoop(unittest.TestCase):

    def test_get_neighbors(self):
        config = {
            "KV_CACHE_K": "q4_0",
            "KV_CACHE_V": "q4_0",
            "THREADS": 12,
            "THREADS_BATCH": None,
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
            "SPEC_DRAFT_N_MAX": 1,
            "CTX_SIZE": 16384,
            "CONT_BATCHING": False,
            "FLASH_ATTN": "on",
            "NO_MMAP": False,
            "TEMP": 0.2,
        }
        
        search_strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        neighbors = search_strategy.get_neighbors(config)
        self.assertGreater(len(neighbors), 0)
        
        # Verify each neighbor only differs by one parameter
        for neighbor in neighbors:
            diffs = sum(1 for k in config if config[k] != neighbor.config[k])
            self.assertEqual(diffs, 1)

    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok(self, mock_estimate):
        mock_estimate.return_value = 5000.0
        cfg = {"MODEL": "m.gguf", "CTX_SIZE": 4096, "KV_CACHE_K": "q4_0"}
        
        # Under limit
        self.assertTrue(autoloop.preflight_vram_ok(cfg, 6000.0))
        # Over limit
        self.assertFalse(autoloop.preflight_vram_ok(cfg, 4000.0))
        # No limit
        self.assertTrue(autoloop.preflight_vram_ok(cfg, None))

    def test_parse_budget_seconds(self):
        self.assertEqual(autoloop.parse_budget_seconds("10m"), 600.0)
        self.assertEqual(autoloop.parse_budget_seconds("30s"), 30.0)
        self.assertEqual(autoloop.parse_budget_seconds("150"), 150.0)
        self.assertEqual(autoloop.parse_budget_seconds("invalid"), 300.0)

if __name__ == "__main__":
    unittest.main()
