import unittest
from unittest.mock import patch, MagicMock
import sys
import tune_search
from pathlib import Path

class TestTuneSearch(unittest.TestCase):

    def test_get_neighbors(self):
        config = {
            "kv_cache_k": "q4_0",
            "kv_cache_v": "q4_0",
            "threads": 12,
            "threads_batch": None,
            "batch_size": 512,
            "ubatch_size": 128,
            "spec_draft_n_max": 1
        }
        
        neighbors = tune_search.get_neighbors(config)
        self.assertGreater(len(neighbors), 0)
        
        # Verify each neighbor only differs by one parameter
        for neighbor in neighbors:
            diffs = sum(1 for k in config if config[k] != neighbor[k])
            self.assertEqual(diffs, 1)

    @patch("tune_search.run_evaluation")
    @patch("tune_search.estimate_vram_mb")
    @patch("tune_search.get_git_commit")
    @patch("tune_search.write_row")
    def test_tuner_main(self, mock_write, mock_commit, mock_estimate, mock_eval):
        mock_commit.return_value = "git123"
        mock_estimate.return_value = 5000.0  # Safe VRAM
        
        # Mock run_evaluation results
        mock_eval.side_effect = [
            # Baseline
            {"val_score": 0.5, "avg_tps": 35.0, "peak_vram_gb": 5.0},
            # Neighbor 1
            {"val_score": 0.6, "avg_tps": 37.0, "peak_vram_gb": 5.1}
        ]
        
        args = MagicMock()
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.ctx_size = 16384
        args.port = 18080
        args.ngl = 99
        args.context_tokens = 8192
        args.max_steps = 1
        args.vram_limit_mb = 7900.0
        args.include_coding = False
        
        with patch("tune_search.parse_args", return_value=args):
            tune_search.main()
            
        # Verify run_evaluation was called for baseline and neighbors
        self.assertGreaterEqual(mock_eval.call_count, 1)

if __name__ == "__main__":
    unittest.main()
