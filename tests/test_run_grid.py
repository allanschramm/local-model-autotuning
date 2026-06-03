import unittest
from unittest.mock import patch, MagicMock, mock_open
import run_grid
from pathlib import Path
import csv
import io

class TestRunGrid(unittest.TestCase):

    @patch("run_grid.LlamaServerRunner")
    @patch("run_grid.BenchmarkHarness")
    @patch("run_grid.run_coding")
    @patch("run_grid.prepare_eval_data")
    @patch("run_grid.discover_claw_tasks")
    @patch("run_grid.build_context_padding")
    @patch("run_grid.open", new_callable=mock_open)
    def test_main_loop(self, mock_file, mock_padding, mock_claw_data, mock_eval_data, mock_coding, mock_harness, mock_runner):
        # Setup mocks
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        mock_res = MagicMock()
        mock_res.val_score = 0.8
        mock_res.val_pass1 = 0.9
        mock_res.val_pass2 = 0.7
        mock_res.avg_tps = 35.0
        
        harness_instance = mock_harness.return_value
        harness_instance.evaluate.return_value = mock_res
        
        mock_coding.return_value = MagicMock(val_pass1=0.85, val_pass2=0.75)
        
        # Limit grid for test speed
        with patch("run_grid.KV_CACHES", ["q4_0"]):
            with patch("run_grid.MAX_TOKENS_LIST", [512]):
                run_grid.main()
        
        # Verify CSV writing
        # First call is to write header (if not exists), subsequent are appends
        self.assertGreaterEqual(mock_file.call_count, 1)
        
        # Check if we correctly called evaluate for Nexus and Claw
        self.assertEqual(harness_instance.evaluate.call_count, 2)
        
        # Check if we called run_coding
        self.assertEqual(mock_coding.call_count, 1)

if __name__ == "__main__":
    unittest.main()
