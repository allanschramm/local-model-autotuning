import unittest
from unittest.mock import patch, MagicMock, mock_open
from autoresearch.runners import run
from pathlib import Path
import csv

class TestRun(unittest.TestCase):

    @patch("autoresearch.runners.run.LlamaServerRunner")
    @patch("autoresearch.runners.run.run_nexus")
    @patch("autoresearch.runners.run.run_claw")
    @patch("autoresearch.runners.run.run_coding")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_single_run_improved(self, mock_file, mock_commit, _mock_coding, mock_claw, mock_nexus, mock_runner):
        # Setup mocks
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        
        # Mock scores
        mock_nexus.return_value = MagicMock(val_score=0.8, avg_tps=40.0)
        mock_claw.return_value = MagicMock(val_score=0.7, avg_tps=30.0)
        _mock_coding.return_value = MagicMock(val_score=0.75)
        
        # Mock get_previous_best to return 0.5 (so we improve)
        with patch("autoresearch.runners.run.get_previous_best", return_value=0.5):
            args = MagicMock()
            args.desc = "Tweak test prompt"
            args.model = "g4-opt-it-Q4_K_M.gguf"
            args.kv = "q4_0"
            args.max_tokens = 512
            args.ctx_size = 16384
            args.port = 18080
            args.threads = 12
            args.ngl = 99
            args.context_tokens = 8192
            args.include_coding = False
            args.grid = False
            
            with patch("sys.exit") as mock_exit:
                run.handle_single_run(args)
                mock_exit.assert_not_called()
                
        # File should have been opened for appending
        mock_file.assert_called_with(run.RESULTS_FILE, "a", newline="")

    @patch("autoresearch.runners.run.LlamaServerRunner")
    @patch("autoresearch.runners.run.run_nexus")
    @patch("autoresearch.runners.run.run_claw")
    @patch("autoresearch.runners.run.run_coding")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_grid_run(self, mock_file, mock_commit, mock_coding, mock_claw, mock_nexus, mock_runner):
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        mock_nexus.return_value = MagicMock(val_score=0.8, avg_tps=40.0)
        mock_claw.return_value = MagicMock(val_score=0.7, avg_tps=30.0)
        mock_coding.return_value = MagicMock(val_score=0.75)
        
        args = MagicMock()
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.ctx_size = 16384
        args.port = 18080
        args.threads = 12
        args.ngl = 99
        args.context_tokens = 8192
        args.include_coding = False
        args.grid = True
        args.grid_kvs = "q4_0"
        args.grid_max_tokens = "512"
        
        run.handle_grid_run(args)
        
        mock_file.assert_called_with(run.RESULTS_FILE, "a", newline="")

    @patch("autoresearch.runners.run.LlamaServerRunner")
    @patch("autoresearch.runners.run.run_nexus")
    @patch("autoresearch.runners.run.run_claw")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_multidimensional_grid_run(self, mock_file, mock_commit, mock_claw, mock_nexus, mock_runner):
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        mock_nexus.return_value = MagicMock(val_score=0.8, avg_tps=40.0)
        mock_claw.return_value = MagicMock(val_score=0.7, avg_tps=30.0)
        
        args = MagicMock()
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.ctx_size = 16384
        args.port = 18080
        args.threads = 12
        args.threads_batch = 16
        args.ngl = 99
        args.context_tokens = 8192
        args.include_coding = False
        args.grid = True
        args.grid_kvs_k = "q8_0,f16"
        args.grid_kvs_v = "q4_0"
        args.grid_max_tokens = "512,1024"
        args.grid_threads = "8,12"
        args.grid_threads_batch = "12,16"
        args.grid_batch_sizes = "512"
        args.grid_ubatch_sizes = "128"
        args.grid_spec_draft_n_max = "1,2"
        
        with patch("autoresearch.runners.run.run_evaluation") as mock_eval:
            mock_eval.return_value = {
                "status": "OK",
                "nexus_val": 0.8, "nexus_tps": 40.0,
                "claw_val": 0.7, "claw_tps": 30.0,
                "val_score": 0.74, "avg_tps": 35.0, "peak_vram_gb": 4.0
            }
            run.handle_grid_run(args)
            # Combinations: 2 (kvs_k) * 1 (kvs_v) * 2 (max_tokens) * 2 (threads) * 2 (threads_batch) * 1 (batch) * 1 (ubatch) * 2 (spec_draft) = 32
            self.assertEqual(mock_eval.call_count, 32)

if __name__ == "__main__":
    unittest.main()
