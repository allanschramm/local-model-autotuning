import unittest
from unittest.mock import patch, MagicMock, mock_open
import run
from pathlib import Path
import csv
import io

class TestRun(unittest.TestCase):

    @patch("run.LlamaServerRunner")
    @patch("run.run_nexus")
    @patch("run.run_claw")
    @patch("run.run_coding")
    @patch("run.get_git_commit")
    @patch("run.open", new_callable=mock_open)
    def test_single_run_improved(self, mock_file, mock_commit, mock_coding, mock_claw, mock_nexus, mock_runner):
        # Setup mocks
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        
        # Mock scores
        mock_nexus.return_value = MagicMock(val_score=0.8, avg_tps=40.0)
        mock_claw.return_value = MagicMock(val_score=0.7, avg_tps=30.0)
        
        # Mock get_previous_best to return 0.5 (so we improve)
        with patch("run.get_previous_best", return_value=0.5):
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

    @patch("run.LlamaServerRunner")
    @patch("run.run_nexus")
    @patch("run.run_claw")
    @patch("run.get_git_commit")
    @patch("run.open", new_callable=mock_open)
    def test_grid_run(self, mock_file, mock_commit, mock_claw, mock_nexus, mock_runner):
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        mock_nexus.return_value = MagicMock(val_score=0.8, avg_tps=40.0)
        mock_claw.return_value = MagicMock(val_score=0.7, avg_tps=30.0)
        
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

if __name__ == "__main__":
    unittest.main()
