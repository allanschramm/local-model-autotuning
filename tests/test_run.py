import unittest
from unittest.mock import patch, MagicMock, mock_open
from autoresearch.runners import run
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from pathlib import Path
import csv

class TestRun(unittest.TestCase):

    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=45.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_single_run_improved(self, mock_file, mock_commit, mock_coding, mock_runner, mock_bench):
        # Setup mocks
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        
        # Mock coding result with all 4 benchmark fields
        mock_coding.return_value = BenchmarkResult(
            val_score=0.75, val_pass1=0.6, val_pass2=0.8, val_pass3=0.7, val_pass4=0.5, avg_tps=40.0
        )
        
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
            args.include_coding = True
            args.grid = False
            
            with patch("sys.exit") as mock_exit:
                run.handle_single_run(args)
                mock_exit.assert_not_called()
                
        # File should have been opened for appending
        mock_file.assert_called_with(run.RESULTS_FILE, "a", newline="")

    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=45.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_grid_run(self, mock_file, mock_commit, mock_coding, mock_runner, mock_bench):
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"
        mock_coding.return_value = BenchmarkResult(
            val_score=0.75, val_pass1=0.6, val_pass2=0.8, val_pass3=0.7, val_pass4=0.5, avg_tps=40.0
        )
        
        args = MagicMock()
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.ctx_size = 16384
        args.port = 18080
        args.threads = 12
        args.ngl = 99
        args.context_tokens = 8192
        args.include_coding = True
        args.grid = True
        args.grid_kvs = "q4_0"
        args.grid_max_tokens = "512"
        
        run.handle_grid_run(args)
        
        mock_file.assert_called_with(run.RESULTS_FILE, "a", newline="")

    @patch("autoresearch.runners.run.run_evaluation")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_multidimensional_grid_run(self, mock_file, mock_commit, mock_eval):
        mock_commit.return_value = "abcdefg"
        
        args = MagicMock()
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.ctx_size = 16384
        args.port = 18080
        args.threads = 12
        args.threads_batch = 16
        args.ngl = 99
        args.context_tokens = 8192
        args.include_coding = True
        args.grid = True
        args.grid_kvs = None
        args.kv = "q4_0"
        args.grid_kvs_k = "q8_0,f16"
        args.grid_kvs_v = "q4_0"
        args.grid_max_tokens = "512,1024"
        args.grid_threads = "8,12"
        args.grid_threads_batch = "12,16"
        args.grid_batch_sizes = "512"
        args.grid_ubatch_sizes = "128"
        args.grid_spec_draft_n_max = "1,2"
        
        mock_eval.return_value = {
            "status": "OK",
            "coding_val": 0.75, "coding_tps": 40.0,
            "lcb_val": 0.6, "he_val": 0.8, "mbpp_val": 0.7, "bigcode_val": 0.5,
            "swe_val": 0.0,
            "val_score": 0.74, "avg_tps": 35.0, "peak_vram_gb": 4.0
        }
        run.handle_grid_run(args)
        # Combinations: 2 (kvs_k) * 1 (kvs_v) * 2 (max_tokens) * 2 (threads) * 2 (threads_batch) * 1 (batch) * 1 (ubatch) * 2 (spec_draft) = 32
        self.assertEqual(mock_eval.call_count, 32)

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_without_coding(self, mock_coding, mock_runner):
        # Setup mocks
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        args = MagicMock()
        args.kv_k = "q4_0"
        args.kv_v = "q4_0"
        args.threads = 12
        args.threads_batch = None
        args.batch_size = 512
        args.ubatch_size = 128
        args.spec_draft_n_max = 1
        args.spec_type = None
        args.coding_task_limit = 30
        
        res = run.run_evaluation(
            args, skip_bench=True, model="g4-opt-it-Q4_K_M.gguf", kv="q4_0", max_tokens=1024,
            include_coding=False
        )
        
        # Verify coding was NOT called
        mock_coding.assert_not_called()
        
        # Check val_score is 0 when coding disabled
        self.assertEqual(res["coding_val"], 0.0)
    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=42.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_validation_mode(self, mock_coding, mock_runner, mock_bench):
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        args = MagicMock()
        args.kv_k = "q4_0"
        args.kv_v = "q4_0"
        args.threads = 12
        args.threads_batch = None
        args.batch_size = 512
        args.ubatch_size = 128
        args.spec_draft_n_max = 1
        args.spec_type = None
        args.coding_task_limit = 30
        
        # validation=True: runs bench mock, then coding with task_limit=2
        mock_coding.return_value = BenchmarkResult(
            val_score=0.75, val_pass1=0.6, val_pass2=0.8, val_pass3=0.7, val_pass4=0.5, avg_tps=40.0
        )
        res = run.run_evaluation(
            args, model="g4-opt-it-Q4_K_M.gguf", kv="q4_0", max_tokens=1024,
            include_coding=True, validation=True
        )
        
        # Validation mode now runs quick 2-task coding
        mock_coding.assert_called_once()
        self.assertEqual(res["coding_val"], 0.75)
        self.assertEqual(res["bench_tg_tps"], 42.0)

    @patch("autoresearch.runners.run.run_evaluation")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_single_run_validation_passes(self, mock_file, mock_commit, mock_eval):
        mock_commit.return_value = "abcdefg"
        # Validation passes bench check + 2-task coding
        mock_eval.return_value = {
            "status": "OK",
            "coding_val": 0.75,
            "lcb_val": 0.6, "he_val": 0.8, "mbpp_val": 0.7, "bigcode_val": 0.5,
            "swe_val": 0.0,
            "val_score": 0.75, "avg_tps": 42.0, "peak_vram_gb": 6.0,
            "bench_tg_tps": 42.0, "bench_pp_tps": 190.0,
        }
        
        args = MagicMock()
        args.desc = "validation test"
        args.model = "ornith-1.0-9b-Q4_K_M.gguf"
        args.kv = "q4_0"
        args.ctx_size = 131072
        args.validation = True
        
        with patch("sys.exit") as mock_exit:
            run.handle_single_run(args)
            mock_exit.assert_not_called()

    @patch("autoresearch.runners.run.run_evaluation")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.runners.run.open", new_callable=mock_open)
    def test_single_run_validation_fails(self, mock_file, mock_commit, mock_eval):
        mock_commit.return_value = "abcdefg"
        # Bench-validation: FAIL status means val_score=0.0
        mock_eval.return_value = {
            "status": "FAIL: bench tg 15.0 < threshold 30.0",
            "coding_val": 0.0,
            "lcb_val": 0.0, "he_val": 0.0, "mbpp_val": 0.0, "bigcode_val": 0.0,
            "swe_val": 0.0,
            "val_score": 0.0, "avg_tps": 0.0, "peak_vram_gb": 0.0
        }
        
        args = MagicMock()
        args.desc = "validation test"
        args.model = "ornith-1.0-9b-Q4_K_M.gguf"
        args.kv = "q4_0"
        args.ctx_size = 131072
        args.validation = True
        
        with patch("sys.exit") as mock_exit:
            run.handle_single_run(args)
            mock_exit.assert_called_once_with(1)

    @patch("autoresearch.runners.run.open", new_callable=mock_open, read_data="commit\tval_score\tswe_score\tlcb_score\the_score\tmbpp_score\tbigcode_score\tmemory_gb\tstatus\tdescription\n"
              "abcdefg\t0.580000\t0.000000\t0.400000\t0.800000\t0.900000\t0.100000\t7.4\tkeep\tornith-1.0-9b-Q4_K_M.gguf baseline\n"
              "1234567\t0.495000\t0.000000\t0.300000\t0.800000\t0.700000\t0.100000\t7.7\tkeep\tQwen3.5-9B-MTP-Q4_K_M.gguf baseline\n")
    def test_get_previous_best_with_model_filter(self, mock_file):
        with patch.object(Path, "exists", return_value=True):
            # Without model filter, returns global max (0.580000)
            self.assertEqual(run.get_previous_best(Path("dummy.tsv")), 0.58)
            # With specific model filter matching the first row
            self.assertEqual(run.get_previous_best(Path("dummy.tsv"), "ornith-1.0-9b-Q4_K_M.gguf"), 0.58)
            # With specific model filter matching the second row
            self.assertEqual(run.get_previous_best(Path("dummy.tsv"), "Qwen3.5-9B-MTP-Q4_K_M.gguf"), 0.495)
            # With a model that doesn't exist yet, returns 0.0
            self.assertEqual(run.get_previous_best(Path("dummy.tsv"), "ornith-1.0-35b-Q4_K_M.gguf"), 0.0)

if __name__ == "__main__":
    unittest.main()
