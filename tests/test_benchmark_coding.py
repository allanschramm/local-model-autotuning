import unittest
from unittest.mock import patch, MagicMock, mock_open
from autoresearch.benchmarks import benchmark_coding
from pathlib import Path
import sys

class TestBenchmarkCoding(unittest.TestCase):

    def test_parse_args(self):
        # Save original state
        orig_ctx = benchmark_coding.CTX_SIZE
        orig_model = benchmark_coding.MODELS_TO_BENCHMARK
        
        test_args = ["prog", "--ctx-size", "2048", "--model", "test-model.gguf"]
        with patch.object(sys, 'argv', test_args):
            benchmark_coding.parse_args()
            self.assertEqual(benchmark_coding.CTX_SIZE, 2048)
            self.assertEqual(benchmark_coding.MODELS_TO_BENCHMARK, ["test-model.gguf"])
        
        # Restore
        benchmark_coding.CTX_SIZE = orig_ctx
        benchmark_coding.MODELS_TO_BENCHMARK = orig_model

    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.mkdir")
    def test_run_evalplus_success(self, _mock_mkdir, mock_glob, mock_run):
        # Mocking Step 1 (codegen) and Step 2 (eval)
        mock_run.side_effect = [
            MagicMock(returncode=0), # Codegen
            MagicMock(stdout="Humaneval (plus) pass@1: 0.75\nHumaneval (base) pass@1: 0.80", returncode=0) # Eval
        ]
        
        # Mock glob to find the jsonl
        mock_file = MagicMock(spec=Path)
        mock_file.name = "samples.jsonl"
        mock_glob.return_value = [mock_file]
        
        output_dir = Path("/tmp/results")
        scores = benchmark_coding.run_evalplus("humaneval", 1234, output_dir, "model.gguf")
        
        self.assertEqual(scores.get("pass1_plus"), 0.75)
        self.assertEqual(scores.get("pass1_base"), 0.80)
        self.assertEqual(mock_run.call_count, 2)

    @patch("autoresearch.benchmarks.benchmark_coding.run_evalplus")
    @patch("pathlib.Path.mkdir")
    def test_run_benchmark(self, _mock_mkdir, mock_run_evalplus):
        # Mock returns for HE and MBPP
        mock_run_evalplus.side_effect = [
            {"pass1_plus": 0.6}, # HumanEval
            {"pass1_plus": 0.4}  # MBPP
        ]
        
        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234
        
        result = benchmark_coding.run_benchmark(client, model_name="test-model")
        
        self.assertEqual(result.val_pass1, 0.6)
        self.assertEqual(result.val_pass2, 0.4)
        self.assertEqual(result.val_score, 0.5)

if __name__ == "__main__":
    unittest.main()