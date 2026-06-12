import unittest
from unittest.mock import patch, MagicMock
from autoresearch.benchmarks import benchmark_coding
from pathlib import Path
import sys


class TestBenchmarkCoding(unittest.TestCase):

    def test_parse_args(self):
        test_args = ["prog", "--ctx-size", "2048", "--model", "test-model.gguf"]
        with patch.object(sys, 'argv', test_args):
            args = benchmark_coding.parse_args()
            self.assertEqual(args.ctx_size, 2048)
            self.assertEqual(args.model, "test-model.gguf")

    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_run_benchmark(self, mock_run_coding_eval):
        # Mock returns for HE and MBPP: (pass_at_1, total_tokens, total_seconds)
        mock_run_coding_eval.side_effect = [
            (0.6, 100, 10.0),  # HumanEval
            (0.4, 200, 5.0)    # MBPP
        ]

        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        result = benchmark_coding.run_benchmark(client, task_limit=10)

        self.assertEqual(result.val_pass1, 0.6)
        self.assertEqual(result.val_pass2, 0.4)
        self.assertEqual(result.val_score, 0.5)
        self.assertEqual(result.total_seconds, 15.0)
        self.assertEqual(result.avg_tps, 20.0)

    @patch("autoresearch.benchmarks.benchmark_coding.run_coding_eval")
    def test_run_benchmark_passes_gen_kwargs(self, mock_run_coding_eval):
        """Verify gen_kwargs are forwarded to run_coding_eval."""
        mock_run_coding_eval.return_value = (0.5, 50, 5.0)

        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.port = 1234

        benchmark_coding.run_benchmark(
            client, task_limit=5,
            temperature=0.6, top_p=0.95, top_k=20
        )

        # Check gen_kwargs were passed to run_coding_eval
        for call in mock_run_coding_eval.call_args_list:
            _, kwargs = call
            self.assertEqual(kwargs.get("temperature"), 0.6)
            self.assertEqual(kwargs.get("top_p"), 0.95)
            self.assertEqual(kwargs.get("top_k"), 20)

    @patch("autoresearch.benchmarks.benchmark_coding._load_problems")
    @patch("autoresearch.benchmarks.benchmark_coding._run_tests")
    def test_run_coding_eval_pass_at_1(self, mock_run_tests, mock_load):
        """Verify pass@1 calculation."""
        # Mock 4 problems, 2 pass
        mock_load.return_value = {
            f"HE/{i}": {
                "prompt": f"def f{i}(x):",
                "test": f"assert f{i}(1) == 1",
                "entry_point": f"f{i}"
            }
            for i in range(4)
        }
        mock_run_tests.side_effect = [True, False, True, False]

        from autoresearch.core.llama_client import LlamaClient
        client = MagicMock(spec=LlamaClient)
        client.complete.return_value = {
            "content": "def f(x): return x",
            "usage": {"total_tokens": 10}
        }

        pass_rate, tokens, elapsed = benchmark_coding.run_coding_eval(
            client, "humaneval", task_limit=4
        )
        self.assertEqual(pass_rate, 0.5)  # 2/4
        self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
