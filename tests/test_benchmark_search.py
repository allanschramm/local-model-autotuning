import unittest
from unittest.mock import patch, MagicMock
import benchmark_search
from autoresearch.core import config
from autoresearch.benchmarks import bench_config

class TestBenchmarkSearch(unittest.TestCase):

    def test_parse_args_defaults(self):
        with patch("sys.argv", ["benchmark_search.py"]):
            args = benchmark_search.parse_args()
            self.assertEqual(args.model, config.MODEL)
            self.assertEqual(args.ctx_size, config.CTX_SIZE)
            self.assertEqual(args.kv, config.KV_CACHE)
            self.assertEqual(args.threads, config.THREADS)
            self.assertEqual(args.include_coding, bench_config.INCLUDE_CODING)

    def test_parse_args_rejects_baseline_cli_overrides(self):
        with patch("sys.argv", [
            "benchmark_search.py",
            "--model", "Ornith-1.0-35B-UD-Q3_K_XL.gguf",
            "--threads", "8",
            "--threads-batch", "8",
            "--batch-size", "1024",
            "--ubatch-size", "256",
        ]):
            with self.assertRaises(SystemExit):
                benchmark_search.parse_args()

    def test_help_hides_config_only_baseline_flags(self):
        with patch("sys.argv", ["benchmark_search.py", "--help"]):
            with patch("sys.stdout") as stdout:
                with self.assertRaises(SystemExit):
                    benchmark_search.parse_args()
        help_text = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertNotIn("--model", help_text)
        self.assertNotIn("--n-cpu-moe", help_text)
        self.assertNotIn("--ngl", help_text)
        self.assertNotIn("--bench-tts-threshold", help_text)
        self.assertNotIn("--grid", help_text)

    @patch("autoresearch.runners.run.handle_single_run")
    def test_main_execution(self, mock_handle):
        with patch("sys.argv", ["benchmark_search.py"]):
            benchmark_search.args = benchmark_search.parse_args()
            # Simulate direct execution behavior of __main__ block
            from autoresearch.runners import run
            run.handle_single_run(benchmark_search.args)
            mock_handle.assert_called_once_with(benchmark_search.args)

if __name__ == "__main__":
    unittest.main()
