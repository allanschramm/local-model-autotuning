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
