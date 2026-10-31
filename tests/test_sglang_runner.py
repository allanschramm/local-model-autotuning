import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.core.sglang_runner import run_sglang_bench_validation


class TestSGLangRunner(unittest.TestCase):
    @patch("subprocess.run")
    def test_sglang_bench_failure_closed_for_large_model_without_vram(self, mock_run):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(RuntimeError) as ctx:
                run_sglang_bench_validation(Path("models/sglang/Qwen-35B-GPTQ"), 1, 512, 128)

        self.assertIn("Refusing bench/server validation", str(ctx.exception))
        mock_run.assert_called_once()

    @patch("autoresearch.core.sglang_runner._sglang_env", return_value={})
    @patch("subprocess.run")
    def test_bench_parses_median_throughput_format(self, mock_run, _mock_env):
        """SGLang 0.5.17 prints 'Decode. median latency: ..., median throughput: N token/s'
        instead of the legacy 'Decode token/s:' (issue #59)."""
        fake = MagicMock()
        fake.stdout = (
            "Decode 0. Batch size: 1, latency: 0.01804 s, throughput: 55.42 token/s\n"
            "Decode.  median latency: 0.01738 s, median throughput: 57.55 token/s\n"
            "Total. latency: 9.003 s, throughput: 113.74 token/s\n"
        )
        fake.stderr = ""
        mock_run.return_value = fake

        tps = run_sglang_bench_validation(Path("models/sglang/Qwen3.8-2B"), 1, 512, 512)

        self.assertAlmostEqual(tps, 57.55, places=2)


if __name__ == "__main__":
    unittest.main()
