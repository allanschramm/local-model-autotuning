import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.core.llama_runner import ServerIntent
from autoresearch.core.sglang_runner import SGLangServerRunner, run_sglang_bench_validation
from autoresearch.runners.evaluation import ExperimentRunner


class TestSGLangRunner(unittest.TestCase):
    def test_build_cmd_adds_quantization_flags(self):
        intent = ServerIntent(
            model_path=Path("models/sglang/Qwen-GPTQ-Int4"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
        )
        runner = SGLangServerRunner(intent)

        cmd = runner._build_cmd(18080)

        self.assertIn("sglang.launch_server", cmd)
        self.assertIn("--context-length", cmd)
        self.assertEqual(cmd[cmd.index("--context-length") + 1], "131072")
        self.assertIn("--quantization", cmd)
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "gptq_marlin")

    @patch("subprocess.run")
    def test_sglang_bench_failure_closed_for_large_model_without_vram(self, mock_run):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(RuntimeError) as ctx:
                run_sglang_bench_validation(Path("models/sglang/Qwen-35B-GPTQ"), 1, 512, 128)

        self.assertIn("Refusing bench/server validation", str(ctx.exception))
        mock_run.assert_called_once()

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.SGLangServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_directory_model_uses_sglang_runner(self, mock_coding, mock_sglang, mock_llama):
        mock_sglang.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4096)
        mock_coding.return_value = BenchmarkResult(
            val_score=0.5,
            val_pass1=0.4,
            val_pass2=0.6,
            val_pass3=0.5,
            val_pass4=0.3,
            avg_tps=30.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "sglang-model").mkdir()
            res = ExperimentRunner(models_dir).run_trial(
                {"model": "sglang-model", "include_coding": True},
                skip_bench=True,
            )

        self.assertEqual(res.status, "OK")
        self.assertEqual(res.val_score, 0.5)
        mock_sglang.assert_called_once()
        mock_llama.assert_not_called()

    @patch("autoresearch.runners.evaluation.run_sglang_bench_validation", return_value=10.0)
    @patch("autoresearch.runners.evaluation.SGLangServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_sglang_bench_below_threshold_fails_before_server(self, mock_coding, mock_sglang, _mock_bench):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "sglang-model").mkdir()
            res = ExperimentRunner(models_dir).run_trial(
                {"model": "sglang-model", "include_coding": True, "bench_tts_threshold": 20.0},
            )

        self.assertIn("FAIL: sglang bench tg 10.0 < threshold 20.0", res.status)
        mock_sglang.assert_not_called()
        mock_coding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
