import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from autoresearch.core.llama_runner import (
    estimate_vram_mb,
    LlamaServerRunner,
    ServerIntent,
    VRAM_QUANT_FACTORS,
    VRAM_DEFAULT_QUANT_FACTOR
)
from autoresearch.runners.run import run_evaluation
from autoresearch.benchmarks.benchmark_harness import BenchmarkResult

class CrashyConfig:
    __slots__ = ()
    @property
    def model(self):
        raise ValueError("Simulated property error")

class BadKey:
    def __str__(self):
        raise RuntimeError("Simulated string conversion error")

class BadDictClass:
    @property
    def __dict__(self):
        raise RuntimeError("Simulated dict access error")

class BadDirClass:
    def __dir__(self):
        raise RuntimeError("Simulated dir listing error")

class TestAdversarialChallenger(unittest.TestCase):

    def test_vram_quant_factors_monotonicity(self):
        """3. Validate VRAM quant factors: turbo4 > turbo3 > turbo2."""
        t4 = VRAM_QUANT_FACTORS.get("turbo4")
        t3 = VRAM_QUANT_FACTORS.get("turbo3")
        t2 = VRAM_QUANT_FACTORS.get("turbo2")
        
        self.assertIsNotNone(t4, "turbo4 factor is missing")
        self.assertIsNotNone(t3, "turbo3 factor is missing")
        self.assertIsNotNone(t2, "turbo2 factor is missing")
        
        # Verify strictly monotonic descending (greater factors mean more memory)
        self.assertGreater(t4, t3, "VRAM factor for turbo4 must be greater than turbo3")
        self.assertGreater(t3, t2, "VRAM factor for turbo3 must be greater than turbo2")
        
        # Print for verification logs
        print(f"[Monotonicity Check] turbo4={t4}, turbo3={t3}, turbo2={t2}")

    def test_vram_factor_resolution(self):
        """Validate resolution of VRAM factors under different cases."""
        # 1. Standard resolution
        v_turbo4 = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="turbo4", kv_cache_v="turbo4")
        v_turbo3 = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="turbo3", kv_cache_v="turbo3")
        v_turbo2 = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="turbo2", kv_cache_v="turbo2")
        
        self.assertGreater(v_turbo4, v_turbo3)
        self.assertGreater(v_turbo3, v_turbo2)
        
        # 2. Mixed case string resolution
        v_mixed = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="TuRbO4", kv_cache_v="tUrBo4")
        self.assertEqual(v_turbo4, v_mixed)
        
        # 3. Fallback resolution for unknown quantization types
        v_unknown = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="unknown-format", kv_cache_v="unknown-format")
        factor_est = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k="nonexistent", kv_cache_v="nonexistent")
        
        model_size_mb = 4000.0
        kv_base_mb = 2048 * 80.0 / 1024.0 # 160.0 MB
        expected_kv = (160.0 / 2.0) * 0.3 + (160.0 / 2.0) * 0.3 # 48.0 MB
        expected_total = model_size_mb + expected_kv + 300.0 # 4348.0 MB
        self.assertAlmostEqual(factor_est, expected_total)

    def test_vram_factor_crashes_on_none_base_kv(self):
        """Check that estimate_vram_mb does not crash if base_kv_cache is None."""
        try:
            val = estimate_vram_mb(Path("non-existent"), 2048, kv_cache_k=None, kv_cache_v=None, base_kv_cache=None)
            crashed = False
        except Exception as e:
            crashed = True
            val = None
            print(f"[Adversarial Finding] estimate_vram_mb crashed on None base_kv_cache: {e}")
        
        self.assertFalse(crashed, "Expected estimate_vram_mb NOT to crash on None base_kv_cache")
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 4344.8)

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_crashy_config_property(self, mock_coding, mock_runner):
        """Check that run_evaluation does not crash when a property raises a non-AttributeError exception."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_coding.return_value = BenchmarkResult(
            val_score=0.5, val_pass1=0.4, val_pass2=0.6, val_pass3=0.5, avg_tps=30.0
        )
        
        cfg = CrashyConfig()
        res = run_evaluation(cfg, skip_bench=True, include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "g4-opt-it-Q4_K_M.gguf")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_dict_non_string_keys(self, mock_coding, mock_runner):
        """Check if run_evaluation handles dicts with non-string keys safely."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        cfg = {
            123: "numeric-key-value",
            None: "none-key-value",
            "model": "model-from-dict.gguf"
        }
        res = run_evaluation(cfg, skip_bench=True, include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "model-from-dict.gguf")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_none_config(self, mock_coding, mock_runner):
        """Check if run_evaluation handles None config safely."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        res = run_evaluation(None, skip_bench=True, model="fallback-override.gguf", include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "fallback-override.gguf")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_list_config(self, mock_coding, mock_runner):
        """Check if run_evaluation handles list config safely."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        res = run_evaluation(["some", "list"], skip_bench=True, model="list-override.gguf", include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "list-override.gguf")

    def test_estimate_vram_mb_ctx_size_none(self):
        """Verify estimate_vram_mb handles ctx_size=None and returns valid float."""
        val = estimate_vram_mb(Path("non-existent"), None)
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 4658.4)

    def test_estimate_vram_mb_ctx_size_string(self):
        """Verify estimate_vram_mb handles string ctx_size and returns valid float."""
        val = estimate_vram_mb(Path("non-existent"), "2048")
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 4344.8)

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_bad_key(self, mock_coding, mock_runner):
        """Verify run_evaluation does not crash on dict with a key that raises on __str__."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        cfg = {BadKey(): "value", "model": "bad-key-model.gguf"}
        res = run_evaluation(cfg, skip_bench=True, include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "bad-key-model.gguf")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_bad_dict_class(self, mock_coding, mock_runner):
        """Verify run_evaluation does not crash on class that raises on __dict__ access."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        cfg = BadDictClass()
        res = run_evaluation(cfg, skip_bench=True, model="bad-dict-override.gguf", include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "bad-dict-override.gguf")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_bad_dir_class(self, mock_coding, mock_runner):
        """Verify run_evaluation does not crash on class that raises on dir() call."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        cfg = BadDirClass()
        res = run_evaluation(cfg, skip_bench=True, model="bad-dir-override.gguf", include_coding=False)
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "bad-dir-override.gguf")

if __name__ == "__main__":
    unittest.main()
