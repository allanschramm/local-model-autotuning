import unittest
from unittest.mock import MagicMock, patch

from autoresearch.core.config import load_config
from autoresearch.core.llama_runner import validate_config
from autoresearch.runners import run


class TestConfigParsing(unittest.TestCase):
    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 1000.0, 8000.0, ""),
    )
    @patch(
        "autoresearch.runners.evaluation.preflight_vram_for_intent", return_value=(True, 1000.0, "")
    )
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_config_normalization_and_fallback(
        self, mock_coding, mock_runner, _mock_preflight, _mock_host
    ):
        # Mock runner context manager
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)

        # 1. Test dictionary with uppercase/lowercase mixed keys and overrides
        cfg_dict = {
            "MODEL": "test-uppercase.gguf",
            "kv": "q4_0",
            "kv_k": None,
            "KV_V": None,
            "THREADS": 4,
        }

        # Override KV through kwargs (overrides dict)
        res = run.run_evaluation(cfg_dict, skip_bench=True, kv="f16", include_coding=False)

        # Retrieve ServerIntent passed to LlamaServerRunner
        intent = mock_runner.call_args[0][0]

        self.assertEqual(intent.model_path.name, "test-uppercase.gguf")
        self.assertEqual(intent.kv_cache, "f16")  # overridden
        self.assertEqual(intent.kv_cache_k, "f16")  # fell back to kv because kv_k was None
        self.assertEqual(intent.kv_cache_v, "f16")  # fell back to kv because kv_v was None
        self.assertEqual(intent.threads, 4)


class TestRuntimeInvariants(unittest.TestCase):
    def test_validate_config_accepts_custom_tps_floor(self):
        cfg = load_config()
        cfg["N_CPU_MOE"] = None
        cfg["TPS_FLOOR"] = 15.0
        out = validate_config(cfg)
        self.assertEqual(out["TPS_FLOOR"], 15.0)


if __name__ == "__main__":
    unittest.main()
