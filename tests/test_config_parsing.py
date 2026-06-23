import unittest
from unittest.mock import patch, MagicMock
from autoresearch.runners import run

class TestConfigParsing(unittest.TestCase):

    @patch("autoresearch.runners.run.LlamaServerRunner")
    @patch("autoresearch.runners.run.run_coding")
    def test_run_evaluation_config_normalization_and_fallback(self, mock_coding, mock_runner):
        # Mock runner context manager
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        
        # 1. Test dictionary with uppercase/lowercase mixed keys and overrides
        cfg_dict = {
            "MODEL": "test-uppercase.gguf",
            "kv": "q4_0",
            "kv_k": None,
            "KV_V": None,
            "THREADS": 4
        }
        
        # Override KV through kwargs (overrides dict)
        res = run.run_evaluation(cfg_dict, kv="f16", include_coding=False)
        
        # Retrieve ServerIntent passed to LlamaServerRunner
        intent = mock_runner.call_args[0][0]
        
        self.assertEqual(intent.model_path.name, "test-uppercase.gguf")
        self.assertEqual(intent.kv_cache, "f16") # overridden
        self.assertEqual(intent.kv_cache_k, "f16") # fell back to kv because kv_k was None
        self.assertEqual(intent.kv_cache_v, "f16") # fell back to kv because kv_v was None
        self.assertEqual(intent.threads, 4)

    @patch("autoresearch.runners.run.LlamaServerRunner")
    @patch("autoresearch.runners.run.run_coding")
    def test_run_evaluation_object_config_normalization(self, mock_coding, mock_runner):
        # Mock runner
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)

        # Config as a custom object
        class CustomConfig:
            def __init__(self):
                self.MODEL = "obj-model.gguf"
                self.KV = "q8_0"
                self.threads = 8

        cfg_obj = CustomConfig()
        
        res = run.run_evaluation(cfg_obj, include_coding=False)
        
        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "obj-model.gguf")
        self.assertEqual(intent.kv_cache, "q8_0")
        self.assertEqual(intent.threads, 8)

if __name__ == "__main__":
    unittest.main()
