import importlib.util
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.core.config import load_config
from autoresearch.core.llama_runner import ConfigError, validate_config
from autoresearch.core.state import SearchState
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

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 1000.0, 8000.0, ""),
    )
    @patch(
        "autoresearch.runners.evaluation.preflight_vram_for_intent", return_value=(True, 1000.0, "")
    )
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_run_evaluation_object_config_normalization(
        self, mock_coding, mock_runner, _mock_preflight, _mock_host
    ):
        # Mock runner
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)

        # Config as a custom object
        class CustomConfig:
            def __init__(self):
                self.MODEL = "obj-model.gguf"
                self.KV = "q8_0"
                self.threads = 8

        cfg_obj = CustomConfig()

        res = run.run_evaluation(cfg_obj, skip_bench=True, include_coding=False)

        intent = mock_runner.call_args[0][0]
        self.assertEqual(intent.model_path.name, "obj-model.gguf")
        self.assertEqual(intent.kv_cache, "q8_0")
        self.assertEqual(intent.threads, 8)


class TestRuntimeInvariants(unittest.TestCase):
    def test_config_segregation(self):
        from autoresearch.core.config import DEFAULTS, ENGINE_DEFAULTS, SAMPLER_DEFAULTS

        self.assertIn("THREADS", ENGINE_DEFAULTS)
        self.assertIn("TEMP", SAMPLER_DEFAULTS)
        self.assertEqual(len(DEFAULTS), len(ENGINE_DEFAULTS) + len(SAMPLER_DEFAULTS))

    def test_rejects_context_below_minimum(self):
        cfg = load_config()
        cfg["CTX_SIZE"] = 1024
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_rejects_flash_attention_off(self):
        cfg = load_config()
        cfg["FLASH_ATTN"] = "off"
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_tps_floor_in_engine_defaults(self):
        from autoresearch.core.config import ENGINE_DEFAULTS

        self.assertIn("TPS_FLOOR", ENGINE_DEFAULTS)
        self.assertIsInstance(ENGINE_DEFAULTS["TPS_FLOOR"], (int, float))
        self.assertGreater(ENGINE_DEFAULTS["TPS_FLOOR"], 0)

    def test_validate_config_accepts_custom_tps_floor(self):
        cfg = load_config()
        cfg["N_CPU_MOE"] = None
        cfg["TPS_FLOOR"] = 15.0
        out = validate_config(cfg)
        self.assertEqual(out["TPS_FLOOR"], 15.0)

    def test_rejects_non_positive_tps_floor(self):
        cfg = load_config()
        cfg["TPS_FLOOR"] = 0.0
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_rejects_invalid_lowercase_override(self):
        cfg = load_config()
        cfg["ctx_size"] = 1024
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = SearchState(path)
            state.mark_visited("abc")
            self.assertTrue(state.is_visited("abc"))
            # Verify serialization round-trip
            fresh_state = SearchState(path)
            self.assertTrue(fresh_state.is_visited("abc"))
            self.assertNotIn("baseline", path.read_text(encoding="utf-8"))

    @patch(
        "sys.argv",
        ["benchmark_search.py", "--no-agentic-quick", "--no-agentic-full", "--desc", "x"],
    )
    def test_parse_args_can_disable_agentic_flags(self):
        args = run.parse_args()
        self.assertFalse(args.agentic_quick)
        self.assertFalse(args.agentic_full)


class TestNGpuLayersAndNuma(unittest.TestCase):
    """Issue #16: config surface for N_GPU_LAYERS + NUMA.

    Runs against the tracked config.py.example template (the local
    config.py Baseline is gitignored and machine-local, so it is not the
    contract under test).
    """

    @classmethod
    def setUpClass(cls):
        example_path = (
            Path(__file__).resolve().parents[1] / "autoresearch" / "core" / "config.py.example"
        )
        loader = SourceFileLoader("_example_config_template", str(example_path))
        spec = importlib.util.spec_from_loader("_example_config_template", loader)
        cls.example = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.example)

    def test_engine_defaults_example_has_new_knobs(self):
        self.assertIn("N_GPU_LAYERS", self.example.ENGINE_DEFAULTS)
        self.assertEqual(self.example.ENGINE_DEFAULTS["N_GPU_LAYERS"], -1)
        self.assertIn("NUMA", self.example.ENGINE_DEFAULTS)
        self.assertIsNone(self.example.ENGINE_DEFAULTS["NUMA"])
        self.assertIn("REASONING_PRESERVE", self.example.ENGINE_DEFAULTS)
        self.assertIsNone(self.example.ENGINE_DEFAULTS["REASONING_PRESERVE"])

    def test_accepts_valid_n_gpu_layers(self):
        for value in (-1, 0, 1, 40, 999):
            cfg = self.example.load_config()
            cfg["N_GPU_LAYERS"] = value
            out = self.example.validate_config(cfg)
            self.assertEqual(out["N_GPU_LAYERS"], value)

    def test_accepts_valid_numa(self):
        for value in (None, "distribute", "isolate"):
            cfg = self.example.load_config()
            cfg["NUMA"] = value
            out = self.example.validate_config(cfg)
            self.assertEqual(out["NUMA"], value)

    def test_accepts_lowercase_key_override(self):
        cfg = self.example.load_config()
        cfg["n_gpu_layers"] = 0
        cfg["numa"] = "isolate"
        out = self.example.validate_config(cfg)
        self.assertEqual(out["N_GPU_LAYERS"], 0)
        self.assertEqual(out["NUMA"], "isolate")

    def test_rejects_bad_n_gpu_layers(self):
        for value in (-2, -100, None, 2.5, "5", True, []):
            cfg = self.example.load_config()
            cfg["N_GPU_LAYERS"] = value
            with self.assertRaises(self.example.ConfigError):
                self.example.validate_config(cfg)

    def test_rejects_bad_numa(self):
        for value in ("DISTRIBUTE", "distributed", "isolated", "", "numa", 0, False):
            cfg = self.example.load_config()
            cfg["NUMA"] = value
            with self.assertRaises(self.example.ConfigError):
                self.example.validate_config(cfg)

    def test_accepts_reasoning_preserve_tri_state(self):
        for value in (None, True, False):
            cfg = self.example.load_config()
            cfg["REASONING_PRESERVE"] = value
            out = self.example.validate_config(cfg)
            self.assertEqual(out["REASONING_PRESERVE"], value)

    def test_rejects_bad_reasoning_preserve(self):
        for value in ("true", "false", 1, 0, "on"):
            cfg = self.example.load_config()
            cfg["REASONING_PRESERVE"] = value
            with self.assertRaises(self.example.ConfigError):
                self.example.validate_config(cfg)

    def test_accepts_valid_reasoning_effort(self):
        valid_efforts = (
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "fast",
            "faster",
            "LOW",
            "Fast",
            "NONE",
        )
        for value in valid_efforts:
            cfg = self.example.load_config()
            cfg["REASONING_EFFORT"] = value
            out = self.example.validate_config(cfg)
            expected = value.lower() if isinstance(value, str) else value
            self.assertEqual(out.get("reasoning_effort"), expected)

    def test_rejects_bad_reasoning_effort(self):
        for value in ("ultra", 123, True, ["low"], "disabled"):
            cfg = self.example.load_config()
            cfg["REASONING_EFFORT"] = value
            with self.assertRaises(self.example.ConfigError):
                self.example.validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
