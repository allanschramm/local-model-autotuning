import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from autoresearch.core.state import SearchState
from autoresearch.core import config
from autoresearch.core.config import write_baseline, load_config, ENGINE_DEFAULTS, SAMPLER_DEFAULTS


class TestSearchState(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "test_state.json"
        self.state = SearchState(self.state_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_baseline_reads_config(self):
        baseline = self.state.get_baseline()
        self.assertEqual(baseline["MODEL"], config.DEFAULTS["MODEL"])
        self.assertEqual(baseline["CTX_SIZE"], config.DEFAULTS["CTX_SIZE"])
        self.assertFalse(self.state_file.exists())

    def test_update_baseline_delegates_to_write_baseline(self):
        with patch("autoresearch.core.state.write_baseline") as mock_write:
            self.state.update_baseline({"MODEL": "custom.gguf", "CTX_SIZE": 8192})
            mock_write.assert_called_once()
            written = mock_write.call_args[0][0]
            self.assertEqual(written["MODEL"], "custom.gguf")
            self.assertEqual(written["CTX_SIZE"], 8192)

    def test_update_baseline_filters_keys(self):
        with patch("autoresearch.core.state.write_baseline") as mock_write:
            self.state.update_baseline({
                "MODEL": "custom.gguf",
                "CTX_SIZE": 8192,
                "FLASH_ATTN": "on",
                "UNKNOWN_KEY": "should_be_filtered",
            })
            written = mock_write.call_args[0][0]
            self.assertNotIn("UNKNOWN_KEY", written)

    def test_visited_memory(self):
        self.assertFalse(self.state.is_visited("config_key_1"))
        self.assertEqual(self.state.visited, set())

        self.state.mark_visited("config_key_1")
        self.assertTrue(self.state.is_visited("config_key_1"))
        self.assertEqual(self.state.visited, {"config_key_1"})

        fresh_state = SearchState(self.state_file)
        self.assertTrue(fresh_state.is_visited("config_key_1"))
        self.assertEqual(fresh_state.visited, {"config_key_1"})

        data = self.state_file.read_text(encoding="utf-8")
        self.assertIn('"schema_version": 2', data)
        self.assertNotIn('"baseline"', data)

        self.state.mark_visited("config_key_2")
        self.assertEqual(self.state.visited, {"config_key_1", "config_key_2"})

    def test_reset_clears_visited_only(self):
        self.state.mark_visited("key1")
        with patch("autoresearch.core.state.write_baseline") as mock_write:
            self.state.reset()
            mock_write.assert_not_called()
        self.assertEqual(self.state.visited, set())

    def test_loads_legacy_schema_v1_visited(self):
        self.state_file.write_text(
            '{"schema_version": 1, "baseline": {"MODEL": "old.gguf"}, "visited": ["a"]}\n',
            encoding="utf-8",
        )
        state = SearchState(self.state_file)
        self.assertTrue(state.is_visited("a"))
        self.assertEqual(state.get_baseline()["MODEL"], config.DEFAULTS["MODEL"])


class TestWriteBaseline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.py"
        self.config_path.write_text(
            "ENGINE_DEFAULTS = {\n"
            "    'MODEL': 'a.gguf',\n"
            "    'CTX_SIZE': 4096,\n"
            "    'KV_CACHE': 'q4_0',\n"
            "    'KV_CACHE_K': 'q4_0',\n"
            "    'KV_CACHE_V': 'q4_0',\n"
            "    'BATCH_SIZE': 256,\n"
            "    'UBATCH_SIZE': 128,\n"
            "    'THREADS': 8,\n"
            "    'THREADS_BATCH': 8,\n"
            "    'FLASH_ATTN': 'on',\n"
            "    'SPEC_TYPE': None,\n"
            "    'SPEC_DRAFT_N_MAX': 0,\n"
            "    'SPEC_DRAFT_MODEL': None,\n"
            "    'NO_MMAP': False,\n"
            "    'JINJA': False,\n"
            "    'REASONING_BUDGET': None,\n"
            "    'REASONING_BUDGET_MESSAGE': None,\n"
            "    'REASONING': None,\n"
            "    'CONT_BATCHING': True,\n"
            "    'N_CPU_MOE': 32,\n"
            "}\n"
            "SAMPLER_DEFAULTS = {\n"
            "    'TEMP': 0.4,\n"
            "    'TOP_P': 0.95,\n"
            "    'TOP_K': 20,\n"
            "    'MIN_P': 0.0,\n"
            "    'REPEAT_PENALTY': 1.0,\n"
            "    'PRESENCE_PENALTY': 0.0,\n"
            "    'FREQUENCY_PENALTY': None,\n"
            "}\n",
            encoding="utf-8",
        )
        self._engine_backup = dict(ENGINE_DEFAULTS)
        self._sampler_backup = dict(SAMPLER_DEFAULTS)

    def tearDown(self):
        ENGINE_DEFAULTS.clear()
        ENGINE_DEFAULTS.update(self._engine_backup)
        SAMPLER_DEFAULTS.clear()
        SAMPLER_DEFAULTS.update(self._sampler_backup)
        config._refresh_defaults()
        self.temp_dir.cleanup()

    def test_write_baseline_round_trip(self):
        result = write_baseline({"MODEL": "b.gguf", "CTX_SIZE": 8192}, path=self.config_path)
        self.assertEqual(result["MODEL"], "b.gguf")
        self.assertEqual(result["CTX_SIZE"], 8192)
        text = self.config_path.read_text(encoding="utf-8")
        self.assertIn("'MODEL': 'b.gguf'", text)
        self.assertIn("'CTX_SIZE': 8192", text)
        self.assertEqual(load_config()["MODEL"], "b.gguf")


if __name__ == "__main__":
    unittest.main()
