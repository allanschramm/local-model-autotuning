import unittest
import tempfile
from pathlib import Path
from autoresearch.core.state import SearchState
from autoresearch.core.config import STATE_SCHEMA_VERSION
from autoresearch.core import config


class TestSearchState(unittest.TestCase):

    def setUp(self):
        # Create a temporary file for state path
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "test_state.json"
        self.state = SearchState(self.state_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_defaults(self):
        # When file doesn't exist, get_baseline should return config.DEFAULTS
        baseline = self.state.get_baseline()
        self.assertEqual(baseline["MODEL"], config.DEFAULTS["MODEL"])
        self.assertEqual(baseline["CTX_SIZE"], config.DEFAULTS["CTX_SIZE"])
        self.assertFalse(self.state_file.exists())

    def test_update_baseline(self):
        new_cfg = {"MODEL": "custom.gguf", "CTX_SIZE": 8192, "FLASH_ATTN": "on", "TEMP": 0.7}
        self.state.update_baseline(new_cfg)
        
        # Verify file is created and contains correct data
        self.assertTrue(self.state_file.exists())
        
        # Verify file round-trip by instantiating a fresh SearchState
        fresh_state = SearchState(self.state_file)
        baseline = fresh_state.get_baseline()
        self.assertEqual(baseline["MODEL"], "custom.gguf")
        self.assertEqual(baseline["CTX_SIZE"], 8192)
        # Defaults should be preserved for missing keys
        self.assertEqual(baseline["KV_CACHE"], config.DEFAULTS["KV_CACHE"])

    def test_update_baseline_merges(self):
        # Initial update
        self.state.update_baseline({"MODEL": "custom.gguf", "CTX_SIZE": 8192})
        # Second update should merge, not overwrite other keys
        self.state.update_baseline({"CTX_SIZE": 16384})
        
        baseline = self.state.get_baseline()
        self.assertEqual(baseline["MODEL"], "custom.gguf")
        self.assertEqual(baseline["CTX_SIZE"], 16384)

    def test_update_baseline_filters_keys(self):
        # update_baseline should filter out keys not in config.CONFIG_KEYS
        new_cfg = {"MODEL": "custom.gguf", "CTX_SIZE": 8192, "FLASH_ATTN": "on", "UNKNOWN_KEY": "should_be_filtered"}
        self.state.update_baseline(new_cfg)
        
        baseline = self.state.get_baseline()
        self.assertNotIn("UNKNOWN_KEY", baseline)

    def test_visited_memory(self):
        self.assertFalse(self.state.is_visited("config_key_1"))
        self.assertEqual(self.state.visited, set())

        self.state.mark_visited("config_key_1")
        self.assertTrue(self.state.is_visited("config_key_1"))
        self.assertEqual(self.state.visited, {"config_key_1"})

        # Verify fresh SearchState instance loads visited keys from disk
        fresh_state = SearchState(self.state_file)
        self.assertTrue(fresh_state.is_visited("config_key_1"))
        self.assertEqual(fresh_state.visited, {"config_key_1"})

        self.state.mark_visited("config_key_2")
        self.assertTrue(self.state.is_visited("config_key_2"))
        self.assertEqual(self.state.visited, {"config_key_1", "config_key_2"})

    def test_reset(self):
        new_cfg = {"MODEL": "custom.gguf", "CTX_SIZE": 8192, "FLASH_ATTN": "on"}
        self.state.update_baseline(new_cfg)
        self.state.mark_visited("key1")

        self.state.reset()
        
        baseline = self.state.get_baseline()
        self.assertEqual(baseline["MODEL"], config.DEFAULTS["MODEL"])
        self.assertEqual(self.state.visited, set())


if __name__ == "__main__":
    unittest.main()
