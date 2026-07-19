import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import autoloop
from autoresearch.core import config as core_config
from autoresearch.core.search import SearchStrategy


class TestAutoLoop(unittest.TestCase):

    def setUp(self):
        self._state_dir = tempfile.TemporaryDirectory()
        self._state_patch = patch.object(autoloop, "STATE_FILE", Path(self._state_dir.name) / "state.json")
        self._state_patch.start()

    def tearDown(self):
        self._state_patch.stop()
        self._state_dir.cleanup()

    def test_get_neighbors(self):
        config = {
            "KV_CACHE_K": "q4_0",
            "KV_CACHE_V": "q4_0",
            "THREADS": 12,
            "THREADS_BATCH": None,
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
            "SPEC_DRAFT_N_MAX": 1,
            "CTX_SIZE": 131072,
            "CONT_BATCHING": False,
            "FLASH_ATTN": "on",
            "NO_MMAP": False,
            "TEMP": 0.2,
            "TOP_P": None,
            "TOP_K": None,
            "MIN_P": None,
            "PRESENCE_PENALTY": None,
            "REPEAT_PENALTY": None,
        }

        search_strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        neighbors = search_strategy.get_neighbors(config)
        self.assertGreater(len(neighbors), 0)

        for neighbor in neighbors:
            all_keys = set(config.keys()) | set(neighbor.config.keys())
            diffs = sum(1 for k in all_keys if config.get(k) != neighbor.config.get(k))
            self.assertEqual(diffs, 1)

    def test_search_space_does_not_mutate_context(self):
        self.assertNotIn("CTX_SIZE", autoloop.SEARCH_SPACE)
        self.assertIn("CTX_SIZE", autoloop.PASSTHROUGH_PARAMS)

    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok(self, mock_estimate):
        mock_estimate.return_value = 5000.0
        cfg = {"MODEL": "m.gguf", "CTX_SIZE": 131072, "KV_CACHE_K": "q4_0"}

        self.assertTrue(autoloop.preflight_vram_ok(cfg, 6000.0))
        self.assertFalse(autoloop.preflight_vram_ok(cfg, 4000.0))
        self.assertTrue(autoloop.preflight_vram_ok(cfg, None))

    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok_fallback(self, mock_estimate):
        """KV_CACHE_K/V not set → falls back to KV_CACHE then q4_0."""
        mock_estimate.return_value = 5000.0
        cfg = {"MODEL": "m.gguf", "CTX_SIZE": 131072, "KV_CACHE": "q8_0"}
        self.assertTrue(autoloop.preflight_vram_ok(cfg, 9999.0))
        mock_estimate.assert_called_once()
        # Should use KV_CACHE value
        self.assertIn("q8_0", str(mock_estimate.call_args))

        mock_estimate.reset_mock()
        cfg2 = {"MODEL": "m.gguf", "CTX_SIZE": 131072}
        self.assertTrue(autoloop.preflight_vram_ok(cfg2, 9999.0))
        # Should fall back to "q4_0" default
        self.assertIn("q4_0", str(mock_estimate.call_args))

    def test_signal_handler(self):
        autoloop._stop_requested = False
        autoloop._signal_handler(None, None)
        self.assertTrue(autoloop._stop_requested)
        autoloop._stop_requested = False  # reset for other tests

    def test_load_config_returns_dict(self):
        """autoloop.load_config wraps config.load_config with search keys."""
        cfg = autoloop.load_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("KV_CACHE_K", cfg)
        self.assertIn("MODEL", cfg)
        self.assertIn("CTX_SIZE", cfg)

    def test_trial_config_maps_include_agentic_flags(self):
        cfg = {
            "MODEL": "m.gguf",
            "INCLUDE_CODING": False,
            "INCLUDE_AGENTIC_QUICK": True,
            "INCLUDE_AGENTIC_FULL": True,
        }
        out = autoloop.trial_config(cfg, {"port": 18080})
        self.assertFalse(out["include_coding"])
        self.assertTrue(out["agentic_quick"])
        self.assertTrue(out["agentic_full"])
        self.assertEqual(out["port"], 18080)

    def test_temp_baseline_in_search_space(self):
        self.assertIn(0.4, autoloop.SEARCH_SPACE["TEMP"])

    def test_get_neighbors_skips_ubatch_gt_batch(self):
        config = {
            "KV_CACHE_K": "q4_0",
            "KV_CACHE_V": "q4_0",
            "THREADS": 8,
            "THREADS_BATCH": 8,
            "BATCH_SIZE": 256,
            "UBATCH_SIZE": 256,
            "SPEC_DRAFT_N_MAX": 0,
            "CONT_BATCHING": True,
            "FLASH_ATTN": "on",
            "NO_MMAP": False,
            "TEMP": 0.4,
            "TOP_P": 0.95,
            "TOP_K": 20,
            "MIN_P": 0.0,
            "PRESENCE_PENALTY": 0.0,
            "REPEAT_PENALTY": 1.05,
        }
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        neighbors = strategy.get_neighbors(config)
        for n in neighbors:
            self.assertLessEqual(n.config["UBATCH_SIZE"], n.config["BATCH_SIZE"])

    @patch("autoloop.load_state", return_value={"visited": []})
    def test_load_visited_no_file(self, _mock_state):
        self.assertEqual(autoloop.load_visited(), set())

    def test_load_visited_with_data(self):
        with patch("autoloop.load_state", return_value={"visited": ["cfg1", "cfg2"]}):
            self.assertEqual(autoloop.load_visited(), {"cfg1", "cfg2"})

    @patch("autoloop.write_state")
    @patch("autoloop.load_state", return_value={"baseline": {}, "visited": []})
    def test_save_visited(self, _mock_load, mock_write):
        autoloop.save_visited({"a", "b"})
        saved = mock_write.call_args[0][0]
        self.assertEqual(saved["visited"], ["a", "b"])

    @patch("autoloop.write_state")
    @patch("autoloop.load_state", return_value={"baseline": {}, "visited": ["x"]})
    def test_save_visited_empty(self, _mock_load, mock_write):
        autoloop.save_visited(set())
        self.assertEqual(mock_write.call_args[0][0]["visited"], [])

    # ── main() tests ───────────────────────────────────────────────

    def _make_trial_result(self, **overrides):
        """Factory for run_trial result namespace."""
        defaults = {
            "val_score": 0.5, "avg_tps": 10.0, "peak_vram_gb": 2.0,
            "swe_val": 0.3, "he_val": 0.4, "mbpp_val": 0.6,
            "lcb_val": 0.5, "bigcode_val": 0.5,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _full_config(self, **overrides):
        cfg = {
            "BATCH_SIZE": 1024, "CONT_BATCHING": True, "CTX_SIZE": 131072,
            "FLASH_ATTN": "on", "KV_CACHE_K": "q4_0", "KV_CACHE_V": "q4_0",
            "MIN_P": 0.0, "NO_MMAP": False, "PRESENCE_PENALTY": 0.0,
            "REPEAT_PENALTY": 1.05, "SPEC_DRAFT_N_MAX": 0, "TEMP": 0.4,
            "THREADS": 8, "THREADS_BATCH": 8, "TOP_K": 20, "TOP_P": 0.95,
            "UBATCH_SIZE": 256,
            "KV_CACHE": "q4_0", "MODEL": "test.gguf", "JINJA": False,
            "REASONING_BUDGET": None, "REASONING_BUDGET_MESSAGE": None,
            "REASONING": None, "SPEC_TYPE": None, "FREQUENCY_PENALTY": None,
            "INCLUDE_CODING": True, "CODING_TASK_LIMIT": 10,
            "INCLUDE_NEXUS": False, "INCLUDE_CLAW": False,
            "INCLUDE_AGENTIC_QUICK": True, "INCLUDE_AGENTIC_FULL": True,
            "N_CPU_MOE": 32,
        }
        cfg.update(overrides)
        return cfg

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--vram-limit-mb", "99999", "--models", "test.gguf"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_main_single_round_no_neighbors(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """main() with --models flag (stdin non-tty fallback from baseline cfg)."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        # Baseline eval ran
        self.assertGreaterEqual(mock_runner.run_trial.call_count, 1)
        # Baseline written
        mock_write_row.assert_called()
        # "Exhausted random search space" reached → no crash

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_main_with_models_flag(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """--models flag with explicit model name."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        mock_write_row.assert_called()
        mock_wcfg.assert_called()

    @patch("sys.argv", ["autoloop.py", "--models", "nonexistent.gguf"])
    @patch("autoloop.MODELS_DIR")
    def test_main_model_not_found(self, mock_models_dir):
        """--models with name not in models dir → fuzzy match fallback then exit."""
        mock_models_dir.glob.return_value = [Path("real.gguf")]
        with self.assertRaises(SystemExit):
            autoloop.main()

    @patch("sys.argv", ["autoloop.py", "--models", "real"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    def test_main_model_fuzzy_match(self, mock_runner_cls, mock_models_dir):
        """--models with partial name → fuzzy match to first result."""
        mock_models_dir.glob.return_value = [Path("real.gguf"), Path("other.gguf")]
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner
        with patch("autoloop.load_config", return_value=self._full_config(MODEL="real.gguf")):
            with patch("autoloop.write_config"):
                with patch("autoloop.get_git_commit", return_value="abc"):
                    with patch("autoloop.write_row"):
                        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
                            with patch.object(SearchStrategy, "random_restart", return_value=None):
                                autoloop.main()

    @patch("sys.argv", ["autoloop.py", "--reset-visited", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop.write_state")
    @patch("autoloop.load_state", return_value={"baseline": {}, "visited": ["old"]})
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    def test_main_reset_visited(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir, mock_load_state, mock_write_state
    ):
        """--reset-visited clears visited keys in local state."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        self.assertEqual(mock_write_state.call_args_list[0].args[0]["visited"], [])
        mock_write_row.assert_called()

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_main_with_neighbor_improvement(
        self, mock_vram, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """Neighbor with better score → writes new config and breaks."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_vram.return_value = 1000.0  # under default 7900MB limit
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result(val_score=0.5)
        mock_runner_cls.return_value = mock_runner

        base_config = self._full_config(MODEL="test.gguf")
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)

        def side_is_imp(bs, bt, bv, s, t, v):
            return (True, "test improvement")

        with patch.object(SearchStrategy, "is_improvement") as mock_is_imp:
            mock_is_imp.side_effect = side_is_imp
            with patch.object(SearchStrategy, "get_neighbors") as mock_gn:
                nbr = strategy.get_neighbors(base_config)[0]
                mock_gn.return_value = [nbr]
                autoloop.main()

        mock_wcfg.assert_called()

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    def test_main_has_no_trial_budget(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """Trials run to completion without a budget override."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        args, kwargs = mock_runner.run_trial.call_args
        self.assertEqual(kwargs, {})

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf",
                         "--vram-limit-mb", "1"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_main_vram_skip(
        self, mock_vram, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """Neighbor exceeding VRAM limit gets skipped."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        # baseline VRAM OK, neighbor VRAM over limit
        mock_vram.return_value = 5000.0  # over 1MB limit

        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        base_config = self._full_config(MODEL="test.gguf")
        nbr = strategy.get_neighbors(base_config)[0]

        with patch.object(SearchStrategy, "get_neighbors", return_value=[nbr]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        # Neighbor was skipped (vram over budget), but baseline still ran
        self.assertGreaterEqual(mock_runner.run_trial.call_count, 1)

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf", "--perplexity-val"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=0.0)
    def test_main_with_perplexity_validation(
        self, mock_vram, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        """Main loop runs successfully with --perplexity-val active."""
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        
        # Mock result for baseline and neighbors
        base_res = self._make_trial_result()
        base_res.bench_ppl = 5.5
        base_res.val_score = 30.0
        
        mock_runner.run_trial.return_value = base_res
        mock_runner_cls.return_value = mock_runner

        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        base_config = self._full_config(MODEL="test.gguf")
        nbr = strategy.get_neighbors(base_config)[0]

        with patch.object(SearchStrategy, "get_neighbors", return_value=[nbr]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        # baseline and neighbor ran with perplexity_val active
        self.assertGreaterEqual(mock_runner.run_trial.call_count, 2)
        # Ensure include_perplexity parameter was set to True
        first_call_args, first_call_kwargs = mock_runner.run_trial.call_args_list[0]
        self.assertTrue(first_call_args[0]["include_perplexity"])

    @patch("autoloop.Path")
    def test_update_model_alias_success(self, mock_path_cls):
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            aliases_dir = Path(tmpdir) / "models" / "aliases"
            alias_dir = aliases_dir / "test-model"
            alias_dir.mkdir(parents=True)
            
            yaml_path = alias_dir / "config.yaml"
            dummy_config = {
                "alias": "test-model",
                "model": "models/test-model-gguf",
                "flags": [],
                "metrics": {"tps": 10.0}
            }
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dummy_config, f)
            
            # Setup path mock to return our temp dir
            mock_path = MagicMock()
            # Path(__file__) is resolved in autoloop.py, mock resolve().parent
            mock_path.resolve.return_value.parent = Path(tmpdir)
            mock_path_cls.return_value = mock_path
            
            new_cfg = {"THREADS": 4, "BATCH_SIZE": 512}
            autoloop.update_model_alias("test-model-v1.gguf", new_cfg, 25.5, "tps")
            
            with open(yaml_path, "r", encoding="utf-8") as f:
                updated = yaml.safe_load(f)
                
            self.assertEqual(updated["metrics"]["tps"], 25.5)
            self.assertIn("--threads 4", updated["flags"])

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf", "--mode", "tps"])
    @patch("autoloop.MODELS_DIR")
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.write_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=0.0)
    @patch("autoloop.update_model_alias")
    def test_main_with_tps_mode(
        self, mock_update_alias, mock_vram, mock_write_row, mock_git, mock_wcfg, mock_lcfg,
        mock_runner_cls, mock_models_dir
    ):
        mock_models_dir.glob.return_value = [Path("test.gguf")]
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner
        
        with patch.object(SearchStrategy, "random_restart", return_value=None):
            autoloop.main()
            
        self.assertGreaterEqual(mock_runner.run_trial.call_count, 1)


if __name__ == "__main__":
    unittest.main()
