import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import autoloop
from autoresearch.core import classify
from autoresearch.core.search import SearchStrategy


class TestAutoLoop(unittest.TestCase):
    def setUp(self):
        self._state_dir = tempfile.TemporaryDirectory()
        temp_file = Path(self._state_dir.name) / "state.json"
        self._state_patch = patch("autoresearch.core.config.STATE_FILE", temp_file)
        self._state_patch.start()
        # Keep classification hermetic: never read/flip the real results.tsv.
        self._rows_patch = patch("autoloop.read_rows", return_value=[])
        self._rows_patch.start()
        self._flips_patch = patch("autoloop.recompute_statuses")
        self._flips_patch.start()

    def tearDown(self):
        self._state_patch.stop()
        self._rows_patch.stop()
        self._flips_patch.stop()
        self._state_dir.cleanup()

    def test_seed_known_vectors_loads_complete_vectors_for_model(self):
        rows = [
            {"model": "a.gguf", "ctx": "8192", "tps": "50.0", "agentic": "0.5", "coding": "0.6"},
            {"model": "a.gguf", "ctx": "8192", "tps": "40.0", "agentic": "", "coding": "0.6"},
            {"model": "b.gguf", "ctx": "8192", "tps": "60.0", "agentic": "0.7", "coding": "0.8"},
        ]
        with patch("autoloop.read_rows", return_value=rows):
            vectors = autoloop._seed_known_vectors("a.gguf", bucket_gb=None)
        self.assertEqual(len(vectors), 1)
        self.assertTrue(vectors[0].complete)
        self.assertEqual(vectors[0].tps, 50.0)
        self.assertEqual(vectors[0].agentic, 0.5)

    def test_seed_known_vectors_filters_by_bucket(self):
        rows = [
            {
                "model": "a.gguf",
                "status": "on_front",
                "ctx": "8192",
                "tps": "50.0",
                "agentic": "0.5",
                "coding": "0.6",
                "config_json": '{"vram_limit_mb": 8192}',
            },
            {
                "model": "a.gguf",
                "status": "on_front",
                "ctx": "8192",
                "tps": "60.0",
                "agentic": "0.7",
                "coding": "0.8",
                "config_json": '{"vram_limit_mb": 6144}',
            },
            {
                "model": "a.gguf",
                "status": "rejected",
                "ctx": "8192",
                "tps": "70.0",
                "agentic": "0.9",
                "coding": "0.9",
                "config_json": '{"vram_limit_mb": 8192}',
            },
        ]
        with patch("autoloop.read_rows", return_value=rows):
            vectors = autoloop._seed_known_vectors("a.gguf", bucket_gb=8)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].tps, 50.0)

    def test_seed_known_vectors_matches_classify_known_vectors(self):
        # ADR 0017 dedup (review of PR #71, Standards Finding 2): the autoloop
        # Search seed IS the classify Known Set — delegation, not a mirror.
        rows = [
            {
                "model": "a.gguf",
                "status": "on_front",
                "ctx": "8192",
                "tps": "50.0",
                "agentic": "0.5",
                "coding": "0.6",
                "config_json": '{"vram_limit_mb": 8192}',
            },
            {
                "model": "b.gguf",
                "status": "on_front",
                "ctx": "8192",
                "tps": "99.0",
                "agentic": "0.9",
                "coding": "0.9",
                "config_json": '{"vram_limit_mb": 8192}',
            },
            {
                "model": "a.gguf",
                "status": "on_front",
                "ctx": "8192",
                "tps": "99.0",
                "agentic": "0.9",
                "coding": "0.9",
                "evaluation_profile": "morris-screen",
            },
        ]
        with patch("autoloop.read_rows", return_value=rows):
            seeded = autoloop._seed_known_vectors("a.gguf", bucket_gb=8)
        self.assertEqual(seeded, classify.known_vectors(rows, model="a.gguf", bucket_gb=8))
        self.assertEqual(len(seeded), 1)

    def test_seed_known_vectors_skips_morris_screen_rows(self):
        rows = [
            {"model": "a.gguf", "ctx": "8192", "tps": "50.0", "agentic": "0.5", "coding": "0.6"},
            {
                "model": "a.gguf",
                "ctx": "8192",
                "tps": "99.0",
                "agentic": "",
                "coding": "",
                "evaluation_profile": "morris-screen",
            },
        ]
        with patch("autoloop.read_rows", return_value=rows):
            vectors = autoloop._seed_known_vectors("a.gguf", bucket_gb=None)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].tps, 50.0)

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

    def test_core_passthrough_surfaces_gpu_and_numa(self):
        self.assertIn("N_GPU_LAYERS", autoloop.CORE_PASSTHROUGH)
        self.assertIn("NUMA", autoloop.CORE_PASSTHROUGH)

    def test_reasoning_preserve_is_passthrough_not_search(self):
        self.assertIn("REASONING_PRESERVE", autoloop.CORE_PASSTHROUGH)
        self.assertNotIn("REASONING_PRESERVE", autoloop.SEARCH_SPACE)

    @patch("autoloop.preflight_host_ok", return_value=True)
    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok(self, mock_estimate, _host):
        mock_estimate.return_value = 5000.0
        cfg = {"MODEL": "m.gguf", "CTX_SIZE": 131072, "KV_CACHE_K": "q4_0"}

        self.assertTrue(autoloop.preflight_vram_ok(cfg, 6000.0))
        self.assertFalse(autoloop.preflight_vram_ok(cfg, 4000.0))
        self.assertTrue(autoloop.preflight_vram_ok(cfg, None))

    @patch("autoloop.preflight_host_ok", return_value=True)
    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok_fallback(self, mock_estimate, _host):
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

    @patch(
        "autoresearch.core.llama_runner.gguf_has_mtp",
        side_effect=lambda p: "embedded-mtp" in str(p).lower(),
    )
    @patch("autoloop.preflight_host_ok", return_value=True)
    @patch("autoloop.estimate_vram_mb")
    def test_preflight_vram_ok_infers_mtp_like_eval(self, mock_estimate, _host, _mtp):
        """MTP-via-GGUF-metadata models pass spec args so autoloop and eval preflight agree."""
        mock_estimate.return_value = 5000.0
        cfg = {"MODEL": "embedded-MTP.gguf", "CTX_SIZE": 131072, "SPEC_DRAFT_N_MAX": 4}

        self.assertTrue(autoloop.preflight_vram_ok(cfg, 9999.0))
        kwargs = mock_estimate.call_args.kwargs
        self.assertEqual(kwargs["spec_type"], "mtp")
        self.assertEqual(kwargs["spec_draft_n_max"], 4)
        self.assertIsNone(kwargs["draft_path"])  # embedded MTP: no external draft

        mock_estimate.reset_mock()
        cfg2 = {"MODEL": "plain.gguf", "CTX_SIZE": 131072, "SPEC_DRAFT_N_MAX": 0}
        self.assertTrue(autoloop.preflight_vram_ok(cfg2, 9999.0))
        kwargs2 = mock_estimate.call_args.kwargs
        self.assertIsNone(kwargs2["spec_type"])
        self.assertEqual(kwargs2["spec_draft_n_max"], 0)

    def test_signal_handler(self):
        autoloop._stop_requested = False
        autoloop._signal_handler(None, None)
        self.assertTrue(autoloop._stop_requested)
        autoloop._stop_requested = False  # reset for other tests

    def test_load_config_returns_dict(self):
        """autoloop.load_config wraps config.load_config with search keys."""
        state = autoloop.SearchState()
        cfg = autoloop.load_config(state.get_baseline())
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
        self.assertFalse(out.get("agentic_coding"))
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

    # ── main() tests ───────────────────────────────────────────────

    def _make_trial_result(self, **overrides):
        """Factory for run_trial result namespace."""
        defaults = {
            "val_score": 0.5,
            "avg_tps": 10.0,
            "peak_vram_gb": 2.0,
            "swe_val": 0.3,
            "he_val": 0.4,
            "mbpp_val": 0.6,
            "lcb_val": 0.5,
            "bigcode_val": 0.5,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _full_config(self, **overrides):
        cfg = {
            "BATCH_SIZE": 1024,
            "CONT_BATCHING": True,
            "CTX_SIZE": 131072,
            "FLASH_ATTN": "on",
            "KV_CACHE_K": "q4_0",
            "KV_CACHE_V": "q4_0",
            "MIN_P": 0.0,
            "NO_MMAP": False,
            "PRESENCE_PENALTY": 0.0,
            "REPEAT_PENALTY": 1.05,
            "SPEC_DRAFT_N_MAX": 0,
            "TEMP": 0.4,
            "THREADS": 8,
            "THREADS_BATCH": 8,
            "TOP_K": 20,
            "TOP_P": 0.95,
            "UBATCH_SIZE": 256,
            "KV_CACHE": "q4_0",
            "MODEL": "test.gguf",
            "JINJA": False,
            "REASONING_BUDGET": None,
            "REASONING_BUDGET_MESSAGE": None,
            "REASONING": None,
            "REASONING_PRESERVE": None,
            "SPEC_TYPE": None,
            "FREQUENCY_PENALTY": None,
            "INCLUDE_CODING": True,
            "CODING_TASK_LIMIT": 10,
            "INCLUDE_AGENTIC_QUICK": True,
            "INCLUDE_AGENTIC_FULL": True,
            "N_CPU_MOE": 32,
            "VRAM_LIMIT_MB": 7900,
        }
        cfg.update(overrides)
        return cfg

    def _evaluable_neighbor(self, strategy, base_config):
        """First neighbor whose changed key survives CPU search-space filtering.

        CPU-only hosts drop SPEC_DRAFT_N_MAX from the active search space, so
        a SPEC_DRAFT_N_MAX-only neighbor serializes to the baseline's config
        key and is skipped as already-visited (flake on no-GPU CI). Pick any
        other single-flag neighbor deterministically.
        """
        return next(
            n
            for n in strategy.get_neighbors(base_config)
            if n.changed not in autoloop.CPU_EXCLUDED_SEARCH_KEYS
        )

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_main_single_round_no_neighbors(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """main() with --models flag (stdin non-tty fallback from baseline cfg)."""
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

    @patch(
        "sys.argv",
        ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf", "--mode", "tps"],
    )
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_main_defaults_use_config_n_gpu_layers(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """_defaults N_GPU_LAYERS comes from config, never a hardcoded 99."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf", N_GPU_LAYERS=0)
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        first_config = mock_runner.run_trial.call_args.args[0]
        self.assertEqual(first_config["N_GPU_LAYERS"], 0)
        self.assertNotIn("ngl", first_config)

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_autoloop_write_row_includes_throughput_and_flat_config(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """AutoLoop baseline row must pass tps/bench_tg and flat engine/sampler fields."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result(
            avg_tps=47.7,
            bench_tg_tps=43.2,
            tps_source="llama-bench",
        )
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        kwargs = mock_write_row.call_args.kwargs
        self.assertEqual(kwargs["tps"], 47.7)
        self.assertEqual(kwargs["bench_tg"], 43.2)
        self.assertEqual(kwargs["tps_source"], "llama-bench")
        self.assertEqual(kwargs["kv"], "q4_0")
        self.assertEqual(kwargs["ctx"], 131072)
        self.assertEqual(kwargs["threads"], 8)
        self.assertEqual(kwargs["batch_size"], 1024)
        self.assertEqual(kwargs["n_cpu_moe"], 32)
        self.assertEqual(kwargs["min_p"], 0.0)
        self.assertEqual(kwargs["presence_penalty"], 0.0)
        self.assertEqual(kwargs["spec_draft_n_max"], 0)

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_main_with_models_flag(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """--models flag with explicit model name."""
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
    @patch("autoloop._available_gguf_names", return_value=["real.gguf"])
    def test_main_model_not_found(self, _mock_models):
        """--models with name not in models dir → fuzzy match fallback then exit."""
        with self.assertRaises(SystemExit):
            autoloop.main()

    @patch("sys.argv", ["autoloop.py", "--models", "real"])
    @patch("autoloop._available_gguf_names", return_value=["real.gguf", "other.gguf"])
    @patch("autoloop.ExperimentRunner")
    def test_main_model_fuzzy_match(self, mock_runner_cls, _mock_models):
        """--models with partial name → fuzzy match to first result."""
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner
        with patch("autoloop.load_config", return_value=self._full_config(MODEL="real.gguf")):
            with patch("autoloop.SearchState.update_baseline"):
                with patch("autoloop.get_git_commit", return_value="abc"):
                    with patch("autoloop.write_row"):
                        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
                            with patch.object(SearchStrategy, "random_restart", return_value=None):
                                autoloop.main()

    @patch(
        "sys.argv", ["autoloop.py", "--reset-visited", "--max-rounds", "1", "--models", "test.gguf"]
    )
    @patch("autoloop.SearchState.reset")
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    def test_main_reset_visited(
        self,
        mock_write_row,
        mock_git,
        mock_update_baseline,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
        mock_reset,
    ):
        """--reset-visited clears visited keys in local state."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        mock_reset.assert_called_once()
        mock_update_baseline.assert_called()
        mock_write_row.assert_called()

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_main_with_neighbor_improvement(
        self,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """Neighbor with better score → writes new config and breaks."""
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
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    def test_main_has_no_trial_budget(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """Trials run to completion without a budget override."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        args, kwargs = mock_runner.run_trial.call_args
        self.assertEqual(kwargs, {})

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_main_vram_skip(
        self,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """Neighbor exceeding VRAM limit gets skipped."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf", VRAM_LIMIT_MB=1)
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        # baseline VRAM OK, neighbor VRAM over limit
        mock_vram.return_value = 5000.0  # over 1MB limit

        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        base_config = self._full_config(MODEL="test.gguf")
        nbr = self._evaluable_neighbor(strategy, base_config)

        with patch.object(SearchStrategy, "get_neighbors", return_value=[nbr]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        # Neighbor was skipped (vram over budget), but baseline still ran
        self.assertEqual(mock_runner.run_trial.call_count, 1)

    @patch(
        "sys.argv",
        ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf", "--perplexity-val"],
    )
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=0.0)
    def test_main_with_perplexity_validation(
        self,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """Main loop runs successfully with --perplexity-val active."""
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
        nbr = self._evaluable_neighbor(strategy, base_config)

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
                "flags": ["--n-gpu-layers 42"],
                "metrics": {"tps": 10.0},
            }
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dummy_config, f)

            # Setup path mock to return our temp dir
            mock_path = MagicMock()
            # Path(__file__) is resolved in autoloop.py, mock resolve().parent
            mock_path.resolve.return_value.parent = Path(tmpdir)
            mock_path_cls.return_value = mock_path

            new_cfg = {
                "THREADS": 4,
                "BATCH_SIZE": 512,
                "N_CPU_MOE": 32,
                "NO_MMAP": True,
            }
            autoloop.update_model_alias("test-model-v1.gguf", new_cfg, 25.5, "tps")

            with open(yaml_path, encoding="utf-8") as f:
                updated = yaml.safe_load(f)

            self.assertEqual(updated["metrics"]["tps"], 25.5)
            self.assertIn("--threads 4", updated["flags"])
            self.assertIn("--n-gpu-layers 42", updated["flags"])
            self.assertIn("--n-cpu-moe 32", updated["flags"])
            self.assertIn("--no-mmap", updated["flags"])
            self.assertNotIn("--n-gpu-layers 99", updated["flags"])

    @patch("autoloop.Path")
    def test_update_model_alias_prefers_config_n_gpu_layers(self, mock_path_cls):
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            aliases_dir = Path(tmpdir) / "models" / "aliases"
            alias_dir = aliases_dir / "test-model"
            alias_dir.mkdir(parents=True)

            yaml_path = alias_dir / "config.yaml"
            dummy_config = {
                "alias": "test-model",
                "model": "models/test-model-gguf",
                "flags": ["--n-gpu-layers 42"],
                "metrics": {"tps": 10.0},
            }
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dummy_config, f)

            mock_path = MagicMock()
            mock_path.resolve.return_value.parent = Path(tmpdir)
            mock_path_cls.return_value = mock_path

            new_cfg = {
                "THREADS": 4,
                "N_GPU_LAYERS": 0,
            }
            autoloop.update_model_alias("test-model-v1.gguf", new_cfg, 25.5, "tps")

            with open(yaml_path, encoding="utf-8") as f:
                updated = yaml.safe_load(f)

            self.assertIn("--n-gpu-layers 0", updated["flags"])
            self.assertNotIn("--n-gpu-layers 42", updated["flags"])
            self.assertIn("--threads 4", updated["flags"])

    def test_family_slug(self):
        self.assertEqual(
            autoloop._family_slug("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), "qwen3.6-35b-a3b-ud"
        )
        self.assertEqual(autoloop._family_slug("POCKET-35B-Q3_K_M.gguf"), "pocket-35b")
        self.assertEqual(
            autoloop._family_slug("LFM2.5-1.2B-Instruct-Q8_0.gguf"), "lfm2.5-1.2b-instruct"
        )
        self.assertEqual(autoloop._family_slug("Qwythos-9B-v2-Q4_K_M.gguf"), "qwythos-9b-v2")
        self.assertEqual(
            autoloop._family_slug("mtp-gemma-4-26B-A4B-it.gguf"), "mtp-gemma-4-26b-a4b-it"
        )
        # Quant tags never survive the slug: quants of a family map to one alias.
        self.assertEqual(
            autoloop._family_slug("Fam-9B-Q4_K_M.gguf"),
            autoloop._family_slug("Fam-9B-Q3_K_XL.gguf"),
        )

    @patch("autoloop.Path")
    def test_update_model_alias_creates_when_missing(self, mock_path_cls):
        """First Trial for a family creates the kebab-case alias if missing."""
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            aliases_dir = Path(tmpdir) / "models" / "aliases"
            aliases_dir.mkdir(parents=True)

            mock_path = MagicMock()
            mock_path.resolve.return_value.parent = Path(tmpdir)
            mock_path_cls.return_value = mock_path

            new_cfg = {
                "CTX_SIZE": 131072,
                "JINJA": True,
                "THREADS": 8,
                "KV_CACHE_K": "q4_0",
                "KV_CACHE_V": "q4_0",
                "FLASH_ATTN": "on",
                "N_CPU_MOE": 32,
            }
            autoloop.update_model_alias("TestFamily-9B-A3B-UD-Q4_K_M.gguf", new_cfg, 25.5, "tps")

            alias_dir = aliases_dir / "testfamily-9b-a3b-ud"
            self.assertTrue(alias_dir.is_dir())
            yaml_path = alias_dir / "config.yaml"
            self.assertTrue(yaml_path.exists())
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.assertEqual(data["alias"], "testfamily-9b-a3b-ud")
            self.assertEqual(data["model"], "models/TestFamily-9B-A3B-UD-Q4_K_M.gguf")
            self.assertEqual(data["port"], 18080)
            self.assertEqual(data["host"], "127.0.0.1")
            self.assertEqual(data["status"], "ready")
            self.assertEqual(data["metrics"]["tps"], 25.5)
            self.assertEqual(data["metrics"]["measured_by"], "autoloop")
            self.assertIn("--jinja", data["flags"])
            self.assertIn("--ctx-size 131072", data["flags"])
            self.assertIn("--threads 8", data["flags"])
            self.assertIn("--n-cpu-moe 32", data["flags"])
            self.assertIn("--cache-type-k q4_0", data["flags"])

    @patch("autoloop.Path")
    def test_update_model_alias_quant_change_overwrites_same_alias(self, mock_path_cls):
        """A new quant of the family overwrites the same alias, never a second one."""
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            aliases_dir = Path(tmpdir) / "models" / "aliases"
            alias_dir = aliases_dir / "testfamily-9b-a3b-ud"
            alias_dir.mkdir(parents=True)
            yaml_path = alias_dir / "config.yaml"
            dummy = {
                "alias": "testfamily-9b-a3b-ud",
                "model": "models/TestFamily-9B-A3B-UD-Q4_K_M.gguf",
                "flags": ["--n-gpu-layers 42"],
                "metrics": {"tps": 20.0},
            }
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dummy, f)

            mock_path = MagicMock()
            mock_path.resolve.return_value.parent = Path(tmpdir)
            mock_path_cls.return_value = mock_path

            autoloop.update_model_alias(
                "TestFamily-9B-A3B-UD-Q4_K_XL.gguf", {"THREADS": 12, "N_GPU_LAYERS": 0}, 30.0, "tps"
            )

            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.assertEqual(len(list(aliases_dir.iterdir())), 1)
            self.assertEqual(data["model"], "models/TestFamily-9B-A3B-UD-Q4_K_XL.gguf")
            self.assertEqual(data["metrics"]["tps"], 30.0)
            self.assertIn("--n-gpu-layers 0", data["flags"])
            self.assertNotIn("--n-gpu-layers 42", data["flags"])
            self.assertIn("--threads 12", data["flags"])

    @patch(
        "sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf", "--mode", "tps"]
    )
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=0.0)
    @patch("autoloop.update_model_alias")
    def test_main_with_tps_mode(
        self,
        mock_update_alias,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "random_restart", return_value=None):
            autoloop.main()

        self.assertGreaterEqual(mock_runner.run_trial.call_count, 1)

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_autoloop_trials_are_classified_not_keep_discard(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """AutoLoop trials write ADR 0006 statuses, never scalar keep/discard (issue #4)."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        # No agentic tier, no coding -> partial vector -> incomplete (never keep).
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        baseline_status = mock_write_row.call_args.args[7]
        self.assertIn(baseline_status, {"incomplete", "on_front", "dominated", "rejected"})
        self.assertNotIn(baseline_status, {"keep", "discard"})

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_autoloop_rejected_baseline_writes_rejected_and_restarts(
        self,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """MODEL_REJECTED baseline lands as rejected and triggers Random Restart."""
        from autoresearch.runners.evaluation import TrialOutcome

        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_vram.return_value = 1000.0
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result(
            outcome=TrialOutcome.MODEL_REJECTED, status="FAIL: VRAM_LIMIT_EXCEEDED"
        )
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with patch.object(SearchStrategy, "random_restart", return_value=None):
                autoloop.main()

        self.assertEqual(mock_write_row.call_args.args[7], "rejected")
        self.assertEqual(mock_write_row.call_args.kwargs["outcome"], "MODEL_REJECTED")

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    def test_autoloop_infra_error_is_recorded_then_stops(
        self, mock_write_row, mock_git, mock_wcfg, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """INFRA_ERROR Trial is persisted as rejected before the loop raises (issue #4)."""
        from autoresearch.runners.evaluation import TrialOutcome

        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result(
            outcome=TrialOutcome.INFRA_ERROR, status="FAIL: server crashed"
        )
        mock_runner_cls.return_value = mock_runner

        with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
            with self.assertRaises(RuntimeError):
                autoloop.main()

        self.assertEqual(mock_write_row.call_args.args[7], "rejected")
        self.assertEqual(mock_write_row.call_args.kwargs["outcome"], "INFRA_ERROR")

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc123")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb")
    def test_autoloop_neighbor_infra_error_is_recorded_then_stops(
        self,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """A neighbor INFRA_ERROR is persisted as rejected before the loop stops."""
        from autoresearch.runners.evaluation import TrialOutcome

        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_vram.return_value = 1000.0
        mock_runner = MagicMock()
        mock_runner.run_trial.side_effect = [
            self._make_trial_result(),  # baseline OK
            self._make_trial_result(outcome=TrialOutcome.INFRA_ERROR, status="FAIL: crashed"),
        ]
        mock_runner_cls.return_value = mock_runner

        base_config = self._full_config(MODEL="test.gguf")
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        with patch.object(SearchStrategy, "get_neighbors") as mock_gn:
            nbr = strategy.get_neighbors(base_config)[0]
            mock_gn.return_value = [nbr]
            with self.assertRaises(RuntimeError):
                autoloop.main()

        # Last write = the neighbor Trial, recorded as rejected before the raise.
        self.assertEqual(mock_write_row.call_args.args[7], "rejected")
        self.assertEqual(mock_write_row.call_args.kwargs["outcome"], "INFRA_ERROR")

    def test_profile_flag_removed(self):
        """ADR 0017: --profile auto-seed path is gone; --models is the only route."""
        with patch("sys.argv", ["autoloop.py", "--profile", "day"]):
            with self.assertRaises(SystemExit):
                autoloop.main()

    @patch("sys.argv", ["autoloop.py", "--dry-run", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch(
        "autoloop.detect_hardware_capabilities",
        return_value={"has_gpu": False, "physical_cores": 8, "ram_mb": 16384.0},
    )
    def test_main_dry_run_runs_no_trials(
        self, mock_detect, mock_lcfg, mock_runner_cls, _mock_models
    ):
        """--dry-run prints the plan; no benchmarks, no runner, no writes."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        with patch("autoloop.get_git_commit", return_value="abc"):
            with patch("autoloop.write_row") as mock_write_row:
                autoloop.main()
        mock_runner_cls.assert_not_called()
        mock_write_row.assert_not_called()

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=1000.0)
    @patch("autoloop.preflight_host_ok", return_value=True)
    @patch("autoloop.update_model_alias")
    def test_main_neighbor_pareto_acceptance(
        self,
        mock_alias,
        mock_host,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """Complete-vector neighbor is accepted via improves_set, not scalar keep (issue #8)."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result(
            agentic_tier="full", agentic_val=0.5, coding_val=0.6
        )
        mock_runner_cls.return_value = mock_runner

        base_config = self._full_config(MODEL="test.gguf")
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        nbr = self._evaluable_neighbor(strategy, base_config)

        with patch.object(SearchStrategy, "get_neighbors", return_value=[nbr]):
            with patch.object(SearchStrategy, "improves_set", return_value=True) as mock_is:
                with patch.object(SearchStrategy, "random_restart", return_value=None):
                    autoloop.main()

        # Pareto acceptance drove the baseline move (scalar keep never called).
        mock_is.assert_called()
        mock_wcfg.assert_called()
        self.assertTrue(mock_write_row.call_count >= 2)

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.SearchState.update_baseline")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=1000.0)
    @patch("autoloop.preflight_host_ok", return_value=True)
    @patch("autoloop.update_model_alias")
    def test_main_neighbor_incomplete_uses_scalar_fallback(
        self,
        mock_alias,
        mock_host,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_wcfg,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """Incomplete-vector neighbor (no agentic/coding) falls back to scalar keep."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        # Default result: agentic_tier="" → incomplete vector → scalar fallback.
        mock_runner.run_trial.return_value = self._make_trial_result(val_score=0.5)
        mock_runner_cls.return_value = mock_runner

        base_config = self._full_config(MODEL="test.gguf")
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        nbr = self._evaluable_neighbor(strategy, base_config)

        with patch.object(SearchStrategy, "get_neighbors", return_value=[nbr]):
            with patch.object(SearchStrategy, "is_improvement") as mock_scalar:
                mock_scalar.return_value = (True, "scalar keep")
                with patch.object(SearchStrategy, "random_restart", return_value=None):
                    autoloop.main()

        mock_scalar.assert_called()
        mock_wcfg.assert_called()


class TestAutoLoopCpuPreflight(TestAutoLoop):
    """CPU preflight + CPU Search Space filtering (issue #19).

    Inherits TestAutoLoop's hermetic setUp/tearDown and config factories.
    """

    def test_search_space_includes_numa(self):
        self.assertIn("NUMA", autoloop.SEARCH_SPACE)
        self.assertEqual(autoloop.SEARCH_SPACE["NUMA"], [None, "distribute", "isolate"])

    def test_search_space_includes_spec_type(self):
        self.assertIn("SPEC_TYPE", autoloop.SEARCH_SPACE)
        self.assertEqual(autoloop.SEARCH_SPACE["SPEC_TYPE"], [None, "ngram-cache"])

    def test_neighbor_generation_reaches_ngram_cache(self):
        strategy = SearchStrategy(autoloop.SEARCH_SPACE, use_pareto_tiebreaker=True)
        base_cfg = self._full_config(SPEC_TYPE=None)
        neighbors = strategy.get_neighbors(base_cfg)
        ngram_neighbors = [n for n in neighbors if n.changed == "SPEC_TYPE"]
        self.assertEqual(len(ngram_neighbors), 1)
        self.assertIsNone(ngram_neighbors[0].old)
        self.assertEqual(ngram_neighbors[0].new, "ngram-cache")
        self.assertEqual(ngram_neighbors[0].config["SPEC_TYPE"], "ngram-cache")

    @patch(
        "autoloop.detect_hardware_capabilities",
        return_value={"has_gpu": False, "physical_cores": 8, "ram_mb": 16384.0},
    )
    def test_apply_cpu_preflight_no_gpu_seeds_zero(self, _mock_detect):
        """Auto (-1) on a GPU-less host -> seed N_GPU_LAYERS=0."""
        out = autoloop.apply_cpu_preflight({"N_GPU_LAYERS": -1, "THREADS": 8})
        self.assertEqual(out["N_GPU_LAYERS"], 0)
        self.assertEqual(out["THREADS"], 8)

    @patch(
        "autoloop.detect_hardware_capabilities",
        return_value={"has_gpu": True, "physical_cores": 8, "ram_mb": 16384.0},
    )
    def test_apply_cpu_preflight_gpu_keeps_auto(self, _mock_detect):
        """Auto (-1) with a GPU present -> leave Auto (-1), no seed."""
        self.assertIsNone(autoloop.apply_cpu_preflight({"N_GPU_LAYERS": -1}))

    @patch("autoloop.detect_hardware_capabilities")
    def test_apply_cpu_preflight_explicit_ngl_not_overridden(self, mock_detect):
        """Explicit N_GPU_LAYERS (0 or N) is never overridden; no detection runs."""
        self.assertIsNone(autoloop.apply_cpu_preflight({"N_GPU_LAYERS": 0}))
        self.assertIsNone(autoloop.apply_cpu_preflight({"N_GPU_LAYERS": 42}))
        mock_detect.assert_not_called()

    def test_apply_cpu_preflight_missing_key_treated_as_auto(self):
        """Stale baseline without N_GPU_LAYERS reads as Auto (-1)."""
        with patch(
            "autoloop.detect_hardware_capabilities",
            return_value={"has_gpu": False, "physical_cores": 8, "ram_mb": 16384.0},
        ):
            out = autoloop.apply_cpu_preflight({"THREADS": 8})
        self.assertEqual(out["N_GPU_LAYERS"], 0)

    def test_filter_search_space_for_cpu_drops_speculative(self):
        space = {
            "SPEC_DRAFT_N_MAX": [0, 1, 2],
            "THREADS": [6, 8],
            "NUMA": [None, "distribute"],
        }
        out = autoloop.filter_search_space_for_cpu(space, 0)
        self.assertNotIn("SPEC_DRAFT_N_MAX", out)
        self.assertIn("THREADS", out)
        self.assertIn("NUMA", out)

    def test_filter_search_space_for_cpu_gpu_or_auto_keeps_speculative(self):
        space = {"SPEC_DRAFT_N_MAX": [0, 1, 2], "THREADS": [6, 8]}
        self.assertIn("SPEC_DRAFT_N_MAX", autoloop.filter_search_space_for_cpu(space, -1))
        self.assertIn("SPEC_DRAFT_N_MAX", autoloop.filter_search_space_for_cpu(space, 1))

    def test_filter_search_space_for_cpu_returns_copy(self):
        space = {"THREADS": [6, 8]}
        out = autoloop.filter_search_space_for_cpu(space, -1)
        out["THREADS"] = [999]
        self.assertEqual(space["THREADS"], [6, 8])

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=1000.0)
    @patch(
        "autoloop.detect_hardware_capabilities",
        return_value={"has_gpu": False, "physical_cores": 8, "ram_mb": 16384.0},
    )
    def test_main_cpu_preflight_seeds_baseline_and_filters_search_space(
        self,
        mock_detect,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """main() on a CPU host seeds N_GPU_LAYERS=0 and drops SPEC_DRAFT_N_MAX."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        captured = {}
        real = autoloop.SearchStrategy

        class Spy(real):
            def __init__(self, search_space, **kwargs):
                captured["search_space"] = search_space
                super().__init__(search_space, **kwargs)

        with patch("autoloop.SearchStrategy", Spy):
            with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
                with patch.object(SearchStrategy, "random_restart", return_value=None):
                    with patch("autoloop.SearchState.update_baseline") as mock_wcfg:
                        autoloop.main()

        # Preflight seeded N_GPU_LAYERS=0 into the Baseline via update_baseline.
        self.assertTrue(
            any(call.args[0].get("N_GPU_LAYERS") == 0 for call in mock_wcfg.call_args_list)
        )
        # Active Search Space excludes the GPU-only speculative knob.
        self.assertIn("SPEC_DRAFT_N_MAX", autoloop.SEARCH_SPACE)
        self.assertNotIn("SPEC_DRAFT_N_MAX", captured["search_space"])

    @patch("sys.argv", ["autoloop.py", "--max-rounds", "1", "--models", "test.gguf"])
    @patch("autoloop._available_gguf_names", return_value=["test.gguf"])
    @patch("autoloop.ExperimentRunner")
    @patch("autoloop.load_config")
    @patch("autoloop.get_git_commit", return_value="abc")
    @patch("autoloop.write_row")
    @patch("autoloop.estimate_vram_mb", return_value=1000.0)
    @patch(
        "autoloop.detect_hardware_capabilities",
        return_value={"has_gpu": True, "physical_cores": 8, "ram_mb": 16384.0},
    )
    def test_main_gpu_keeps_auto_and_full_search_space(
        self,
        mock_detect,
        mock_vram,
        mock_write_row,
        mock_git,
        mock_lcfg,
        mock_runner_cls,
        _mock_models,
    ):
        """main() with a GPU keeps Auto (-1) and the full Search Space."""
        mock_lcfg.return_value = self._full_config(MODEL="test.gguf")
        mock_runner = MagicMock()
        mock_runner.run_trial.return_value = self._make_trial_result()
        mock_runner_cls.return_value = mock_runner

        captured = {}
        real = autoloop.SearchStrategy

        class Spy(real):
            def __init__(self, search_space, **kwargs):
                captured["search_space"] = search_space
                super().__init__(search_space, **kwargs)

        with patch("autoloop.SearchStrategy", Spy):
            with patch.object(SearchStrategy, "get_neighbors", return_value=[]):
                with patch.object(SearchStrategy, "random_restart", return_value=None):
                    with patch("autoloop.SearchState.update_baseline") as mock_wcfg:
                        autoloop.main()

        # No N_GPU_LAYERS=0 seed on a GPU host.
        self.assertFalse(
            any(call.args[0].get("N_GPU_LAYERS") == 0 for call in mock_wcfg.call_args_list)
        )
        self.assertIn("SPEC_DRAFT_N_MAX", captured["search_space"])


class TestClimbFingerprint(unittest.TestCase):
    """TPS climb writes the ADR 0014 Fingerprint file on keep (issue #51)."""

    def _cfg(self, **over):
        cfg = {
            "MODEL": "climb-model.gguf",
            "CTX_SIZE": 65536,
            "N_GPU_LAYERS": -1,
            "KV_CACHE": "q4_0",
            "KV_CACHE_K": "q4_0",
            "KV_CACHE_V": "q4_0",
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
            "THREADS": 8,
            "FLASH_ATTN": "on",
            "CONT_BATCHING": True,
            "SPEC_DRAFT_N_MAX": 0,
            "TPS_FLOOR": 20.0,
            "VRAM_LIMIT_MB": 7900.0,
            "TEMP": 0.0,
            "TOP_P": 0.9,
        }
        cfg.update(over)
        return cfg

    def test_kept_tps_neighbor_writes_matching_engine_without_sampler(self):
        from autoresearch.core.fingerprint import load

        with tempfile.TemporaryDirectory() as tmp:
            path = autoloop.write_climb_fingerprint(
                "climb-model.gguf",
                self._cfg(),
                outcome=autoloop.TrialOutcome.OK,
                status="on_front",
                is_tps_climb=True,
                directory=tmp,
            )
            self.assertIsNotNone(path)
            loaded = load(path)
            self.assertEqual(loaded["model"], "climb-model.gguf")
            self.assertEqual(loaded["engine"]["CTX_SIZE"], 65536)
            self.assertEqual(loaded["engine"]["MODEL"], "climb-model.gguf")
            self.assertIsNone(loaded["sampler"])
            self.assertNotIn("TEMP", loaded["engine"])
            self.assertNotIn("TOP_P", loaded["engine"])

    def test_rejected_climb_keeps_good_file(self):
        from autoresearch.core.fingerprint import dump, load, path_for

        with tempfile.TemporaryDirectory() as tmp:
            good = path_for("climb-model.gguf", tmp)
            dump(good, model="climb-model.gguf", engine={"CTX_SIZE": 32768})
            out = autoloop.write_climb_fingerprint(
                "climb-model.gguf",
                self._cfg(),
                outcome=autoloop.TrialOutcome.OK,
                status="rejected",
                is_tps_climb=True,
                directory=tmp,
            )
            self.assertIsNone(out)
            self.assertEqual(load(good)["engine"]["CTX_SIZE"], 32768)

    def test_failed_outcome_keeps_good_file(self):
        from autoresearch.core.fingerprint import dump, load, path_for

        with tempfile.TemporaryDirectory() as tmp:
            good = path_for("climb-model.gguf", tmp)
            dump(good, model="climb-model.gguf", engine={"CTX_SIZE": 32768})
            out = autoloop.write_climb_fingerprint(
                "climb-model.gguf",
                self._cfg(),
                outcome=autoloop.TrialOutcome.INFRA_ERROR,
                status="rejected",
                is_tps_climb=True,
                directory=tmp,
            )
            self.assertIsNone(out)
            self.assertEqual(load(good)["engine"]["CTX_SIZE"], 32768)

    def test_non_tps_climb_skips_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = autoloop.write_climb_fingerprint(
                "climb-model.gguf",
                self._cfg(),
                outcome=autoloop.TrialOutcome.OK,
                status="on_front",
                is_tps_climb=False,
                directory=tmp,
            )
            self.assertIsNone(out)
            self.assertEqual(list(Path(tmp).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
