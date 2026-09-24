import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import autoloop
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


if __name__ == "__main__":
    unittest.main()
