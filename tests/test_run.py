import csv
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.runners import run


class TestRun(unittest.TestCase):
    def test_mini_swe_agent_category_precedes_default_full(self):
        args = SimpleNamespace(
            validation=False,
            mini_swe_agent=True,
            agentic_full=True,
            agentic_coding=False,
            agentic_quick=False,
        )
        self.assertEqual(run.determine_category(args), "mini-swe-agent")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch(
        "autoresearch.runners.evaluation.preflight_vram_for_intent",
        return_value=(False, 8108.0, "VRAM_PREFLIGHT est=8108MB > limit=7900MB"),
    )
    def test_run_trial_rejects_peak_estimate_before_server(self, _mock_preflight, mock_server):
        from autoresearch.runners.evaluation import ExperimentRunner, TrialOutcome

        result = ExperimentRunner(Path("models")).run_trial(
            {"MODEL": "test.gguf", "CTX_SIZE": 131072, "FLASH_ATTN": "on"},
            skip_bench=True,
        )

        self.assertEqual(result.outcome, TrialOutcome.MODEL_REJECTED)
        self.assertEqual(result.diagnostic, "VRAM_PREFLIGHT est=8108MB > limit=7900MB")
        self.assertAlmostEqual(result.peak_vram_gb, 8108.0 / 1024.0)
        mock_server.assert_not_called()

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 1.0, 1.0, ""),
    )
    def test_run_trial_vram_headroom_rejects_with_both_budgets(self, _mock_host):
        """Issue #10: effective budget = free-at-start minus headroom (mocked free VRAM)."""
        from autoresearch.runners.evaluation import ExperimentRunner, TrialOutcome

        # Pin headroom + configured limit: machine Baseline may override both.
        with (
            patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=6000.0),
            patch("autoresearch.core.llama_runner.detect_total_vram_mb", return_value=None),
            patch("autoresearch.core.llama_runner.resolve_vram_headroom_mb", return_value=512.0),
            patch.dict(os.environ, {"AUTORESEARCH_VRAM_FREE_CLAMP": "1"}, clear=False),
        ):
            result = ExperimentRunner(Path("models")).run_trial(
                {
                    "MODEL": "test.gguf",
                    "CTX_SIZE": 131072,
                    "FLASH_ATTN": "on",
                    "VRAM_LIMIT_MB": 7900.0,
                },
                skip_bench=True,
            )

        self.assertEqual(result.outcome, TrialOutcome.MODEL_REJECTED)
        self.assertIn("effective=5488MB", result.diagnostic)
        self.assertIn("configured=7900MB", result.diagnostic)

    @patch(
        "autoresearch.runners.evaluation.detect_used_total_vram_mb", side_effect=FileNotFoundError
    )
    @patch("autoresearch.runners.evaluation.subprocess.Popen")
    @patch("autoresearch.runners.evaluation.resolve_llama_cli", return_value=Path("llama-cli.exe"))
    @patch("autoresearch.runners.evaluation.resolve_vram_limit_mb", return_value=7900.0)
    def test_llama_bench_caps_ctx(self, _mock_limit, mock_resolve, mock_popen, _mock_smi):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("Generation: 7.4 t/s", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        from autoresearch.runners.evaluation import BENCH_CTX_CAP, run_llama_bench_validation

        run_llama_bench_validation(Path("model.gguf"), ctx_size=65536)

        command = mock_popen.call_args.args[0]
        self.assertEqual(command[command.index("-c") + 1], str(BENCH_CTX_CAP))

    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=45.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    @patch("autoresearch.runners.run.get_git_commit")
    @patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=20000.0)
    def test_single_run_coding_only_classifies_incomplete(
        self, _mock_free, mock_commit, mock_coding, mock_runner, mock_bench
    ):
        """Issue #13: Trial Status, not scalar beat-the-previous-best. A coding-only
        Trial (agentic axis missing) is INCOMPLETE even though 0.75 > previous best."""
        import tempfile

        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        mock_commit.return_value = "abcdefg"

        # Mock coding result with all 4 benchmark fields
        mock_coding.return_value = BenchmarkResult(
            val_score=0.75, val_pass1=0.6, val_pass2=0.8, val_pass3=0.7, val_pass4=0.5, avg_tps=40.0
        )

        args = MagicMock()
        args.desc = "Tweak test prompt"
        args.model = "g4-opt-it-Q4_K_M.gguf"
        args.kv = "q4_0"
        args.max_tokens = 512
        args.ctx_size = 131072
        args.port = 18080
        args.threads = 12
        args.ngl = 99
        args.context_tokens = 8192
        args.include_coding = True
        args.grid = False

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            with patch.object(run, "RESULTS_FILE", path):
                with patch("sys.exit") as mock_exit:
                    run.handle_single_run(args)
                    mock_exit.assert_not_called()
            with open(path, encoding="utf-8") as f:
                row = next(csv.DictReader(f, delimiter="\t"))

        # Status reflects the Objective Vector, not a previous-best comparison.
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["val_score"], "0.750000")

    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    @patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=20000.0)
    def test_include_agentic_full_key_enables_claw(self, _mock_free, mock_coding, mock_runner):
        """bench_config INCLUDE_AGENTIC_FULL lowercases to include_agentic_full — must enable."""
        mock_runner.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4000)
        with patch("autoresearch.runners.evaluation.get_full_tier_tasks", return_value=["T002"]):
            with patch(
                "autoresearch.runners.evaluation.run_agentic_eval",
                return_value={"score": 0.8, "total": 1},
            ) as mock_agentic:
                res = run.run_evaluation(
                    {
                        "MODEL": "test.gguf",
                        "CTX_SIZE": 131072,
                        "FLASH_ATTN": "on",
                        "INCLUDE_CODING": False,
                        "INCLUDE_AGENTIC_QUICK": False,
                        "INCLUDE_AGENTIC_FULL": True,
                    },
                    skip_bench=True,
                )
        mock_coding.assert_not_called()
        mock_agentic.assert_called_once()
        self.assertEqual(res["agentic_val"], 0.8)
        self.assertEqual(res["val_score"], 0.8)
        self.assertEqual(res["tps_source"], "skipped")
        self.assertEqual(res["outcome"], "OK")

    @patch("autoresearch.runners.evaluation.run_mini_swe_agent_eval")
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=20000.0)
    def test_mini_swe_agent_forwards_generation_params(self, _mock_free, mock_runner, mock_mini):
        entered = MagicMock(
            port=18080,
            peak_vram_mb=4000,
            vram_killed=False,
            intent=MagicMock(host="127.0.0.1", model_path=Path("test.gguf")),
        )
        mock_runner.return_value.__enter__.return_value = entered
        mock_mini.return_value = {
            "score": 1.0,
            "passed": 1,
            "total": 1,
            "detail": "task=pass",
            "suite_signature": "sig",
        }
        result = run.run_evaluation(
            {
                "MODEL": "test.gguf",
                "CTX_SIZE": 131072,
                "FLASH_ATTN": "on",
                "TEMP": 0.33,
                "TOP_P": 0.88,
                "TOP_K": 17,
            },
            skip_bench=True,
            include_coding=True,
            agentic_full=True,
            mini_swe_agent=True,
        )
        self.assertEqual(result["mini_swe_agent_val"], 1.0)
        self.assertIn("suite=sig", result["mini_swe_agent_detail"])
        params = mock_mini.call_args.kwargs["gen_params"]
        self.assertEqual(params.temp, 0.33)
        self.assertEqual(params.top_p, 0.88)
        self.assertEqual(params.top_k, 17)

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 1000.0, 8000.0, ""),
    )
    @patch(
        "autoresearch.runners.evaluation.preflight_vram_for_intent", return_value=(True, 1000.0, "")
    )
    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=17.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    def test_default_tps_floor_rejects_moe_speed(
        self, mock_runner, mock_bench, _mock_vram, _mock_host
    ):
        """TPS Floor 20 rejects 17 t/s (typical MoE on 8GB)."""
        res = run.run_evaluation(
            {
                "MODEL": "gemma-4-26B-A4B.gguf",
                "CTX_SIZE": 65536,
                "FLASH_ATTN": "on",
                "TPS_FLOOR": 20.0,
            },
            include_coding=False,
            agentic_quick=False,
            agentic_full=False,
        )
        self.assertIn("FAIL: bench tg 17.0 < threshold 20.0", res["status"])
        self.assertEqual(res["outcome"], "MODEL_REJECTED")
        mock_runner.assert_not_called()

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 1000.0, 8000.0, ""),
    )
    @patch(
        "autoresearch.runners.evaluation.preflight_vram_for_intent", return_value=(True, 1000.0, "")
    )
    @patch("autoresearch.runners.evaluation.run_llama_bench_validation", return_value=16.0)
    def test_custom_tps_floor_rejects_below_floor(self, mock_bench, _mock_vram, _mock_host):
        """Custom TPS_FLOOR=18 rejects 16 t/s at the bench gate."""
        res = run.run_evaluation(
            {
                "MODEL": "moe.gguf",
                "CTX_SIZE": 65536,
                "FLASH_ATTN": "on",
                "TPS_FLOOR": 18.0,
            },
            include_coding=False,
            agentic_quick=False,
            agentic_full=False,
        )
        self.assertIn("FAIL: bench tg 16.0 < threshold 18.0", res["status"])
        self.assertEqual(res["outcome"], "MODEL_REJECTED")

    def test_unix_results_lock_unlocks_with_fcntl_lock_un(self):
        """POSIX unlock is fcntl.LOCK_UN (Windows never imports fcntl)."""
        import tempfile
        import types

        ops: list[int] = []
        fake = types.SimpleNamespace(
            LOCK_EX=2,
            LOCK_UN=8,
            flock=lambda _fd, op: ops.append(op),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            with (
                patch.object(run, "fcntl", fake, create=True),
                patch.object(run.sys, "platform", "linux"),
            ):
                with run._results_lock(path):
                    self.assertEqual(ops, [fake.LOCK_EX])
        self.assertEqual(ops, [fake.LOCK_EX, fake.LOCK_UN])

    def test_write_row_keeps_zeros_and_throughput_columns(self):
        """write_row must record legitimate zeros and throughput fields, not blank them."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            run.write_row(
                path,
                "abc123",
                0.5,
                0.0,
                0.1,
                0.2,
                4.5,
                "on_front",
                "desc",
                lcb_score=0.0,
                bigcode_score=0.0,
                category="10-task",
                elapsed_sec=12.0,
                model="m.gguf",
                tps=47.7,
                bench_tg=43.2,
                kv="q4_0",
                ctx=65536,
                threads=6,
                threads_batch=8,
                batch_size=256,
                ubatch_size=128,
                n_cpu_moe=0,
                temp=0.4,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                repeat_penalty=1.0,
                presence_penalty=0.0,
                cont_batching=True,
                flash_attn="on",
                no_mmap=True,
                spec_draft_n_max=0,
                tps_source="llama-bench",
            )
            with open(path, encoding="utf-8") as f:
                row = next(csv.DictReader(f, delimiter="\t"))

        self.assertEqual(row["tps"], "47.7")
        self.assertEqual(row["bench_tg"], "43.2")
        self.assertEqual(row["min_p"], "0.0")
        self.assertEqual(row["presence_penalty"], "0.0")
        self.assertEqual(row["n_cpu_moe"], "0")
        self.assertEqual(row["spec_draft_n_max"], "0")
        self.assertEqual(row["tps_source"], "llama-bench")

    def test_write_row_records_reasoning_columns(self):
        """write_row must persist reasoning_budget/reasoning_effort as flat columns."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            run.write_row(
                path,
                "abc123",
                0.5,
                0.0,
                0.1,
                0.2,
                4.5,
                "on_front",
                "desc",
                model="m.gguf",
                reasoning_budget=2048,
                reasoning_effort="low",
            )
            with open(path, encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split("\t")
            with open(path, encoding="utf-8") as f:
                row = next(csv.DictReader(f, delimiter="\t"))

        self.assertIn("reasoning_budget", header)
        self.assertIn("reasoning_effort", header)
        self.assertEqual(row["reasoning_budget"], "2048")
        self.assertEqual(row["reasoning_effort"], "low")

    @patch("autoresearch.runners.run.run_evaluation")
    @patch("autoresearch.runners.run.get_git_commit", return_value="abcdefg")
    def test_successful_single_run_logs_throughput_columns(self, mock_commit, mock_eval):
        """Successful single-run must populate tps/bench_tg/tps_source on the TSV row."""
        import tempfile

        mock_eval.return_value = {
            "status": "OK",
            "val_score": 0.55,
            "coding_val": 0.55,
            "lcb_val": 0.4,
            "he_val": 0.7,
            "mbpp_val": 0.6,
            "bigcode_val": 0.2,
            "swe_val": 0.0,
            "avg_tps": 47.7,
            "peak_vram_gb": 7.4,
            "bench_tg_tps": 43.2,
            "elapsed_sec": 90.0,
            "outcome": "OK",
            "diagnostic": "",
            "task_ids": ["he-1", "mbpp-2"],
            "tps_source": "llama-bench",
        }
        with patch("sys.argv", ["benchmark_search.py", "--desc", "throughput columns"]):
            args = run.parse_args()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            with patch.object(run, "RESULTS_FILE", path):
                run.handle_single_run(args)
            with open(path, encoding="utf-8") as f:
                row = next(csv.DictReader(f, delimiter="\t"))

        self.assertEqual(row["tps"], "47.7")
        self.assertEqual(row["bench_tg"], "43.2")
        self.assertEqual(row["tps_source"], "llama-bench")
        self.assertEqual(row["outcome"], "OK")
        self.assertEqual(row["task_ids"], "he-1,mbpp-2")

    @patch("autoresearch.runners.run.run_evaluation")
    @patch("autoresearch.runners.run.get_git_commit", return_value="abcdefg")
    def test_failed_single_run_logs_complete_baseline(self, mock_commit, mock_eval):
        mock_eval.return_value = {
            "status": "FAIL: bench tg 7.4 < threshold 20.0",
            "peak_vram_gb": 0.0,
            "elapsed_sec": 1.0,
            "outcome": "MODEL_REJECTED",
            "diagnostic": "slow",
        }
        with patch("sys.argv", ["benchmark_search.py", "--desc", "failed baseline"]):
            args = run.parse_args()

        with patch("autoresearch.runners.run.write_row") as mock_write:
            with self.assertRaises(SystemExit):
                run.handle_single_run(args)

        # Issue #13: hard failures write the `rejected` Trial Status, not scalar discard.
        self.assertEqual(mock_write.call_args.args[7], "rejected")
        recorded = json.loads(mock_write.call_args.kwargs["config_json"])
        self.assertEqual(recorded["model"], args.model)
        self.assertEqual(recorded["ctx_size"], args.ctx_size)
        self.assertEqual(recorded["batch_size"], args.batch_size)
        self.assertEqual(recorded["n_cpu_moe"], args.n_cpu_moe)
        self.assertEqual(recorded["temp"], run.config.TEMP)

    def test_moe_full_gpu_vram_reject_message(self):
        import tempfile

        from autoresearch.runners.evaluation import ExperimentRunner, ServerIntent, TrialOutcome

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            intent = ServerIntent(
                model_path=path,
                ctx_size=65536,
                kv_cache="q4_0",
                flash_attn="on",
                n_cpu_moe=0,
                n_cpu_moe_auto=False,
            )
            with (
                patch(
                    "autoresearch.runners.evaluation.ServerIntent.from_config",
                    return_value=(intent, {"vram_limit_mb": 7900}),
                ),
                patch(
                    "autoresearch.runners.evaluation.preflight_vram_for_intent",
                    return_value=(False, 16000.0, "VRAM_PREFLIGHT est=16000MB > limit=7900MB"),
                ),
                patch("autoresearch.runners.evaluation.gguf_is_moe", return_value=True),
            ):
                with patch("autoresearch.runners.evaluation.gguf_block_count", return_value=40):
                    res = ExperimentRunner(Path("models")).run_trial({"MODEL": "moe.gguf"})
            self.assertEqual(res.outcome, TrialOutcome.MODEL_REJECTED)
            self.assertIn("MoE full-GPU", res.status)
            self.assertIn("N_CPU_MOE=None", res.diagnostic)
        finally:
            path.unlink(missing_ok=True)

    def test_moe_vram_reject_only_when_est_exceeds_limit(self):
        """_moe_vram_reject must not fire when est is under the limit (issue #10 regression)."""
        from autoresearch.core.llama_runner import ServerIntent
        from autoresearch.runners.evaluation import _moe_vram_reject

        intent = ServerIntent(
            model_path=MagicMock(spec=Path),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=0,
        )
        intent.model_path.is_file.return_value = True
        with patch("autoresearch.runners.evaluation.gguf_is_moe", return_value=True):
            self.assertIsNone(_moe_vram_reject(intent, 6650.0, 7900.0))
            self.assertIsNotNone(_moe_vram_reject(intent, 8000.0, 7900.0))

    def test_vram_kill_during_enter_is_watchdog_kill(self):
        """Issue #72: runtime policy kills are WATCHDOG_KILL with scope, never MODEL_REJECTED."""
        from autoresearch.core.llama_runner import ServerIntent
        from autoresearch.runners.evaluation import ExperimentRunner, TrialOutcome

        intent = ServerIntent(
            model_path=Path("model.gguf"),
            ctx_size=100000,
            kv_cache="turbo3",
            flash_attn="on",
        )
        reason = "WATCHDOG_KILL scope=cuda_free cuda_free=50MB<floor=256MB (dense=model.gguf)"
        runner = MagicMock(vram_killed=True, peak_vram_mb=7978.0)
        runner.vram_kill_reason = reason
        runner.__enter__.side_effect = RuntimeError(reason)
        with (
            patch(
                "autoresearch.runners.evaluation.ServerIntent.from_config",
                return_value=(intent, {"vram_limit_mb": 7900}),
            ),
            patch(
                "autoresearch.runners.evaluation.preflight_vram_for_intent",
                return_value=(True, 6906.0, ""),
            ),
            patch(
                "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
                return_value=(True, 6906.0, 27790.0, ""),
            ),
            patch("autoresearch.runners.evaluation.LlamaServerRunner", return_value=runner),
        ):
            result = ExperimentRunner(Path("models")).run_trial({}, skip_bench=True)

        self.assertEqual(result.outcome, TrialOutcome.WATCHDOG_KILL)
        self.assertIn("cuda_free", result.diagnostic)
        self.assertTrue(result.status.startswith("FAIL: WATCHDOG_KILL"))
        self.assertNotEqual(result.outcome, TrialOutcome.MODEL_REJECTED)

    def test_vram_kill_legacy_message_maps_to_watchdog_kill(self):
        """Pre-fix RuntimeError('VRAM_LIMIT_EXCEEDED') without scope still lands honest."""
        from autoresearch.core.llama_runner import ServerIntent
        from autoresearch.runners.evaluation import ExperimentRunner, TrialOutcome

        intent = ServerIntent(
            model_path=Path("model.gguf"),
            ctx_size=100000,
            kv_cache="turbo3",
            flash_attn="on",
        )
        runner = MagicMock(vram_killed=True, peak_vram_mb=7978.0)
        runner.__enter__.side_effect = RuntimeError("VRAM_LIMIT_EXCEEDED")
        with (
            patch(
                "autoresearch.runners.evaluation.ServerIntent.from_config",
                return_value=(intent, {"vram_limit_mb": 7900}),
            ),
            patch(
                "autoresearch.runners.evaluation.preflight_vram_for_intent",
                return_value=(True, 6906.0, ""),
            ),
            patch(
                "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
                return_value=(True, 6906.0, 27790.0, ""),
            ),
            patch("autoresearch.runners.evaluation.LlamaServerRunner", return_value=runner),
        ):
            result = ExperimentRunner(Path("models")).run_trial({}, skip_bench=True)

        self.assertEqual(result.outcome, TrialOutcome.WATCHDOG_KILL)
        self.assertIn("WATCHDOG_KILL", result.diagnostic)


if __name__ == "__main__":
    unittest.main()
