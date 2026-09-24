import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.core import llama_runner
from autoresearch.core.llama_runner import (
    LlamaServerRunner,
    ServerIntent,
    engine_version_tag,
)


class TestLlamaRunner(unittest.TestCase):
    def setUp(self):
        self.intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            port=18080,
        )

    def _check_binary_priority(self, prefer_gpu: bool) -> tuple[int, int]:
        with patch.object(llama_runner, "should_prefer_gpu_build", return_value=prefer_gpu):
            paths = llama_runner._candidate_binary(Path("root"), "llama-server")
            exe = llama_runner._exe("llama-server")
            cuda_idx = paths.index(Path(f"root/build-cuda/bin/{exe}"))
            cpu_idx = paths.index(Path(f"root/build-cpu/bin/{exe}"))
            return cuda_idx, cpu_idx

    def test_candidate_binary_priority_respects_hardware(self):
        cuda_idx, cpu_idx = self._check_binary_priority(prefer_gpu=False)
        self.assertLess(cpu_idx, cuda_idx)

        cuda_idx, cpu_idx = self._check_binary_priority(prefer_gpu=True)
        self.assertLess(cuda_idx, cpu_idx)

    def test_engine_version_tag_fork_release(self):
        server = Path(
            "D:/repo/llama.cpp-releases/turboquant/tqp-v0.3.0/build-cuda/bin/llama-server.exe"
        )
        self.assertEqual(engine_version_tag(server), "turboquant@tqp-v0.3.0")

    def test_engine_version_tag_stock_submodule(self):
        server = Path("D:/repo/llama.cpp/build-cuda/bin/llama-server.exe")
        self.assertEqual(engine_version_tag(server), "")

    def test_engine_version_tag_deep_nested_engine(self):
        server = Path("llama.cpp-releases/some-fork/v1.2.3-rc/bin/llama-server.exe")
        self.assertEqual(engine_version_tag(server), "some-fork@v1.2.3-rc")

    def test_engine_version_tag_unknown_path(self):
        self.assertEqual(engine_version_tag(Path("/usr/bin/llama-server")), "")

    def test_resolve_spec_estimate_args_uses_gguf_metadata(self):
        """MTP detection must read GGUF metadata, not the filename."""
        from autoresearch.core.llama_runner import resolve_spec_estimate_args

        with patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=True):
            spec, enabled, draft = resolve_spec_estimate_args(
                Path("models/no-mtp-in-name.gguf"), None, 1, None
            )
        self.assertEqual(spec, "mtp")
        self.assertTrue(enabled)

    def test_resolve_spec_estimate_args_ignores_filename_without_metadata(self):
        """A filename containing 'MTP' must not enable spec if metadata says no."""
        from autoresearch.core.llama_runner import resolve_spec_estimate_args

        with patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=False):
            spec, enabled, draft = resolve_spec_estimate_args(
                Path("models/Fake-MTP.gguf"), None, 1, None
            )
        self.assertIsNone(spec)
        self.assertFalse(enabled)

    def test_gguf_has_mtp_reads_nextn_key(self):
        from autoresearch.core import model_arch

        class FakeField:
            def __init__(self, v):
                self._v = v

            def contents(self):
                return self._v

        fake = MagicMock()
        fake.fields = {"nemotron_h_moe.nextn_predict_layers": FakeField(1)}
        with patch("gguf.GGUFReader", return_value=fake):
            self.assertTrue(model_arch.gguf_has_mtp(Path("models/mtp-meta-test.gguf")))

    def test_gguf_has_mtp_false_when_no_key(self):
        from autoresearch.core import model_arch

        class FakeField:
            def __init__(self, v):
                self._v = v

            def contents(self):
                return self._v

        fake = MagicMock()
        fake.fields = {"nemotron_h_moe.block_count": FakeField(53)}
        with patch("gguf.GGUFReader", return_value=fake):
            self.assertFalse(model_arch.gguf_has_mtp(Path("models/no-mtp-meta.gguf")))

    @patch("autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True)
    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("subprocess.check_output")
    @patch("ctypes.CDLL")
    def test_vram_sampler(self, mock_cdll, mock_output, mock_resolve, _mock_prefer_gpu):
        import threading

        called_event = threading.Event()

        def check_output_side_effect(*args, **kwargs):
            called_event.set()
            return "1000\n"

        mock_cdll.side_effect = Exception("Mock NVML load failure")
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_output.side_effect = check_output_side_effect

        runner = LlamaServerRunner(self.intent)
        runner._start_vram_sampler()

        # Robust event synchronization: wait until check_output gets called
        called_event.wait(5.0)

        runner._stop_event.set()
        runner._vram_thread.join()
        self.assertGreaterEqual(runner.peak_vram_mb, 1000)

    def test_estimate_vram_mb_includes_draft(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / "draft.gguf"
            draft.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MiB
            base = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0")
            with_draft = estimate_vram_mb(
                Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0", draft_path=draft
            )
            self.assertAlmostEqual(with_draft - base, 10.0, places=1)

    def test_estimate_vram_mb_includes_speculative_workspace_by_draft_window(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        base = estimate_vram_mb(Path("models/non-existent.gguf"), 2048)
        mtp_two = estimate_vram_mb(
            Path("models/non-existent.gguf"),
            2048,
            spec_type="draft-mtp",
            spec_draft_n_max=2,
        )
        mtp_four = estimate_vram_mb(
            Path("models/non-existent.gguf"),
            2048,
            spec_type="draft-mtp",
            spec_draft_n_max=4,
        )

        self.assertAlmostEqual(mtp_two - base, 1024.0)
        self.assertAlmostEqual(mtp_four - mtp_two, 512.0)

    def test_estimate_vram_mb_moe_external_draft_skips_spec_workspace(self):
        """MoE expert-CPU offload + any speculative draft: charge draft weights only.

        Flat speculative workspace (512 + 256*n) false-rejects DFlash on 8 GB
        when measured peaks are ~4 GB. Embedded MTP (no draft file) also skips the
        flat workspace — measured peaks are 3.6-4.2 GB vs 6.8-9.1 GB estimated.
        """
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            draft = Path(tmp) / "dflash.gguf"
            model.write_bytes(b"m" * (100 * 1024 * 1024))
            draft.write_bytes(b"d" * (10 * 1024 * 1024))

            with (
                patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True),
                patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40),
            ):
                base = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=40)
                with_dflash = estimate_vram_mb(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    n_cpu_moe=40,
                    draft_path=draft,
                    spec_type="draft-dflash",
                    spec_draft_n_max=15,
                )
                embedded_mtp = estimate_vram_mb(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    n_cpu_moe=40,
                    spec_type="draft-mtp",
                    spec_draft_n_max=2,
                )

            self.assertAlmostEqual(with_dflash - base, 10.0, places=1)
            self.assertAlmostEqual(embedded_mtp - base, 0.0, places=1)

    def test_estimate_vram_mb_n_cpu_moe_shrinks_weight(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                full = estimate_vram_mb(model, 2048, "q4_0", "q4_0")
                vitriol = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            self.assertLess(vitriol, full * 0.4)
            self.assertGreater(vitriol, 1000.0)

    def test_preflight_vram_rejects_over_limit(self):
        from autoresearch.core.llama_runner import preflight_vram

        ok, est, reason = preflight_vram(
            Path("models/non-existent.gguf"),
            131072,
            kv_cache_k="q4_0",
            kv_cache_v="q4_0",
            vram_limit_mb=1.0,
        )
        self.assertFalse(ok)
        self.assertGreater(est, 1.0)
        self.assertIn("VRAM_PREFLIGHT", reason)

    def test_preflight_vram_for_intent_accounts_for_configured_ctx_and_mtp(self):
        from autoresearch.core.llama_runner import preflight_vram_for_intent

        intent = ServerIntent(
            model_path=Path("models/embedded-MTP.gguf"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
            spec_type="draft-mtp",
            spec_draft_n_max=4,
        )

        ok, est, reason = preflight_vram_for_intent(intent, vram_limit_mb=8000.0)

        self.assertFalse(ok)
        self.assertGreater(est, 8000.0)
        self.assertIn("VRAM_PREFLIGHT", reason)

    def test_preflight_vram_for_intent_only_counts_enabled_external_draft(self):
        from autoresearch.core.llama_runner import preflight_vram_for_intent

        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / "draft.gguf"
            draft.write_bytes(b"x" * (10 * 1024 * 1024))
            common = dict(
                model_path=Path("models/base.gguf"),
                ctx_size=2048,
                kv_cache="q4_0",
                flash_attn="on",
                spec_draft_model=str(draft),
                spec_draft_n_max=2,
            )

            disabled = preflight_vram_for_intent(ServerIntent(**common), 10000.0)[1]
            enabled = preflight_vram_for_intent(ServerIntent(**common, spec_type="draft"), 10000.0)[
                1
            ]

        self.assertAlmostEqual(enabled - disabled, 1034.0, places=1)

    def test_preflight_vram_passes_large_moe_with_n_cpu_moe(self):
        from autoresearch.core.llama_runner import preflight_vram

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=14 * 1024 * 1024 * 1024)
            ):
                ok, est, reason = preflight_vram(
                    model,
                    65536,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    vram_limit_mb=7900.0,
                    n_cpu_moe=30,
                )
            self.assertTrue(ok, reason)
            self.assertLessEqual(est, 7900.0)

    def test_estimate_host_memory_shrinks_to_expert_bytes_when_offloaded(self):
        from autoresearch.core.llama_runner import estimate_host_memory_mb, estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            size = 10 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                host = estimate_host_memory_mb(model, 2048, "q4_0", "q4_0")
                host_off = estimate_host_memory_mb(
                    model, 2048, "q4_0", "q4_0", n_cpu_moe=32, unified=False
                )
                host_off_unified = estimate_host_memory_mb(
                    model, 2048, "q4_0", "q4_0", n_cpu_moe=32, unified=True
                )
                vram_off = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            # Discrete + offload: expert bytes only (header unreadable -> 0.72 fallback).
            self.assertLess(host_off, host)
            self.assertGreater(host_off, vram_off)
            # Unified hosts hold the whole model in RAM regardless of offload.
            self.assertAlmostEqual(host_off_unified, host, places=1)

    def test_preflight_host_rejects_12gb_on_16gb_unified(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "big.gguf"
            model.write_bytes(b"x")
            size = 12 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                ok, est, budget, reason = preflight_host_memory(
                    model,
                    2048,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    ram_mb=16384.0,
                    unified=True,
                )
            self.assertFalse(ok)
            self.assertIn("HOST_MEMORY_PREFLIGHT", reason)
            self.assertLess(budget, 12000.0)
            self.assertGreater(est, budget)

    def test_preflight_host_passes_when_under_budget(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "small.gguf"
            model.write_bytes(b"x")
            size = 2 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                ok, est, budget, reason = preflight_host_memory(
                    model,
                    2048,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    ram_mb=16384.0,
                    unified=True,
                )
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "")
            self.assertLessEqual(est, budget)

    def test_preflight_host_fail_closed_unified_unknown_ram(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=None):
            ok, est, budget, reason = preflight_host_memory(
                Path("models/non-existent.gguf"),
                2048,
                ram_mb=None,
                unified=True,
            )
        self.assertFalse(ok)
        self.assertIn("ram_unknown", reason)

    def test_preflight_host_discrete_unknown_ram_passes(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=None):
            ok, est, budget, reason = preflight_host_memory(
                Path("models/non-existent.gguf"),
                2048,
                ram_mb=None,
                unified=False,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_estimate_vram_offload_uses_gguf_block_count(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                with patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True):
                    with patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40):
                        full = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=40)
                        half = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=20)
            self.assertLess(full, half)

    def test_estimate_vram_offload_falls_back_to_32_ref(self):
        from autoresearch.core.llama_runner import VRAM_MOE_NON_EXPERT_FRAC, estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                with patch(
                    "autoresearch.core.llama_runner.gguf_is_moe",
                    side_effect=RuntimeError("no arch"),
                ):
                    # n=32 / fallback 32 → full expert offload → ~28% of file + kv + overhead
                    est = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            file_mb = 10 * 1024
            self.assertAlmostEqual(
                est,
                file_mb * VRAM_MOE_NON_EXPERT_FRAC + 300.0 + (2048 * 80.0 / 1024.0) * 0.28,
                delta=50.0,
            )

    def test_resolve_n_cpu_moe_auto_block_count(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, 41)):
                n, auto = model_arch.resolve_n_cpu_moe(path, None)
            self.assertEqual(n, 41)
            self.assertTrue(auto)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_n_cpu_moe_explicit_and_dense(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, 41)):
                n, auto = model_arch.resolve_n_cpu_moe(path, 0)
            self.assertEqual(n, 0)
            self.assertFalse(auto)
            with patch.object(model_arch, "_gguf_arch_info", return_value=(False, 22)):
                n, auto = model_arch.resolve_n_cpu_moe(path, None)
            self.assertIsNone(n)
            self.assertFalse(auto)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_n_cpu_moe_moe_without_block_count_fails(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, None)):
                with self.assertRaises(ValueError) as ctx:
                    model_arch.resolve_n_cpu_moe(path, None)
            self.assertIn("block_count", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_vram_sampler_kills_dense_on_cuda_free_exhausted(self):
        """Kill when CUDA-reported free memory drops below floor — not on NVML used."""
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("Bonsai-27B-Q1_0.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        # Force nvidia-smi path (no NVML)
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500,8192\n"):
                with patch("autoresearch.core.hardware.cuda_free_mb", return_value=50.0):
                    with patch(
                        "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                    ):
                        runner._start_vram_sampler()
                    import time

                    time.sleep(0.35)
                    runner._stop_event.set()
                    if runner._vram_thread:
                        runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()
        self.assertIn("cuda_free", runner.vram_kill_reason)

    def test_vram_sampler_kills_moe_on_cuda_free_exhausted(self):
        """MoE must die on CUDA-free exhaustion too — skipping enabled WDDM shared spill."""
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("POCKET-26B-Q4_K_M.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=30,
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            with patch("autoresearch.core.llama_runner.is_dense_model", return_value=False):
                runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500,8192\n"):
                with patch("autoresearch.core.hardware.cuda_free_mb", return_value=50.0):
                    with patch(
                        "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                    ):
                        runner._start_vram_sampler()
                    import time

                    time.sleep(0.35)
                    runner._stop_event.set()
                    if runner._vram_thread:
                        runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()
        self.assertIn("cuda_free", runner.vram_kill_reason)

    def test_vram_sampler_ignores_nvml_over_limit_when_cuda_healthy(self):
        """NVML used>ceil alone must NOT kill — committed desktop memory is evictable."""
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("Bonsai-27B-Q1_0.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500,8192\n"):
                with patch("autoresearch.core.hardware.cuda_free_mb", return_value=8000.0):
                    with patch(
                        "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                    ):
                        runner._start_vram_sampler()
                    import time

                    time.sleep(0.5)
                    runner._stop_event.set()
                    if runner._vram_thread:
                        runner._vram_thread.join(timeout=1.0)
        self.assertFalse(runner.vram_killed)
        proc.kill.assert_not_called()

    def test_resolve_vram_limit_clamps_to_physical(self):
        from autoresearch.core.llama_runner import resolve_vram_limit_mb

        with patch("autoresearch.core.llama_runner.detect_total_vram_mb", return_value=8188.0):
            # physical − keepout(256) = 7932
            self.assertEqual(resolve_vram_limit_mb(8600), 7932.0)
            self.assertEqual(resolve_vram_limit_mb(7900), 7900.0)
            self.assertEqual(resolve_vram_limit_mb(7000), 7000.0)


class TestKvCalibration(unittest.TestCase):
    """GGUF-derived KV cache sizing (sparse-GQA fix, measured 2026-08)."""

    class _FakeField:
        def __init__(self, value):
            self._value = value

        def contents(self):
            return self._value

    def _kv_bytes(self, fields):
        from autoresearch.core.model_arch import gguf_kv_bytes_per_token_f16

        fake_fields = {k: self._FakeField(v) for k, v in fields.items()}

        class FakeReader:
            def __init__(self, _path):
                self.fields = fake_fields

        with patch("gguf.GGUFReader", FakeReader):
            return gguf_kv_bytes_per_token_f16(Path("m.gguf"))

    def test_dense_scalar_head_count_kv(self):
        # llama default path: scalar kv heads on every layer
        b = self._kv_bytes(
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "llama.embedding_length": 4096,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
            }
        )
        self.assertEqual(b, 32 * 8 * (128 + 128))

    def test_sparse_gqa_per_layer_array(self):
        # LFM2.5-8B-A1B: head_count_kv is a per-layer array (8 on attn, 0 on conv)
        b = self._kv_bytes(
            {
                "general.architecture": "lfm2moe",
                "lfm2moe.block_count": 24,
                "lfm2moe.embedding_length": 2048,
                "lfm2moe.attention.head_count": 32,
                "lfm2moe.attention.head_count_kv": [
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    8,
                    0,
                    0,
                ],
            }
        )
        self.assertEqual(b, 48 * (64 + 64))

    def _kv_f16_mb(self, fields, ctx):
        from autoresearch.core.model_arch import gguf_kv_f16_mb

        fake_fields = {k: self._FakeField(v) for k, v in fields.items()}

        class FakeReader:
            def __init__(self, _path):
                self.fields = fake_fields

        with patch("gguf.GGUFReader", FakeReader):
            return gguf_kv_f16_mb(Path("m.gguf"), ctx)

    def test_gemma4_swa_charges_window_not_full_ctx(self):
        # 5 SWA layers + 1 full; window 1024; SWA dims 256; full dims 512
        pattern = [True, True, True, True, True, False]
        kv_heads = [8, 8, 8, 8, 8, 2]
        fields = {
            "general.architecture": "gemma4",
            "gemma4.block_count": 6,
            "gemma4.embedding_length": 2816,
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": kv_heads,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": pattern,
        }
        # SWA false-path (bytes/token × full ctx) must not apply
        self.assertIsNone(self._kv_bytes(fields))
        ctx = 65536
        mb = self._kv_f16_mb(fields, ctx)
        swa_cells = 5 * 8 * (256 + 256) * 1024
        full_cells = 1 * 2 * (512 + 512) * ctx
        expected = (swa_cells + full_cells) / (1024.0 * 1024.0)
        self.assertAlmostEqual(mb, expected, places=3)
        # Old bug: charge every layer at full ctx + full dims
        bogous = (5 * 8 * (512 + 512) * ctx + full_cells) / (1024.0 * 1024.0)
        self.assertLess(mb, bogous * 0.25)

    def test_non_swa_f16_mb_matches_bytes_per_token_times_ctx(self):
        fields = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
        b = self._kv_bytes(fields)
        mb = self._kv_f16_mb(fields, 65536)
        self.assertAlmostEqual(mb, 65536 * b / (1024.0 * 1024.0), places=6)

    @unittest.skipUnless(
        Path("models/FINAL-Bench/pocket-26b-gguf/POCKET-26B-Q4_K_M.gguf").exists(),
        "POCKET-26B GGUF not downloaded",
    )
    def test_real_pocket26_65k_moe_offload_fits_8gb(self):
        from autoresearch.core.llama_runner import estimate_vram_mb
        from autoresearch.core.model_arch import resolve_n_cpu_moe

        p = Path("models/FINAL-Bench/pocket-26b-gguf/POCKET-26B-Q4_K_M.gguf")
        n, _ = resolve_n_cpu_moe(p, None)
        # Measured claw peak ~4.5GB @65k; old estimator 8548MB false-rejected.
        est = estimate_vram_mb(p, 65536, "q4_0", "q4_0", n_cpu_moe=n)
        self.assertLess(est, 7900.0)
        self.assertGreater(est, 3500.0)

    @unittest.skipUnless(
        Path("models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf").exists(),
        "LFM2.5-8B-A1B GGUF not downloaded",
    )
    def test_real_lfm_file_matches_measured_kv(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        p = Path("models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf")
        # est 65k q4_0 ~ 5324MB vs measured load 5399 / peak 5638 (2026-08)
        est = estimate_vram_mb(p, 65536, "q4_0", "q4_0")
        self.assertLess(est, 5600.0)
        self.assertGreater(est, 5000.0)


class TestVramHeadroomPreflight(unittest.TestCase):
    """Issue #10: dynamic VRAM headroom from free-at-start."""

    def test_effective_limit_caps_by_free_minus_headroom(self):
        limit = llama_runner.effective_vram_limit_mb(7900.0, free_vram_mb=6000.0, headroom_mb=512.0)
        self.assertEqual(limit, 5488.0)

    def test_effective_limit_uses_configured_when_free_unknown(self):
        limit = llama_runner.effective_vram_limit_mb(7900.0, free_vram_mb=None, headroom_mb=512.0)
        self.assertEqual(limit, 7900.0)

    def test_effective_limit_never_exceeds_configured(self):
        limit = llama_runner.effective_vram_limit_mb(4000.0, free_vram_mb=6000.0, headroom_mb=0.0)
        self.assertEqual(limit, 4000.0)

    def test_preflight_effective_free_clamp_opt_in_rejects_and_records_both_budgets(self):
        with (
            patch.object(llama_runner, "detect_free_vram_mb", return_value=6000.0),
            patch.object(llama_runner, "detect_total_vram_mb", return_value=None),
            patch.dict(os.environ, {"AUTORESEARCH_VRAM_FREE_CLAMP": "1"}, clear=False),
        ):
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                131072,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertFalse(ok)
        self.assertIn("effective=5488MB", reason)
        self.assertIn("configured=7900MB", reason)
        self.assertIn("free=6000MB", reason)

    def test_preflight_effective_passes_when_configured_binds(self):
        # free - headroom far above configured -> configured wins, no rewrite
        with patch.object(llama_runner, "detect_free_vram_mb", return_value=20000.0):
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                2048,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_preflight_effective_default_ignores_free_at_start(self):
        """Free clamp is opt-in (operator decision 2026-09-04): dense budget = configured."""
        with (
            patch.dict(os.environ, {"AUTORESEARCH_VRAM_FREE_CLAMP": ""}, clear=False),
            patch.object(llama_runner, "detect_free_vram_mb", return_value=6000.0),
            patch.object(llama_runner, "detect_total_vram_mb", return_value=None),
        ):
            self.assertFalse(llama_runner.free_vram_clamp_enabled())
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                131072,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertTrue(ok, f"est={est} reason={reason!r}")
        self.assertEqual(reason, "")

    def test_preflight_for_intent_uses_free_vram_when_clamped(self):
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with (
            patch.object(llama_runner, "detect_free_vram_mb", return_value=5000.0),
            patch.object(llama_runner, "detect_total_vram_mb", return_value=None),
            patch.dict(os.environ, {"AUTORESEARCH_VRAM_FREE_CLAMP": "1"}, clear=False),
        ):
            ok, _, reason = llama_runner.preflight_vram_for_intent(
                intent, 7900.0, headroom_mb=512.0
            )
        self.assertFalse(ok)
        self.assertIn("effective=4488MB", reason)
        self.assertIn("configured=7900MB", reason)

    def test_preflight_moe_offload_skips_free_vram_clamp(self):
        """MoE n_cpu_moe>0 uses configured budget; free clamp would false-reject."""
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with (
                patch("pathlib.Path.stat", return_value=MagicMock(st_size=14 * 1024 * 1024 * 1024)),
                patch.object(llama_runner, "detect_free_vram_mb", return_value=5000.0),
                patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True),
                patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40),
            ):
                ok, est, reason = llama_runner.preflight_vram_effective(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    vram_limit_mb=7900.0,
                    n_cpu_moe=40,
                    headroom_mb=512.0,
                )
        self.assertTrue(ok, f"expected pass, est={est} reason={reason!r}")
        self.assertEqual(reason, "")
        self.assertLess(est, 7900.0)

    def test_is_ngram_spec_type(self):
        self.assertTrue(llama_runner.is_ngram_spec_type("ngram-cache"))
        self.assertTrue(llama_runner.is_ngram_spec_type("ngram-simple"))
        self.assertTrue(llama_runner.is_ngram_spec_type("draft-mtp,ngram-mod"))
        self.assertFalse(llama_runner.is_ngram_spec_type("draft-mtp"))
        self.assertFalse(llama_runner.is_ngram_spec_type("none"))
        self.assertFalse(llama_runner.is_ngram_spec_type(None))

    def test_is_pure_ngram_spec_type(self):
        self.assertTrue(llama_runner.is_pure_ngram_spec_type("ngram-cache"))
        self.assertTrue(llama_runner.is_pure_ngram_spec_type("ngram-simple"))
        self.assertFalse(llama_runner.is_pure_ngram_spec_type("draft-mtp,ngram-mod"))
        self.assertFalse(llama_runner.is_pure_ngram_spec_type("draft-mtp"))
        self.assertFalse(llama_runner.is_pure_ngram_spec_type("none"))
        self.assertFalse(llama_runner.is_pure_ngram_spec_type(None))

    def test_is_spec_enabled(self):
        self.assertTrue(llama_runner.is_spec_enabled("ngram-cache", 0))
        self.assertTrue(llama_runner.is_spec_enabled("draft-mtp", 2))
        self.assertFalse(llama_runner.is_spec_enabled("draft-mtp", 0))
        self.assertFalse(llama_runner.is_spec_enabled("none", 4))
        self.assertFalse(llama_runner.is_spec_enabled(None, 0))

    def test_estimate_vram_pure_ngram_zero_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"x")
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=4 * 1024 * 1024 * 1024)):
                base_est = llama_runner.estimate_vram_mb(
                    model, 2048, "q4_0", "q4_0", spec_type=None
                )
                ngram_est = llama_runner.estimate_vram_mb(
                    model, 2048, "q4_0", "q4_0", spec_type="ngram-cache", spec_draft_n_max=0
                )
            self.assertEqual(base_est, ngram_est)


if __name__ == "__main__":
    unittest.main()
