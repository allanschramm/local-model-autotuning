import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import subprocess
import tempfile
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent, resolve_llama_server
from autoresearch.core import llama_runner

class TestLlamaRunner(unittest.TestCase):

    def setUp(self):
        self.intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            port=18080
        )

    def test_candidate_binary_includes_windows_release_exe(self):
        with patch.object(llama_runner, "IS_WINDOWS", True):
            paths = llama_runner._candidate_binary(Path("root"), "llama-server")

        self.assertIn(Path("root/build-cuda/bin/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cuda/bin/Release/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cpu/bin/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cpu/bin/Release/llama-server.exe"), paths)
        self.assertIn(Path("root/build/bin/Release/llama-server.exe"), paths)

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

    def test_resolve_llama_server_found(self):
        mock_cuda = MagicMock(spec=Path)
        mock_cuda.exists.return_value = True
        mock_cuda.__str__.return_value = "/fake/cuda"
        mock_cuda.resolve.return_value = Path("/fake/cuda")
        mock_cuda.absolute.return_value = Path("/fake/cuda")
        
        with patch("autoresearch.core.llama_runner.LLAMA_SERVER_CANDIDATES", (mock_cuda,)):
            path = resolve_llama_server()
            self.assertEqual(path, Path("/fake/cuda"))

    def test_resolve_llama_server_not_found(self):
        mock_fail = MagicMock(spec=Path)
        mock_fail.exists.return_value = False
        
        with patch("autoresearch.core.llama_runner.LLAMA_SERVER_CANDIDATES", (mock_fail,)):
            with self.assertRaises(FileNotFoundError):
                resolve_llama_server()

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_basic(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)
        cmd = runner._build_cmd(18080)
        self.assertEqual(Path(cmd[0]), Path("/bin/llama-server"))
        self.assertIn("--port", cmd)
        self.assertIn("18080", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_mtp(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/Gemma-MTP.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on"
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--spec-type", cmd)
        self.assertIn("mtp", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_vitriol_moe(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/DeepSeek-V3-MoE-A3B.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=40,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "40")
        self.assertNotIn("--override-tensor", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_vitriol_moe_full_gpu(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/LFM2.5-8B-A1B.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=0,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "0")

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(40, True))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_auto_n_cpu_moe_from_block_count(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/moe.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "moe.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": None,
            },
            Path("models"),
        )
        self.assertEqual(intent.n_cpu_moe, 40)
        self.assertTrue(intent.n_cpu_moe_auto)

    @patch("autoresearch.core.config.is_dense_model", return_value=False)
    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(0, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_keeps_explicit_zero(self, mock_path, mock_resolve_n, _mock_dense):
        mock_path.return_value = Path("models/moe.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "moe.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": 0,
            },
            Path("models"),
        )
        self.assertEqual(intent.n_cpu_moe, 0)
        self.assertFalse(intent.n_cpu_moe_auto)
        mock_resolve_n.assert_called_once()
        self.assertEqual(mock_resolve_n.call_args.args[1], 0)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_traditional_speculative(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/qwen35-9b.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            spec_type="draft-dflash",
            spec_draft_model="qwen35-9b-dflash-Q4_K_M.gguf",
            spec_draft_n_max=2
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--spec-type", cmd)
        self.assertIn("draft-dflash", cmd)
        self.assertIn("--spec-draft-model", cmd)
        expected_draft_path = Path("models/qwen35-9b-dflash-Q4_K_M.gguf")
        self.assertIn(str(expected_draft_path), cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    def test_wait_for_server_success(self, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_time.side_effect = [0, 1]
        mock_res = MagicMock()
        mock_res.status = 200
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res
        
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.return_value = None
        
        self.assertTrue(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_for_server_crash(self, _mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_urlopen.side_effect = Exception("Not ready")
        
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.side_effect = [None, 1]
        
        self.assertFalse(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_for_server_backoff(self, mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_urlopen.side_effect = Exception("Not ready")
        
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.side_effect = [None, None, None, 1]
        
        self.assertFalse(runner._wait_for_server(18080))
        
        # Verify backoff values
        sleep_args = [args[0] for args, kwargs in mock_sleep.call_args_list]
        self.assertEqual(sleep_args, [0.05, 0.1, 0.2])

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("subprocess.check_output")
    @patch("ctypes.CDLL")
    def test_vram_sampler(self, mock_cdll, mock_output, mock_resolve):
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

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_advanced_tuning(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            kv_cache_k="f16",
            kv_cache_v="q4_0",
            threads_batch=16,
            spec_draft_n_max=2
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        
        # Verify Key/Value KV cache parameters
        self.assertIn("--cache-type-k", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "f16")
        self.assertIn("--cache-type-v", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q4_0")
        
        # Verify Threads Batch parameters
        self.assertIn("--threads-batch", cmd)
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "16")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_mtp_advanced_tuning(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/Gemma-MTP.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            kv_cache_k="q8_0",
            kv_cache_v="q4_0",
            spec_draft_n_max=3
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        
        # Verify MTP advanced spec settings
        self.assertIn("--spec-draft-n-max", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "3")
        self.assertIn("--spec-draft-type-k", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-type-k") + 1], "q8_0")
        self.assertIn("--spec-draft-type-v", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-type-v") + 1], "q4_0")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_extra_flags(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            no_mmap=True,
            jinja=True,
            reasoning_budget=1024,
            reasoning_budget_message="Thinking budget reached. Proceed to final answer now."
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        
        self.assertIn("--no-mmap", cmd)
        self.assertIn("--jinja", cmd)
        self.assertIn("--reasoning-budget", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "1024")
        self.assertIn("--reasoning-budget-message", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-budget-message") + 1], "Thinking budget reached. Proceed to final answer now.")

    def test_estimate_vram_mb(self):
        from autoresearch.core.llama_runner import (
            estimate_vram_mb,
            VRAM_KB_PER_TOKEN_F16,
            VRAM_OVERHEAD_MB,
            VRAM_DEFAULT_QUANT_FACTOR,
            VRAM_QUANT_FACTORS
        )
        self.assertEqual(VRAM_KB_PER_TOKEN_F16, 80.0)
        self.assertEqual(VRAM_OVERHEAD_MB, 300.0)
        self.assertEqual(VRAM_DEFAULT_QUANT_FACTOR, 0.3)
        self.assertEqual(VRAM_QUANT_FACTORS["q4"], 0.28)

        # Test with 4 arguments (backward-compatibility check)
        v1 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0")
        self.assertGreater(v1, 4000)
        
        # Test with 5 arguments
        v2 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0", "q4_0")
        self.assertEqual(v1, v2)
        
        # Test default/none cache parameters
        v3 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048)
        self.assertEqual(v1, v3)

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

    def test_estimate_vram_mb_n_cpu_moe_shrinks_weight(self):
        from autoresearch.core.llama_runner import estimate_vram_mb
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x" * (10 * 1024 * 1024 * 1024))  # 10 GiB
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

    def test_preflight_vram_passes_large_moe_with_n_cpu_moe(self):
        from autoresearch.core.llama_runner import preflight_vram
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x" * (14 * 1024 * 1024 * 1024))  # 14 GiB file
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

    def test_estimate_vram_offload_uses_gguf_block_count(self):
        from autoresearch.core.llama_runner import estimate_vram_mb
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x" * (10 * 1024 * 1024 * 1024))
            with patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True):
                with patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40):
                    full = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=40)
                    half = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=20)
            self.assertLess(full, half)

    def test_estimate_vram_offload_falls_back_to_32_ref(self):
        from autoresearch.core.llama_runner import estimate_vram_mb, VRAM_MOE_NON_EXPERT_FRAC
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x" * (10 * 1024 * 1024 * 1024))
            with patch("autoresearch.core.llama_runner.gguf_is_moe", side_effect=RuntimeError("no arch")):
                # n=32 / fallback 32 → full expert offload → ~28% of file + kv + overhead
                est = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            file_mb = 10 * 1024
            self.assertAlmostEqual(est, file_mb * VRAM_MOE_NON_EXPERT_FRAC + 300.0 + (2048 * 80.0 / 1024.0) * 0.28, delta=50.0)

    def test_dense_n_cpu_moe_rejected(self):
        from autoresearch.core.config import validate_config, ConfigError
        with self.assertRaises(ConfigError) as ctx:
            validate_config({
                "MODEL": "Bonsai-27B-Q1_0.gguf",
                "CTX_SIZE": 65536,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": 32,
            })
        self.assertIn("MoE-only", str(ctx.exception))

    def test_moe_n_cpu_moe_allowed(self):
        from autoresearch.core.config import validate_config
        cfg = validate_config({
            "MODEL": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
            "CTX_SIZE": 65536,
            "FLASH_ATTN": "on",
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
            "N_CPU_MOE": 32,
            "VRAM_LIMIT_MB": 7900,
        })
        self.assertEqual(cfg["N_CPU_MOE"], 32)

    def test_ornith_moe_via_gguf_not_filename(self):
        from autoresearch.core.config import validate_config
        from autoresearch.core.model_arch import is_moe_model
        # Filename has no A3B/MOE token — classification must come from GGUF.
        self.assertTrue(is_moe_model("Ornith-1.0-35B-UD-Q4_K_XL.gguf"))
        self.assertTrue(is_moe_model("Laguna-XS-2.1-Q3_K_XL.gguf"))
        cfg = validate_config({
            "MODEL": "Ornith-1.0-35B-UD-Q4_K_XL.gguf",
            "CTX_SIZE": 65536,
            "FLASH_ATTN": "on",
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
            "N_CPU_MOE": 32,
            "VRAM_LIMIT_MB": 7900,
        })
        self.assertEqual(cfg["N_CPU_MOE"], 32)

    def test_missing_gguf_treated_dense_for_n_cpu_moe(self):
        from autoresearch.core.config import validate_config, ConfigError
        with self.assertRaises(ConfigError) as ctx:
            validate_config({
                "MODEL": "Totally-Fake-MoE-A3B.gguf",
                "CTX_SIZE": 65536,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": 32,
            })
        self.assertIn("MoE-only", str(ctx.exception))

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

    def test_resolve_n_cpu_moe_missing_file_skips_auto(self):
        from autoresearch.core import model_arch
        n, auto = model_arch.resolve_n_cpu_moe(Path("missing-moe.gguf"), None)
        self.assertIsNone(n)
        self.assertFalse(auto)
        n, auto = model_arch.resolve_n_cpu_moe(Path("missing-moe.gguf"), 32)
        self.assertEqual(n, 32)
        self.assertFalse(auto)

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

    def test_resolve_n_cpu_moe_unreadable_file_fails_auto(self):
        from autoresearch.core import model_arch
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", side_effect=OSError("bad gguf")):
                with self.assertRaises(ValueError) as ctx:
                    model_arch.resolve_n_cpu_moe(path, None)
            self.assertIn("cannot read GGUF", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_vram_sampler_kills_dense_over_limit(self):
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
        intent = ServerIntent(
            model_path=Path("Bonsai-27B-Q1_0.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with patch("autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")):
            runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        # Force nvidia-smi path (no NVML)
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500\n"):
                runner._start_vram_sampler()
                # Allow sampler thread to fire once
                import time
                time.sleep(0.35)
                runner._stop_event.set()
                if runner._vram_thread:
                    runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()

if __name__ == "__main__":
    unittest.main()
