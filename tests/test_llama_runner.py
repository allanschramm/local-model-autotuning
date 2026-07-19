import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import subprocess
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
        self.assertIn(Path("root/build/bin/Release/llama-server.exe"), paths)

    def test_resolve_llama_server_found(self):
        mock_cuda = MagicMock(spec=Path)
        mock_cuda.exists.return_value = True
        mock_cuda.__str__.return_value = "/fake/cuda"
        mock_cuda.resolve.return_value = Path("/fake/cuda")
        
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
            flash_attn="on"
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--override-tensor", cmd)
        self.assertIn(".*exps.*=CPU", cmd)

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

if __name__ == "__main__":
    unittest.main()
