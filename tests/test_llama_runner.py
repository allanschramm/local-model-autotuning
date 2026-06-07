import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import subprocess
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent, resolve_llama_server

class TestLlamaRunner(unittest.TestCase):

    def setUp(self):
        self.intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            port=18080
        )

    def test_resolve_llama_server_found(self):
        mock_cuda = MagicMock(spec=Path)
        mock_cuda.exists.return_value = True
        mock_cuda.__str__.return_value = "/fake/cuda"
        
        with patch("autoresearch.core.llama_runner.LLAMA_SERVER_CANDIDATES", (mock_cuda,)):
            path = resolve_llama_server()
            self.assertEqual(str(path), "/fake/cuda")

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
        self.assertEqual(cmd[0], "/bin/llama-server")
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
        self.assertIn("draft-mtp", cmd)

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
    def test_wait_for_server_timeout(self, _mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        # 0, 1, 2, ... 301
        mock_time.side_effect = range(310)
        mock_urlopen.side_effect = Exception("Not ready")
        
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.return_value = None
        
        self.assertFalse(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("subprocess.check_output")
    @patch("ctypes.CDLL")
    def test_vram_sampler(self, mock_cdll, mock_output, mock_resolve):
        mock_cdll.side_effect = Exception("Mock NVML load failure")
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_output.return_value = "1000\n"
        runner = LlamaServerRunner(self.intent)
        runner._start_vram_sampler()
        
        # Poll for peak_vram_mb to be set
        import time
        for _ in range(20):
            if runner.peak_vram_mb >= 1000:
                break
            time.sleep(0.1)
            
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

if __name__ == "__main__":
    unittest.main()