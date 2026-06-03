import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import subprocess
import llama_runner
from llama_runner import LlamaServerRunner, ServerIntent, resolve_llama_server

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
        
        with patch("llama_runner.LLAMA_SERVER_CANDIDATES", (mock_cuda,)):
            path = resolve_llama_server()
            self.assertEqual(str(path), "/fake/cuda")

    def test_resolve_llama_server_not_found(self):
        mock_fail = MagicMock(spec=Path)
        mock_fail.exists.return_value = False
        
        with patch("llama_runner.LLAMA_SERVER_CANDIDATES", (mock_fail,)):
            with self.assertRaises(FileNotFoundError):
                resolve_llama_server()

    @patch("llama_runner.resolve_llama_server")
    def test_build_cmd_basic(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)
        cmd = runner._build_cmd(18080)
        self.assertEqual(cmd[0], "/bin/llama-server")
        self.assertIn("--port", cmd)
        self.assertIn("18080", cmd)

    @patch("llama_runner.resolve_llama_server")
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

    @patch("llama_runner.resolve_llama_server")
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

    @patch("llama_runner.resolve_llama_server")
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

    @patch("llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_for_server_timeout(self, mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        # 0, 1, 2, ... 301
        mock_time.side_effect = range(310)
        mock_urlopen.side_effect = Exception("Not ready")
        
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.return_value = None
        
        self.assertFalse(runner._wait_for_server(18080))

    @patch("llama_runner.resolve_llama_server")
    @patch("subprocess.check_output")
    def test_vram_sampler(self, mock_output, mock_resolve):
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

if __name__ == "__main__":
    unittest.main()