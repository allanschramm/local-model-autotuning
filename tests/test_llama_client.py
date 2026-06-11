import unittest
from unittest.mock import patch, MagicMock
from autoresearch.core.llama_client import LlamaClient
import json

class TestLlamaClient(unittest.TestCase):
    def setUp(self):
        self.client = LlamaClient(port=8080)

    def test_llama_client_init(self):
        self.assertEqual(self.client.port, 8080)
        self.assertEqual(self.client.base_url, "http://127.0.0.1:8080")

    @patch("urllib.request.urlopen")
    def test_complete_success(self, mock_urlopen):
        # Mock response
        mock_res = MagicMock()
        mock_res.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Hello world", "tool_calls": []}}],
            "usage": {"total_tokens": 15}
        }).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        response = self.client.complete("Say hello")

        self.assertEqual(response, {
            "content": "Hello world",
            "usage": {"total_tokens": 15},
            "choices": [{"message": {"content": "Hello world", "tool_calls": []}}]
        })
        mock_urlopen.assert_called_once()
        
        # Verify payload
        args, _ = mock_urlopen.call_args
        req = args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(req.get_method(), "POST")
        
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Say hello"}])
        self.assertEqual(payload["temperature"], 0.1)

    @patch("urllib.request.urlopen")
    def test_complete_with_tools(self, mock_urlopen):
        # Mock response with tool calls
        mock_res = MagicMock()
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}]
        mock_res.read.return_value = json.dumps({
            "choices": [{"message": {"content": "", "tool_calls": tool_calls}}],
            "usage": {"total_tokens": 25}
        }).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        response = self.client.complete("Use tool", tools=tools)

        self.assertEqual(response, {
            "content": "",
            "usage": {"total_tokens": 25},
            "choices": [{"message": {"content": "", "tool_calls": tool_calls}}]
        })
        
        # Verify payload contains tools
        args, _ = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["tools"], tools)

    @patch("urllib.request.urlopen")
    def test_complete_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        with self.assertRaisesRegex(RuntimeError, "LlamaClient request failed"):
            self.client.complete("Fail me")

if __name__ == "__main__":
    unittest.main()
