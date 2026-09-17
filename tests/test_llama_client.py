import json
import unittest
from unittest.mock import MagicMock, patch

from autoresearch.core.llama_client import LlamaClient


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
        mock_res.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Hello world", "tool_calls": []}}],
                "usage": {"total_tokens": 15},
            }
        ).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        response = self.client.complete("Say hello")

        self.assertEqual(
            response,
            {
                "content": "Hello world",
                "reasoning_content": "",
                "usage": {"total_tokens": 15},
                "choices": [
                    {
                        "message": {
                            "content": "Hello world",
                            "reasoning_content": "",
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )
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
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }
        ]
        mock_res.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "", "tool_calls": tool_calls}}],
                "usage": {"total_tokens": 25},
            }
        ).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        response = self.client.complete("Use tool", tools=tools)

        self.assertEqual(
            response,
            {
                "content": "",
                "reasoning_content": "",
                "usage": {"total_tokens": 25},
                "choices": [
                    {"message": {"content": "", "reasoning_content": "", "tool_calls": tool_calls}}
                ],
            },
        )

        # Verify payload contains tools
        args, _ = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["tools"], tools)

    @patch("urllib.request.urlopen")
    def test_complete_custom_params(self, mock_urlopen):
        mock_res = MagicMock()
        mock_res.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "Custom", "tool_calls": []}}],
                "usage": {"total_tokens": 10},
            }
        ).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        self.client.complete("Say hello", max_tokens=100, temp=0.5)

        args, _ = mock_urlopen.call_args
        req = args[0]
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.5)

    @patch("urllib.request.urlopen")
    def test_complete_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        with self.assertRaisesRegex(RuntimeError, "LlamaClient request failed"):
            self.client.complete("Fail me")

    @patch("urllib.request.urlopen")
    def test_complete_model_specific_stop_token(self, mock_urlopen):
        """Verify model-specific stop token <|ifm|im_end|> is respected in complete payload."""
        mock_res = MagicMock()
        mock_res.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        ).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        # Via stop in complete kwargs as list
        self.client.complete("prompt", stop=["<|ifm|im_end|>"])
        args, _ = mock_urlopen.call_args
        payload = json.loads(args[0].data.decode())
        self.assertEqual(payload["stop"], ["<|ifm|im_end|>"])

        # Via stop in complete kwargs as single string (normalized to list)
        self.client.complete("prompt", stop="<|ifm|im_end|>")
        args, _ = mock_urlopen.call_args
        payload = json.loads(args[0].data.decode())
        self.assertEqual(payload["stop"], ["<|ifm|im_end|>"])

        # Via GenerationParams with list
        from autoresearch.core.llama_client import GenerationParams

        gen = GenerationParams(stop=["<|ifm|im_end|>", "</s>"], reasoning_effort="low")
        self.client.complete("prompt", gen=gen)
        args, _ = mock_urlopen.call_args
        payload = json.loads(args[0].data.decode())
        self.assertEqual(payload["stop"], ["<|ifm|im_end|>", "</s>"])
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(payload["chat_template_args"], {"reasoning_effort": "low"})

        # Via GenerationParams with single string (normalized to list in to_payload)
        gen_str = GenerationParams(stop="<|ifm|im_end|>")
        self.assertEqual(gen_str.to_payload()["stop"], ["<|ifm|im_end|>"])

    @patch("urllib.request.urlopen")
    def test_complete_enforces_420s_timeout_floor(self, mock_urlopen):
        """Verify 420s turn timeout floor is enforced even if a lower timeout is requested."""
        mock_res = MagicMock()
        mock_res.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 5}}
        ).encode()
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        # Request 120s timeout on client with default timeout
        self.client.complete("prompt", timeout=120.0)
        _, kwargs = mock_urlopen.call_args
        self.assertGreaterEqual(kwargs.get("timeout"), 420.0)

        # Client initialized with sub-floor timeout
        client_low = LlamaClient(port=8080, timeout=30.0)
        client_low.complete("prompt")
        _, kwargs = mock_urlopen.call_args
        self.assertGreaterEqual(kwargs.get("timeout"), 420.0)

        # Client allows timeout >= 420.0s
        client_high = LlamaClient(port=8080, timeout=600.0)
        client_high.complete("prompt")
        _, kwargs = mock_urlopen.call_args
        self.assertEqual(kwargs.get("timeout"), 600.0)

        # Client and complete handle None timeout safely
        client_none = LlamaClient(port=8080, timeout=None)
        self.assertGreaterEqual(client_none.timeout, 420.0)
        client_none.complete("prompt", timeout=None)
        _, kwargs = mock_urlopen.call_args
        self.assertGreaterEqual(kwargs.get("timeout"), 420.0)


if __name__ == "__main__":
    unittest.main()
