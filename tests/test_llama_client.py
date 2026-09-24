import json
import unittest
from unittest.mock import MagicMock, patch

from autoresearch.core.llama_client import LlamaClient


class TestLlamaClient(unittest.TestCase):
    def setUp(self):
        self.client = LlamaClient(port=8080)

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


if __name__ == "__main__":
    unittest.main()
