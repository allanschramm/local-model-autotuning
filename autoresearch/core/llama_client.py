import json
import urllib.request
import time
from typing import Any, Dict

class LlamaClient:
    """Deep module for llama-server communication."""
    def __init__(self, port: int, timeout: int = 600):
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://127.0.0.1:{port}"

    def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": kwargs.get("maxtok", 512),
            "temperature": kwargs.get("temp", 0.1),
            "stream": False,
            "stop": kwargs.get("stop", ["</s>", "Instruction:", "User:", "Task:"])
        }
        
        # Forward additional generation parameters
        for key in ["top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty"]:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                raw_res = json.loads(res.read().decode())
                # Map to OAI format for BenchmarkHarness compatibility
                return {
                    "content": raw_res.get("content", ""),
                    "usage": {
                        "total_tokens": raw_res.get("tokens_predicted", 0)
                    },
                    "choices": [{
                        "message": {
                            "content": raw_res.get("content", ""),
                            "tool_calls": [] # Handled by regex usually or specific logic
                        }
                    }]
                }
        except Exception as e:
            raise RuntimeError(f"LlamaClient request failed: {e}")