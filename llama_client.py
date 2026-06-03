import json
import urllib.request
import time
from typing import Any, Dict

class LlamaClient:
    """Deep module for llama-server communication."""
    def __init__(self, port: int, timeout: int = 180):
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
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode())
        except Exception as e:
            raise RuntimeError(f"LlamaClient request failed: {e}")