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
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "stream": False,
            "stop": kwargs.get("stop", ["</s>", "Instruction:", "User:", "Task:"])
        }
        
        # Forward tools if present
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
            
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
                choices = raw_res.get("choices", [])
                choice = choices[0] if (choices and isinstance(choices[0], dict)) else {}
                message = choice.get("message", {})
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls") or []
                usage = raw_res.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                
                return {
                    "content": content,
                    "usage": {
                        "total_tokens": total_tokens
                    },
                    "choices": [{
                        "message": {
                            "content": content,
                            "tool_calls": tool_calls
                        }
                    }]
                }
        except Exception as e:
            raise RuntimeError(f"LlamaClient request failed: {e}")