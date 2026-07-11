import json
import urllib.request
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List


@dataclass
class GenerationParams:
    """Typed container for generation parameters.

    Replaces the 5-file **kwargs passthrough chain with a single typed object.
    Each layer passes gen_params through instead of manually constructing,
    filtering, and forwarding kwargs dicts.
    """
    temp: float = 0.1
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int = 512
    stop: list[str] | None = None

    def to_payload(self) -> dict:
        """Build API payload dict, omitting None fields."""
        d: dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temp,
        }
        if self.stop is not None:
            d["stop"] = self.stop
        for key in ("top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    def with_overrides(self, **overrides) -> 'GenerationParams':
        """Return a new instance with the given fields overridden."""
        return replace(self, **overrides)


class LlamaClient:
    """Deep module for llama-server communication."""
    def __init__(self, port: int):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"

    def complete(self, prompt: str, gen: GenerationParams | None = None, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"

        if gen is not None:
            payload = gen.to_payload()
        else:
            payload = {
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temp", 0.1),
            }
            for key in ["top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty"]:
                if key in kwargs and kwargs[key] is not None:
                    payload[key] = kwargs[key]

        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = False

        # Stop tokens: from gen, from kwargs, or default
        if "stop" not in payload:
            payload["stop"] = kwargs.get("stop", ["</s>", "Instruction:", "User:"])

        # Forward tools if present
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as res:
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
                    "reasoning_content": message.get("reasoning_content") or "",
                    "usage": {
                        "total_tokens": total_tokens
                    },
                    "choices": [{
                        "message": {
                            "content": content,
                            "reasoning_content": message.get("reasoning_content") or "",
                            "tool_calls": tool_calls,
                        }
                    }],
                }
        except Exception as e:
            raise RuntimeError(f"LlamaClient request failed: {e}")
