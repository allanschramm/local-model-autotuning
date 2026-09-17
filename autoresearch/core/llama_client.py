import json
import urllib.request
from dataclasses import dataclass, replace
from typing import Any


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
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    chat_template_args: dict[str, Any] | None = None

    def to_payload(self) -> dict:
        """Build API payload dict, omitting None fields."""
        d: dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temp,
        }
        if self.stop is not None:
            if isinstance(self.stop, str):
                d["stop"] = [self.stop]
            elif isinstance(self.stop, (tuple, set)):
                d["stop"] = list(self.stop)
            else:
                d["stop"] = self.stop
        for key in (
            "top_p",
            "min_p",
            "top_k",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            val = getattr(self, key)
            if val is not None:
                d[key] = val

        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
            chat_kwargs = dict(self.chat_template_kwargs or self.chat_template_args or {})
            chat_kwargs["reasoning_effort"] = self.reasoning_effort
            d["chat_template_kwargs"] = chat_kwargs
            d["chat_template_args"] = chat_kwargs
        elif self.chat_template_kwargs is not None:
            d["chat_template_kwargs"] = self.chat_template_kwargs
            d["chat_template_args"] = self.chat_template_kwargs
        elif self.chat_template_args is not None:
            d["chat_template_kwargs"] = self.chat_template_args
            d["chat_template_args"] = self.chat_template_args

        return d

    def with_overrides(self, **overrides) -> "GenerationParams":
        """Return a new instance with the given fields overridden."""
        return replace(self, **overrides)


class LlamaClient:
    """Deep module for llama-server communication."""

    def __init__(self, port: int, timeout: float = 420.0):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        # Enforce 420s turn timeout floor to prevent premature cancellation
        effective_timeout = timeout if timeout is not None else 420.0
        self.timeout = max(float(effective_timeout), 420.0)

    def complete(
        self, prompt: str, gen: GenerationParams | None = None, **kwargs
    ) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"

        if gen is not None:
            payload = gen.to_payload()
        else:
            payload = {
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temp", 0.1),
            }
            for key in [
                "top_p",
                "min_p",
                "top_k",
                "repeat_penalty",
                "presence_penalty",
                "frequency_penalty",
            ]:
                if key in kwargs and kwargs[key] is not None:
                    payload[key] = kwargs[key]

        if "reasoning_effort" not in payload and kwargs.get("reasoning_effort") is not None:
            effort = kwargs["reasoning_effort"]
            payload["reasoning_effort"] = effort
            chat_kwargs = dict(
                payload.get("chat_template_kwargs")
                or kwargs.get("chat_template_kwargs")
                or kwargs.get("chat_template_args")
                or {}
            )
            chat_kwargs["reasoning_effort"] = effort
            payload["chat_template_kwargs"] = chat_kwargs
            payload["chat_template_args"] = chat_kwargs
        elif (
            "chat_template_kwargs" not in payload and kwargs.get("chat_template_kwargs") is not None
        ):
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]
            payload["chat_template_args"] = kwargs["chat_template_kwargs"]
        elif "chat_template_args" not in payload and kwargs.get("chat_template_args") is not None:
            payload["chat_template_kwargs"] = kwargs["chat_template_args"]
            payload["chat_template_args"] = kwargs["chat_template_args"]

        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = False

        # Stop tokens: from gen, from kwargs, or default
        if "stop" not in payload:
            raw_stop = kwargs.get("stop", ["</s>", "Instruction:", "User:"])
            if isinstance(raw_stop, str):
                payload["stop"] = [raw_stop]
            elif isinstance(raw_stop, (tuple, set)):
                payload["stop"] = list(raw_stop)
            else:
                payload["stop"] = raw_stop
        elif isinstance(payload.get("stop"), str):
            payload["stop"] = [payload["stop"]]
        elif isinstance(payload.get("stop"), (tuple, set)):
            payload["stop"] = list(payload["stop"])

        # Forward tools if present
        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )

        req_timeout = kwargs.get("timeout")
        effective_timeout = req_timeout if req_timeout is not None else self.timeout
        timeout = max(float(effective_timeout if effective_timeout is not None else 420.0), 420.0)

        try:
            # Enforce 420s turn timeout floor to prevent premature cancellation of reasoning turns.
            with urllib.request.urlopen(req, timeout=timeout) as res:
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
                    "usage": {"total_tokens": total_tokens},
                    "choices": [
                        {
                            "message": {
                                "content": content,
                                "reasoning_content": message.get("reasoning_content") or "",
                                "tool_calls": tool_calls,
                            }
                        }
                    ],
                }
        except Exception as e:
            raise RuntimeError(f"LlamaClient request failed: {e}")
