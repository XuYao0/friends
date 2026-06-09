from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI


ToolChoice = Literal["none", "auto", "required"]
ResponseFormatType = Literal["text", "json_object"]
ThinkingType = Literal["enabled", "disabled"]
ReasoningEffort = Literal["high", "max"]


@dataclass(frozen=True, slots=True)
class DeepSeekChatConfig:
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: ThinkingType | None = None
    reasoning_effort: ReasoningEffort | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    raw: dict[str, Any]
    message: dict[str, Any]
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def content_json(self) -> dict[str, Any]:
        if not self.content:
            return {}
        data = json.loads(self.content)
        if not isinstance(data, dict):
            raise ValueError("Expected assistant content to be a JSON object")
        return data


class DeepSeekChatCompletionClient:
    """Thin DeepSeek chat completion client.

    The client intentionally returns the raw provider response together with a
    small normalized view. Agent code should still validate tool arguments and
    structured JSON before executing anything.
    """

    def __init__(
        self,
        config: DeepSeekChatConfig | None = None,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] = OpenAI,
    ) -> None:
        self.config = config or DeepSeekChatConfig()
        self._client = client or client_factory(
            api_key=self._api_key(),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | Mapping[str, Any] | None = None,
        response_format: ResponseFormatType | Mapping[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        request: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [dict(message) for message in messages],
        }
        normalized_tools = normalize_tools(tools or [])
        if normalized_tools:
            request["tools"] = normalized_tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if response_format is not None:
            request["response_format"] = _response_format(response_format)
        if temperature is not None or self.config.temperature is not None:
            request["temperature"] = temperature if temperature is not None else self.config.temperature
        if max_tokens is not None or self.config.max_tokens is not None:
            request["max_tokens"] = max_tokens if max_tokens is not None else self.config.max_tokens

        extra_body = self._extra_body()
        if extra_body:
            request["extra_body"] = extra_body

        response = self._client.chat.completions.create(**request)
        raw = _to_dict(response)
        return _result_from_raw(raw)

    def complete_json(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | Mapping[str, Any] | None = None,
    ) -> ChatCompletionResult:
        return self.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format="json_object",
        )

    def _api_key(self) -> str:
        api_key = os.environ.get(self.config.api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing DeepSeek API key env var: {self.config.api_key_env}")
        return api_key

    def _extra_body(self) -> dict[str, Any]:
        extra_body: dict[str, Any] = {}
        if self.config.thinking is not None:
            extra_body["thinking"] = {"type": self.config.thinking}
        if self.config.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        return extra_body


def normalize_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_tool(tool) for tool in tools]


def _normalize_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and "function" in tool:
        return dict(tool)
    parameters = dict(tool.get("input_schema") or tool.get("parameters") or {"type": "object"})
    return {
        "type": "function",
        "function": {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "parameters": parameters,
            "strict": bool(tool.get("strict", False)),
        },
    }


def _response_format(response_format: ResponseFormatType | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(response_format, str):
        return {"type": response_format}
    return dict(response_format)


def _to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    raise TypeError(f"Unsupported chat completion response type: {type(response)!r}")


def _result_from_raw(raw: dict[str, Any]) -> ChatCompletionResult:
    choices = raw.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = dict(first_choice.get("message") or {})
    return ChatCompletionResult(
        raw=raw,
        message=message,
        content=str(message.get("content") or ""),
        tool_calls=list(message.get("tool_calls") or []),
        finish_reason=str(first_choice.get("finish_reason") or ""),
        usage=dict(raw.get("usage") or {}),
    )
