from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from acagent.tools import ToolCall, ToolResult


StructuredActionKind = Literal["tool_calls", "final", "parse_error"]


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    provider_id: str
    raw: dict[str, Any]
    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class StructuredLlmAction:
    kind: StructuredActionKind
    raw: dict[str, Any]
    message: dict[str, Any]
    content: str = ""
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def is_final(self) -> bool:
        return self.kind == "final"

    @property
    def has_tool_calls(self) -> bool:
        return self.kind == "tool_calls" and bool(self.tool_calls)

    @property
    def is_parse_error(self) -> bool:
        return self.kind == "parse_error"


class StructuredToolCallingAdapter:
    """Convert provider chat completions into AC Agent internal actions."""

    def from_completion(self, completion: Any) -> StructuredLlmAction:
        raw = dict(getattr(completion, "raw", {}) or {})
        message = dict(getattr(completion, "message", {}) or {})
        content = str(getattr(completion, "content", "") or message.get("content") or "")
        finish_reason = str(
            getattr(completion, "finish_reason", "")
            or _finish_reason_from_raw(raw)
            or ""
        )
        usage = dict(getattr(completion, "usage", {}) or raw.get("usage") or {})
        raw_tool_calls = list(
            getattr(completion, "tool_calls", []) or message.get("tool_calls") or []
        )

        if not raw_tool_calls:
            return StructuredLlmAction(
                kind="final",
                raw=raw,
                message=message,
                content=content,
                finish_reason=finish_reason,
                usage=usage,
            )

        tool_calls: list[ProviderToolCall] = []
        for raw_tool_call in raw_tool_calls:
            try:
                tool_calls.append(_provider_tool_call(raw_tool_call))
            except ValueError as exc:
                return StructuredLlmAction(
                    kind="parse_error",
                    raw=raw,
                    message=message,
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    error=str(exc),
                )

        return StructuredLlmAction(
            kind="tool_calls",
            raw=raw,
            message=message,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def assistant_tool_call_message(self, action: StructuredLlmAction) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": action.message.get("content"),
            "tool_calls": [provider_call.raw for provider_call in action.tool_calls],
        }

    def tool_result_message(
        self,
        provider_call: ProviderToolCall,
        tool_result: ToolResult,
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": provider_call.provider_id,
            "content": json.dumps(
                _tool_result_payload(tool_result),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


def _provider_tool_call(raw_tool_call: Any) -> ProviderToolCall:
    if not isinstance(raw_tool_call, dict):
        raise ValueError("Provider tool call must be a JSON object")
    function = raw_tool_call.get("function", {})
    if not isinstance(function, dict):
        raise ValueError("Provider tool call function must be a JSON object")
    name = str(function.get("name", ""))
    if not name:
        raise ValueError("Provider tool call is missing function.name")
    return ProviderToolCall(
        provider_id=str(raw_tool_call.get("id", "")),
        raw=dict(raw_tool_call),
        tool_call=ToolCall(
            name=name,
            arguments=_arguments_from_provider(function.get("arguments")),
        ),
    )


def _arguments_from_provider(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    data = json.loads(str(arguments))
    if not isinstance(data, dict):
        raise ValueError("Tool call arguments must decode to a JSON object")
    return data


def _finish_reason_from_raw(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    first_choice = choices[0] if choices else {}
    if not isinstance(first_choice, dict):
        return ""
    return str(first_choice.get("finish_reason") or "")


def _tool_result_payload(tool_result: ToolResult) -> dict[str, Any]:
    return {
        "tool_name": tool_result.tool_name,
        "ok": tool_result.ok,
        "output": tool_result.output,
        "error": None
        if tool_result.error is None
        else {
            "code": tool_result.error.code,
            "message": tool_result.error.message,
        },
        "memory_version_before": tool_result.memory_version_before,
        "memory_version_after": tool_result.memory_version_after,
    }
