import json

from acagent.llm import StructuredToolCallingAdapter
from acagent.tools import ToolResult


def test_adapter_parses_provider_tool_calls_and_builds_messages() -> None:
    adapter = StructuredToolCallingAdapter()
    completion = FakeCompletion(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_events",
                        "arguments": '{"query":"Ross Carol","top_k":2}',
                    },
                }
            ],
        },
        raw={"choices": [{"finish_reason": "tool_calls"}], "usage": {"total_tokens": 12}},
    )

    action = adapter.from_completion(completion)
    assistant_message = adapter.assistant_tool_call_message(action)
    tool_message = adapter.tool_result_message(
        action.tool_calls[0],
        ToolResult(tool_name="search_events", output={"events": []}),
    )

    assert action.kind == "tool_calls"
    assert action.finish_reason == "tool_calls"
    assert action.usage == {"total_tokens": 12}
    assert action.tool_calls[0].provider_id == "call_1"
    assert action.tool_calls[0].tool_call.name == "search_events"
    assert action.tool_calls[0].tool_call.arguments == {"query": "Ross Carol", "top_k": 2}
    assert assistant_message["tool_calls"][0]["id"] == "call_1"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["ok"] is True


def test_adapter_returns_final_action_when_no_tool_calls() -> None:
    adapter = StructuredToolCallingAdapter()
    completion = FakeCompletion(
        message={"role": "assistant", "content": '{"status":"done"}'},
        raw={"choices": [{"finish_reason": "stop"}]},
    )

    action = adapter.from_completion(completion)

    assert action.kind == "final"
    assert action.is_final
    assert action.content == '{"status":"done"}'
    assert action.finish_reason == "stop"


def test_adapter_returns_parse_error_for_invalid_tool_arguments() -> None:
    adapter = StructuredToolCallingAdapter()
    completion = FakeCompletion(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "search_events",
                        "arguments": "not-json",
                    },
                }
            ],
        },
        raw={"choices": [{"finish_reason": "tool_calls"}]},
    )

    action = adapter.from_completion(completion)

    assert action.kind == "parse_error"
    assert action.is_parse_error
    assert "Expecting value" in action.error
    assert action.raw["choices"][0]["finish_reason"] == "tool_calls"


class FakeCompletion:
    def __init__(self, message, raw):
        self.message = message
        self.raw = raw
        self.tool_calls = message.get("tool_calls", [])
