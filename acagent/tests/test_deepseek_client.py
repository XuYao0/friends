import pytest

from acagent.llm import DeepSeekChatCompletionClient, DeepSeekChatConfig, normalize_tools


def test_normalize_tools_converts_agent_tool_schema() -> None:
    tools = normalize_tools(
        [
            {
                "name": "search_events",
                "description": "Search event memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "search_events",
                "description": "Search event memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "strict": False,
            },
        }
    ]


def test_deepseek_client_builds_chat_completion_request(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake_client = FakeOpenAIClient(
        response={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_events",
                                    "arguments": '{"query":"Ross"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"total_tokens": 12},
        }
    )
    client = DeepSeekChatCompletionClient(
        DeepSeekChatConfig(
            model="deepseek-v4-flash",
            temperature=0.2,
            max_tokens=100,
            thinking="disabled",
            reasoning_effort="high",
        ),
        client=fake_client,
    )

    result = client.create(
        messages=[{"role": "user", "content": "Please update memory."}],
        tools=[{"name": "search_events", "description": "Search.", "input_schema": {"type": "object"}}],
        tool_choice="auto",
    )

    request = fake_client.chat.completions.last_request
    assert request["model"] == "deepseek-v4-flash"
    assert request["messages"] == [{"role": "user", "content": "Please update memory."}]
    assert request["tool_choice"] == "auto"
    assert request["tools"][0]["function"]["name"] == "search_events"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 100
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "high",
    }
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["function"]["name"] == "search_events"
    assert result.usage == {"total_tokens": 12}


def test_deepseek_client_complete_json_sets_response_format(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake_client = FakeOpenAIClient(
        response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"updated": true}'},
                }
            ],
        }
    )
    client = DeepSeekChatCompletionClient(client=fake_client)

    result = client.complete_json(messages=[{"role": "user", "content": "Return JSON."}])

    request = fake_client.chat.completions.last_request
    assert request["response_format"] == {"type": "json_object"}
    assert result.content_json() == {"updated": True}


def test_deepseek_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekChatCompletionClient()


class FakeOpenAIClient:
    def __init__(self, response):
        self.chat = FakeChat(response)


class FakeChat:
    def __init__(self, response):
        self.completions = FakeCompletions(response)


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response
