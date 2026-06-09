from acagent.memory import MemoryState
from acagent.runner import MemoryUpdateLoop
from acagent.transcript import TranscriptChunk, TranscriptItem


def test_memory_update_loop_calls_update_memory_tool_until_final_message() -> None:
    memory = MemoryState()
    chunk = TranscriptChunk(
        status="update_memory",
        transcript="[U1] Monica: I am worried.",
        items=[
            TranscriptItem(
                item_id="U1",
                kind="utterance",
                speaker="Monica",
                text="I am worried.",
            )
        ],
        utterance_count=1,
        speakers=["Monica"],
    )
    llm_client = FakeMemoryUpdateClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "update_memory",
                                "arguments": (
                                    '{"character_updates":[{"character":"Monica",'
                                    '"field":"short_term_traits",'
                                    '"item":{"text":"Monica says she is worried.",'
                                    '"fact_type":"fact",'
                                    '"evidence_refs":[{"utterance_id":"U1"}]}}],'
                                    '"current_event_update":{"scene_id":"",'
                                    '"summary":"Monica says she is worried.",'
                                    '"characters":["Monica"]}}'
                                ),
                            },
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": '{"status":"memory_updated","notes":[]}',
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = MemoryUpdateLoop(llm_client=llm_client).run(chunk=chunk, memory=memory)

    assert result.memory_updated
    assert memory.characters["Monica"].short_term_traits[0].text == "Monica says she is worried."
    assert memory.current_event.summary == "Monica says she is worried."
    assert result.final_message["content"] == '{"status":"memory_updated","notes":[]}'
    assert len(result.llm_turns) == 2
    assert llm_client.requests[1]["messages"][-1]["role"] == "tool"
    assert "memory_version_after" in llm_client.requests[1]["messages"][-1]["content"]


def test_memory_update_loop_can_search_then_update_memory() -> None:
    memory = MemoryState()
    chunk = TranscriptChunk(
        status="update_memory",
        transcript="[U2] Ross: Carol moved out.",
        items=[TranscriptItem(item_id="U2", kind="utterance", speaker="Ross", text="Carol moved out.")],
        utterance_count=1,
        speakers=["Ross"],
    )
    llm_client = FakeMemoryUpdateClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_events",
                                "arguments": '{"query":"Carol Ross","characters":["Ross"],"top_k":3}',
                            },
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "update_memory",
                                "arguments": '{"character_updates":[],"event_updates":[]}',
                            },
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={"role": "assistant", "content": '{"status":"memory_updated","notes":[]}'},
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = MemoryUpdateLoop(llm_client=llm_client).run(chunk=chunk, memory=memory)

    assert [item.tool_name for item in result.tool_results] == ["search_events", "update_memory"]
    assert llm_client.requests[1]["messages"][-1]["role"] == "tool"
    assert llm_client.requests[2]["messages"][-1]["role"] == "tool"


def test_memory_update_loop_returns_parse_errors_to_model() -> None:
    memory = MemoryState()
    chunk = TranscriptChunk(
        status="update_memory",
        transcript="[U1] Monica: I am worried.",
        items=[TranscriptItem(item_id="U1", kind="utterance", speaker="Monica", text="I am worried.")],
        utterance_count=1,
        speakers=["Monica"],
    )
    llm_client = FakeMemoryUpdateClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "bad_call",
                            "type": "function",
                            "function": {"name": "update_memory", "arguments": "not-json"},
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={"role": "assistant", "content": '{"status":"memory_updated","notes":[]}'},
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = MemoryUpdateLoop(llm_client=llm_client).run(chunk=chunk, memory=memory)

    assert result.final_message["content"] == '{"status":"memory_updated","notes":[]}'
    assert result.llm_turns[0].output.error["code"] == "tool_arguments_parse_error"
    feedback = llm_client.requests[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert "tool_arguments_parse_error" in feedback["content"]


def test_memory_update_loop_returns_invalid_final_json_to_model() -> None:
    memory = MemoryState()
    chunk = TranscriptChunk(
        status="update_memory",
        transcript="[U1] Monica: I am worried.",
        items=[TranscriptItem(item_id="U1", kind="utterance", speaker="Monica", text="I am worried.")],
        utterance_count=1,
        speakers=["Monica"],
    )
    llm_client = FakeMemoryUpdateClient(
        [
            FakeCompletion(
                message={"role": "assistant", "content": "not-json"},
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
            FakeCompletion(
                message={"role": "assistant", "content": '{"status":"memory_updated","notes":[]}'},
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = MemoryUpdateLoop(llm_client=llm_client).run(chunk=chunk, memory=memory)

    assert result.final_message["content"] == '{"status":"memory_updated","notes":[]}'
    assert result.llm_turns[0].output.error["code"] == "final_memory_update_parse_error"
    assert "final_memory_update_parse_error" in llm_client.requests[1]["messages"][-1]["content"]


class FakeCompletion:
    def __init__(self, message, raw):
        self.message = message
        self.raw = raw
        self.tool_calls = message.get("tool_calls", [])


class FakeMemoryUpdateClient:
    def __init__(self, completions):
        self.completions = list(completions)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.completions.pop(0)
