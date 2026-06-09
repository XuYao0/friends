import json

from acagent.memory import MemoryState
from acagent.runner import LabelLoop
from acagent.schemas import EventDetail, EventIndex, EventUpdate, MemoryDelta
from acagent.storage import JsonlPredictionStore
from acagent.transcript import TranscriptChunk, TranscriptItem


def test_label_loop_parses_final_prediction() -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross is upset about Carol."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        ]
    )

    result = LabelLoop(llm_client=llm_client).run(chunk=chunk, memory=memory, trace_id="trace-1")

    assert result.prediction_record is not None
    assert result.prediction_record.utterance_id == "U2"
    assert result.prediction_record.prediction.emotions == ["sadness"]
    assert result.prediction_record.prediction.intensities == ["medium"]
    assert result.prediction_record.trace_id == "trace-1"
    assert result.raw_prediction["analysis"]["final_reason"] == "Ross is upset about Carol."


def test_label_loop_can_update_memory_then_save_prediction(tmp_path) -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    prediction_store = JsonlPredictionStore(tmp_path / "predictions.jsonl")
    llm_client = FakeLabelClient(
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
                                    '{"character_updates":[{"character":"Ross",'
                                    '"field":"recent_events",'
                                    '"item":{"text":"Ross says Carol moved out.",'
                                    '"fact_type":"fact",'
                                    '"evidence_refs":[{"utterance_id":"U2"}]}}]}'
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
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["high"],
                            "analysis": {"final_reason": "The move-out signals loss."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = LabelLoop(llm_client=llm_client).run(
        chunk=chunk,
        memory=memory,
        prediction_store=prediction_store,
        trace_id="trace-2",
    )

    assert result.memory_updated
    assert memory.characters["Ross"].recent_events[0].text == "Ross says Carol moved out."
    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["trace_id"] == "trace-2"
    assert saved["utterance_id"] == "U2"
    assert saved["prediction"]["emotions"] == ["sadness"]
    assert saved["gold_meld"] == {"emotion": "sadness"}
    assert llm_client.requests[1]["messages"][-1]["role"] == "tool"


def test_label_loop_returns_invalid_final_prediction_to_model() -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["unknown"],
                            "intensities": ["medium"],
                            "analysis": {},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross reports loss."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = LabelLoop(llm_client=llm_client).run(chunk=chunk, memory=memory, trace_id="trace-3")

    assert result.prediction_record is not None
    assert result.prediction_record.prediction.emotions == ["sadness"]
    assert result.llm_turns[0].output.error["code"] == "final_prediction_parse_error"
    assert "final_prediction_parse_error" in llm_client.requests[1]["messages"][-1]["content"]
    assert llm_client.requests[1]["tools"] == []
    assert llm_client.requests[1]["tool_choice"] == "none"


def test_label_loop_parses_markdown_fenced_final_prediction() -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": (
                        "```json\n"
                        "{\n"
                        '  "utterance_id": "U2",\n'
                        '  "emotions": ["neutral"],\n'
                        '  "intensities": ["none"],\n'
                        '  "analysis": {"final_reason": "Routine greeting."}\n'
                        "}\n"
                        "```"
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        ]
    )

    result = LabelLoop(llm_client=llm_client).run(chunk=chunk, memory=memory, trace_id="trace-3b")

    assert result.prediction_record is not None
    assert result.prediction_record.prediction.emotions == ["neutral"]
    assert result.llm_turns[-1].output.error is None


def test_label_loop_returns_tool_argument_parse_error_to_model() -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "bad_call",
                            "type": "function",
                            "function": {"name": "search_events", "arguments": "not-json"},
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross reports loss."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = LabelLoop(llm_client=llm_client).run(chunk=chunk, memory=memory, trace_id="trace-4")

    assert result.prediction_record is not None
    assert result.llm_turns[0].output.error["code"] == "tool_arguments_parse_error"
    assert "tool_arguments_parse_error" in llm_client.requests[1]["messages"][-1]["content"]


def test_label_loop_returns_no_prediction_when_no_valid_prediction(tmp_path) -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    prediction_store = JsonlPredictionStore(tmp_path / "predictions.jsonl")
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["neutral", "sadness"],
                            "intensities": ["none", "medium"],
                            "analysis": {},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        ]
    )

    result = LabelLoop(llm_client=llm_client, max_llm_turns=1).run(
        chunk=chunk,
        memory=memory,
        prediction_store=prediction_store,
        trace_id="trace-5",
    )

    assert result.prediction_record is None
    assert result.raw_prediction == {}
    assert result.llm_turns[-1].output.error["code"] == "max_llm_turns_exhausted"
    assert not (tmp_path / "predictions.jsonl").exists()


def test_label_loop_blocks_read_event_after_hard_limit() -> None:
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            event_updates=[
                EventUpdate(
                    index=EventIndex(
                        event_id="EVT_1",
                        short_summary="Ross is upset about Carol.",
                        detail_id="D1",
                    ),
                    detail=EventDetail(
                        detail_id="D1",
                        event_id="EVT_1",
                        description="Ross is upset because Carol moved out.",
                    ),
                )
            ]
        )
    )
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "read_1",
                            "type": "function",
                            "function": {
                                "name": "read_event",
                                "arguments": '{"event_id":"EVT_1"}',
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
                            "id": "read_2",
                            "type": "function",
                            "function": {
                                "name": "read_event",
                                "arguments": '{"event_id":"EVT_1"}',
                            },
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross reports loss."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = LabelLoop(llm_client=llm_client, max_read_events=1).run(
        chunk=chunk,
        memory=memory,
        trace_id="trace-6",
    )

    assert [item.ok for item in result.tool_results] == [True, False]
    assert result.tool_results[1].error.code == "max_read_events_exceeded"
    assert result.llm_turns[1].tool_result.error.code == "max_read_events_exceeded"
    assert "max_read_events_exceeded" in llm_client.requests[2]["messages"][-1]["content"]
    assert llm_client.requests[2]["tools"] == []
    assert llm_client.requests[2]["tool_choice"] == "none"


def test_label_loop_blocks_tool_calls_after_hard_limit() -> None:
    memory = MemoryState()
    chunk = _label_chunk()
    llm_client = FakeLabelClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "search_1",
                            "type": "function",
                            "function": {
                                "name": "search_events",
                                "arguments": '{"query":"Carol"}',
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
                            "id": "search_2",
                            "type": "function",
                            "function": {
                                "name": "search_events",
                                "arguments": '{"query":"Ross"}',
                            },
                        }
                    ],
                },
                raw={"choices": [{"finish_reason": "tool_calls"}]},
            ),
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "U2",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross reports loss."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            ),
        ]
    )

    result = LabelLoop(llm_client=llm_client, max_tool_calls=1).run(
        chunk=chunk,
        memory=memory,
        trace_id="trace-7",
    )

    assert [item.ok for item in result.tool_results] == [True, False]
    assert result.tool_results[1].error.code == "max_tool_calls_exceeded"
    assert "max_tool_calls_exceeded" in llm_client.requests[2]["messages"][-1]["content"]
    assert llm_client.requests[2]["tools"] == []
    assert llm_client.requests[2]["tool_choice"] == "none"


def _label_chunk() -> TranscriptChunk:
    return TranscriptChunk(
        status="label",
        transcript="[U1] Rachel: Are you okay?\n[U2] Ross: Carol moved out today.",
        items=[
            TranscriptItem(item_id="U1", kind="utterance", speaker="Rachel", text="Are you okay?"),
            TranscriptItem(
                item_id="U2",
                kind="utterance",
                speaker="Ross",
                text="Carol moved out today.",
            ),
        ],
        utterance_count=2,
        speakers=["Rachel", "Ross"],
        target_utterance_id="U2",
        meld={"emotion": "sadness"},
    )


class FakeCompletion:
    def __init__(self, message, raw):
        self.message = message
        self.raw = raw
        self.tool_calls = message.get("tool_calls", [])


class FakeLabelClient:
    def __init__(self, completions):
        self.completions = list(completions)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.completions.pop(0)
