import json

from acagent.memory import MemoryState
from acagent.runner import LabelLoop, MemoryUpdateLoop, WorkflowRunner, WorkflowRunnerConfig
from acagent.trace import JsonlTraceWriter


def test_workflow_runner_dispatches_update_and_label_chunks(tmp_path) -> None:
    screenplay_path = tmp_path / "friends.jsonl"
    screenplay_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "utterance",
                        "season": 1,
                        "episode": 1,
                        "content": {
                            "global_utterance_id": 1,
                            "speaker": "Monica",
                            "utterance": "I am worried.",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "utterance",
                        "season": 1,
                        "episode": 1,
                        "content": {
                            "global_utterance_id": 2,
                            "speaker": "Ross",
                            "utterance": "Carol moved out today.",
                        },
                        "long_context_selected": True,
                        "meld": {"emotion": "sadness"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    memory = MemoryState()
    update_client = FakeClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_update",
                            "type": "function",
                            "function": {
                                "name": "update_memory",
                                "arguments": (
                                    '{"character_updates":[{"character":"Monica",'
                                    '"field":"short_term_traits",'
                                    '"item":{"text":"Monica is worried.",'
                                    '"fact_type":"fact",'
                                    '"evidence_refs":[{"utterance_id":"S01E01_U000001"}]}}]}'
                                ),
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
    label_client = FakeClient(
        [
            FakeCompletion(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "utterance_id": "S01E01_U000002",
                            "emotions": ["sadness"],
                            "intensities": ["medium"],
                            "analysis": {"final_reason": "Ross reports a painful separation."},
                        }
                    ),
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            )
        ]
    )
    progress_messages: list[str] = []

    result = WorkflowRunner(
        memory_update_loop=MemoryUpdateLoop(llm_client=update_client),
        label_loop=LabelLoop(llm_client=label_client),
        config=WorkflowRunnerConfig(transcript_path=screenplay_path, batch_size=1),
        memory=memory,
        trace_writer=JsonlTraceWriter(tmp_path / "traces.jsonl"),
        progress_writer=progress_messages.append,
    ).run()

    assert result.chunks_processed == 2
    assert result.update_chunks == 1
    assert result.label_chunks == 1
    assert len(result.predictions) == 1
    assert result.predictions[0].utterance_id == "S01E01_U000002"
    assert result.predictions[0].memory_version == "mem_00001"
    assert memory.characters["Monica"].short_term_traits[0].text == "Monica is worried."
    assert len(update_client.requests) == 2
    assert len(label_client.requests) == 1
    trace_records = JsonlTraceWriter(tmp_path / "traces.jsonl").read_records()
    assert [record["trace_id"] for record in trace_records] == ["trace_00001", "trace_00002"]
    assert trace_records[0]["llm_turns"][0]["input"]["messages"][0]["role"] == "system"
    assert trace_records[0]["llm_turns"][0]["output"]["raw"] == {
        "choices": [{"finish_reason": "tool_calls"}]
    }
    assert trace_records[0]["llm_turns"][0]["tool_result"]["tool_name"] == "update_memory"
    assert trace_records[1]["is_eval_point"] is True
    assert trace_records[1]["final_prediction"]["emotions"] == ["sadness"]
    assert progress_messages == [
        "chunk=1 phase=start status=update_memory utterance_id=S01E01_U000001 memory_version=mem_00000",
        "chunk=1 phase=done status=update_memory utterance_id=S01E01_U000001 memory_version=mem_00001",
        "chunk=2 phase=start status=label utterance_id=S01E01_U000002 memory_version=mem_00001",
        "chunk=2 phase=done status=label utterance_id=S01E01_U000002 memory_version=mem_00001",
    ]


class FakeCompletion:
    def __init__(self, message, raw):
        self.message = message
        self.raw = raw
        self.tool_calls = message.get("tool_calls", [])


class FakeClient:
    def __init__(self, completions):
        self.completions = list(completions)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.completions.pop(0)
