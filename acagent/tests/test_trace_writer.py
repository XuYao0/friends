import json

from acagent.schemas import Emotion, EmotionPrediction, Intensity
from acagent.tools import ToolCall, ToolResult
from acagent.trace import (
    AgentTrace,
    JsonlTraceWriter,
    LlmTurnInput,
    LlmTurnOutput,
    LlmTurnTrace,
)


def test_jsonl_trace_writer_appends_and_reads_records(tmp_path) -> None:
    writer = JsonlTraceWriter(tmp_path / "traces.jsonl")
    trace = AgentTrace(
        trace_id="trace_00001",
        utterance_id="U1",
        episode_id="S01E01",
        scene_id="",
        turn_index=1,
        is_eval_point=True,
        memory_version_before_agent="mem_00000",
        memory_version_after_agent="mem_00001",
        final_prediction=EmotionPrediction(
            emotions=[Emotion.SADNESS],
            intensities=[Intensity.MEDIUM],
            analysis={"final_reason": "Ross is upset."},
        ),
        llm_turns=[
            LlmTurnTrace(
                input=LlmTurnInput(
                    messages=[{"role": "user", "content": "predict"}],
                    tools=[{"name": "search_events"}],
                ),
                output=LlmTurnOutput(
                    raw={"choices": [{"finish_reason": "tool_calls"}]},
                    tool_call=ToolCall(name="search_events", arguments={"query": "Ross"}),
                    finish=False,
                ),
                tool_result=ToolResult(tool_name="search_events", output={"events": []}),
            )
        ],
    )

    writer.append(trace)
    records = writer.read_records()
    raw_lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0]) == records[0]
    assert records[0]["trace_id"] == "trace_00001"
    assert records[0]["llm_turns"][0]["input"]["messages"][0]["content"] == "predict"
    assert records[0]["llm_turns"][0]["output"]["tool_call"]["name"] == "search_events"
    assert records[0]["llm_turns"][0]["tool_result"]["output"] == {"events": []}
    assert records[0]["final_prediction"]["emotions"] == ["sadness"]
