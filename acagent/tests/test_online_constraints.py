from acagent.data_io import EvalPointLoader
from acagent.runner import AgentAction, OnlineAgentRunner, WorkflowInput
from acagent.runner.agent_loop import AgentLoop
from acagent.schemas import Emotion, EmotionPrediction, Intensity, Utterance
from acagent.tools import ToolCall


def test_eval_point_updates_current_batch_before_prediction() -> None:
    utterances = [
        Utterance("S01E01", "SC1", "U1", 1, "A", "one"),
        Utterance("S01E01", "SC1", "U2", 2, "B", "two"),
        Utterance("S01E01", "SC1", "U3", 3, "A", "three"),
    ]
    runner = OnlineAgentRunner(
        eval_points=EvalPointLoader(["U2"]),
    )

    predictions = runner.run(utterances)

    assert predictions[0].utterance_id == "U2"
    assert predictions[0].memory_version == "mem_00002"
    assert predictions[0].prediction.emotions == [Emotion.NEUTRAL]


class ScriptedEvalModel:
    def next_action(self, workflow_input: WorkflowInput, tool_results):
        if len(tool_results) == 0:
            return AgentAction(tool_call=ToolCall(name="search_events"))
        if len(tool_results) == 1:
            return AgentAction(tool_call=ToolCall(name="update_memory"))
        return AgentAction(
            final_prediction=EmotionPrediction(
                emotions=[Emotion.NEUTRAL],
                intensities=[Intensity.NONE],
                analysis={"final_reason": ""},
            )
        )


def test_eval_point_workflow_runs_tool_calls_then_prediction() -> None:
    utterance = Utterance("S01E01", "SC1", "U1", 1, "A", "one")
    runner = OnlineAgentRunner(
        eval_points=EvalPointLoader(["U1"]),
        agent_loop=AgentLoop(model=ScriptedEvalModel()),
    )

    predictions = runner.run([utterance])
    trace = runner.trace_logger.get(predictions[0].trace_id)

    assert [turn.output.tool_call.name for turn in trace.llm_turns if turn.output.tool_call] == [
        "search_events",
        "update_memory",
    ]
    assert [turn.tool_result.tool_name for turn in trace.llm_turns if turn.tool_result] == [
        "search_events",
        "update_memory",
    ]
    assert trace.llm_turns[-1].output.final_prediction.emotions == [Emotion.NEUTRAL]
    assert trace.memory_version_before_agent == "mem_00000"
    assert trace.memory_version_after_agent == "mem_00001"
    assert predictions[0].memory_version == "mem_00001"
