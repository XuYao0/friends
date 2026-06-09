from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from acagent.runner.context import WorkflowInput
from acagent.schemas import Emotion, EmotionPrediction, Intensity
from acagent.trace import LlmTurnInput, LlmTurnOutput, LlmTurnTrace
from acagent.tools import ToolCall, ToolResult, ToolRegistry, build_default_registry
from acagent.tools.executor import ToolExecutor


@dataclass(frozen=True, slots=True)
class AgentAction:
    tool_call: ToolCall | None = None
    final_prediction: EmotionPrediction | None = None
    finish: bool = False


class AgentModel(Protocol):
    def next_action(
        self,
        workflow_input: WorkflowInput,
        tool_results: list[ToolResult],
    ) -> AgentAction:
        ...


class EmptyAgentModel:
    """Minimal stand-in for a future LLM tool-calling model."""

    def next_action(
        self,
        workflow_input: WorkflowInput,
        tool_results: list[ToolResult],
    ) -> AgentAction:
        if not any(result.tool_name == "update_memory" for result in tool_results):
            return AgentAction(tool_call=ToolCall(name="update_memory"))
        if workflow_input.is_eval_point:
            return AgentAction(
                final_prediction=EmotionPrediction(
                    emotions=[Emotion.NEUTRAL],
                    intensities=[Intensity.NONE],
                    analysis={
                        "observable_facts": [],
                        "memory_evidence": [],
                        "inferences": [],
                        "uncertainties": [],
                        "final_reason": "",
                    },
                )
            )
        return AgentAction(finish=True)


@dataclass(slots=True)
class AgentLoopResult:
    prediction: EmotionPrediction | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_turns: list[LlmTurnTrace] = field(default_factory=list)

    @property
    def memory_updated(self) -> bool:
        return any(result.tool_name == "update_memory" and result.ok for result in self.tool_results)


class AgentLoop:
    def __init__(
        self,
        model: AgentModel | None = None,
        registry: ToolRegistry | None = None,
        max_tool_calls: int = 8,
    ) -> None:
        self.model = model or EmptyAgentModel()
        self.registry = registry or build_default_registry()
        self.executor = ToolExecutor(self.registry)
        self.max_tool_calls = max_tool_calls

    def run(self, workflow_input: WorkflowInput) -> AgentLoopResult:
        tool_results: list[ToolResult] = []
        llm_turns: list[LlmTurnTrace] = []
        for _ in range(self.max_tool_calls + 1):
            turn_input = self._build_turn_input(workflow_input, tool_results)
            action = self.model.next_action(workflow_input, tool_results)
            if action.tool_call is not None:
                tool_result = self.executor.execute(workflow_input, action.tool_call)
                tool_results.append(tool_result)
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(
                            raw={},
                            tool_call=action.tool_call,
                            final_prediction=None,
                            finish=False,
                        ),
                        tool_result=tool_result,
                    )
                )
                continue
            if action.final_prediction is not None:
                action.final_prediction.validate()
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(
                            raw={},
                            final_prediction=action.final_prediction,
                            finish=False,
                        ),
                    )
                )
                return AgentLoopResult(
                    prediction=action.final_prediction,
                    tool_results=tool_results,
                    llm_turns=llm_turns,
                )
            if action.finish:
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(raw={}, finish=True),
                    )
                )
                return AgentLoopResult(tool_results=tool_results, llm_turns=llm_turns)
            llm_turns.append(
                LlmTurnTrace(
                    input=turn_input,
                    output=LlmTurnOutput(raw={}),
                )
            )
            return AgentLoopResult(tool_results=tool_results, llm_turns=llm_turns)
        return AgentLoopResult(tool_results=tool_results, llm_turns=llm_turns)

    def _build_turn_input(
        self,
        workflow_input: WorkflowInput,
        tool_results: list[ToolResult],
    ) -> LlmTurnInput:
        return LlmTurnInput(
            messages=[
                {
                    "role": "system",
                    "content": "AC Agent tool-calling loop.",
                },
                {
                    "role": "user",
                    "content": {
                        "current_utterance": {
                            "utterance_id": workflow_input.current_utterance.utterance_id,
                            "speaker": workflow_input.current_utterance.speaker,
                            "text": workflow_input.current_utterance.text,
                        },
                        "batch_utterance_ids": [item.utterance_id for item in workflow_input.batch],
                        "local_context_utterance_ids": [item.utterance_id for item in workflow_input.local_context],
                        "is_eval_point": workflow_input.is_eval_point,
                        "memory_version": workflow_input.memory.version_id,
                        "tool_results": [self._tool_result_summary(item) for item in tool_results],
                    },
                },
            ],
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in self.registry.tools.values()
            ],
        )

    def _tool_result_summary(self, result: ToolResult) -> dict[str, object]:
        return {
            "tool_name": result.tool_name,
            "output": result.output,
            "error": None if result.error is None else {"code": result.error.code, "message": result.error.message},
            "memory_version_before": result.memory_version_before,
            "memory_version_after": result.memory_version_after,
        }
