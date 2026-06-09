from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from acagent.schemas import EmotionPrediction
from acagent.tools.base import ToolCall, ToolResult


@dataclass(slots=True)
class LlmTurnInput:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class LlmTurnOutput:
    raw: dict[str, Any] = field(default_factory=dict)
    tool_call: ToolCall | None = None
    final_prediction: EmotionPrediction | None = None
    finish: bool = False
    error: dict[str, Any] | None = None


@dataclass(slots=True)
class LlmTurnTrace:
    input: LlmTurnInput
    output: LlmTurnOutput
    tool_result: ToolResult | None = None


@dataclass(slots=True)
class AgentTrace:
    trace_id: str
    utterance_id: str
    episode_id: str
    scene_id: str
    turn_index: int
    is_eval_point: bool
    memory_version_before_agent: str
    memory_version_after_agent: str = ""
    llm_turns: list[LlmTurnTrace] = field(default_factory=list)
    final_prediction: EmotionPrediction | None = None


class TraceLogger:
    def __init__(self, start_counter: int = 0) -> None:
        self._counter = start_counter
        self.traces: dict[str, AgentTrace] = {}

    def start(
        self,
        *,
        utterance_id: str,
        episode_id: str,
        scene_id: str,
        turn_index: int,
        is_eval_point: bool,
        memory_version_before_agent: str,
    ) -> AgentTrace:
        self._counter += 1
        trace_id = f"trace_{self._counter:05d}"
        trace = AgentTrace(
            trace_id=trace_id,
            utterance_id=utterance_id,
            episode_id=episode_id,
            scene_id=scene_id,
            turn_index=turn_index,
            is_eval_point=is_eval_point,
            memory_version_before_agent=memory_version_before_agent,
        )
        self.traces[trace_id] = trace
        return trace

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(trace) for trace in self.traces.values()]

    def get(self, trace_id: str) -> AgentTrace:
        return self.traces[trace_id]
