from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from acagent.llm import PromptRenderer, StructuredToolCallingAdapter
from acagent.memory import MemoryState
from acagent.runner.context import WorkflowInput
from acagent.runner.error_feedback import error_feedback_message, trace_error
from acagent.runner.memory_update import _utterance_from_item
from acagent.runner.tool_budget import ToolBudget, ToolBudgetGuard, tool_budget_error_result
from acagent.schemas import Emotion, EmotionPrediction, Intensity, PredictionRecord
from acagent.storage.prediction_store import JsonlPredictionStore
from acagent.trace import LlmTurnInput, LlmTurnOutput, LlmTurnTrace
from acagent.transcript import TranscriptChunk
from acagent.tools import ToolRegistry, ToolResult, build_default_registry
from acagent.tools.executor import ToolExecutor


class ChatCompletionClient(Protocol):
    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(slots=True)
class LabelLoopResult:
    prediction_record: PredictionRecord | None = None
    final_message: dict[str, Any] = field(default_factory=dict)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_turns: list[LlmTurnTrace] = field(default_factory=list)
    trace_id: str = ""
    raw_prediction: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_updated(self) -> bool:
        return any(result.tool_name == "update_memory" and result.ok for result in self.tool_results)


class LabelLoop:
    def __init__(
        self,
        llm_client: ChatCompletionClient,
        *,
        registry: ToolRegistry | None = None,
        prompt_path: str | Path = "acagent/prompts/emotion_labeling.md",
        max_llm_turns: int = 8,
        max_read_events: int = 3,
        max_tool_calls: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry or build_default_registry()
        self.executor = ToolExecutor(self.registry)
        self.prompt_path = Path(prompt_path)
        self.max_llm_turns = max_llm_turns
        self.max_read_events = max_read_events
        self.max_tool_calls = max_tool_calls
        self.prompt_renderer = PromptRenderer()
        self.structured_adapter = StructuredToolCallingAdapter()

    def run(
        self,
        *,
        chunk: TranscriptChunk,
        memory: MemoryState,
        memory_store: Any | None = None,
        prediction_store: JsonlPredictionStore | None = None,
        trace_id: str | None = None,
    ) -> LabelLoopResult:
        workflow_input = _workflow_input_from_chunk(chunk, memory, memory_store)
        messages = self._initial_messages(chunk, workflow_input)
        tools = self._tool_schemas()
        tool_results: list[ToolResult] = []
        llm_turns: list[LlmTurnTrace] = []
        resolved_trace_id = trace_id or f"label-{uuid4().hex}"
        budget_guard = ToolBudgetGuard(
            ToolBudget(
                max_read_events=self.max_read_events,
                max_tool_calls=self.max_tool_calls,
            )
        )
        force_final_without_tools = False

        for _ in range(self.max_llm_turns):
            active_tools = [] if force_final_without_tools else tools
            turn_input = LlmTurnInput(messages=list(messages), tools=active_tools)
            completion = self.llm_client.create(
                messages=messages,
                tools=active_tools,
                tool_choice="none" if force_final_without_tools else "auto",
            )
            action = self.structured_adapter.from_completion(completion)

            if action.is_parse_error:
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(
                            raw=action.raw,
                            finish=False,
                            error=trace_error(
                                code="tool_arguments_parse_error",
                                message=action.error,
                            ),
                        ),
                    )
                )
                messages.append(
                    error_feedback_message(
                        code="tool_arguments_parse_error",
                        message=action.error,
                        instruction=(
                            "Regenerate the tool call with valid JSON object arguments, "
                            "or return the final prediction JSON if no tool is needed."
                        ),
                    )
                )
                continue

            if action.is_final:
                try:
                    record, raw_prediction = _prediction_record_from_message(
                        message=action.message,
                        fallback_utterance_id=workflow_input.current_utterance.utterance_id,
                        memory_version=workflow_input.memory.version_id,
                        trace_id=resolved_trace_id,
                    )
                except ValueError as exc:
                    force_final_without_tools = True
                    llm_turns.append(
                        LlmTurnTrace(
                            input=turn_input,
                            output=LlmTurnOutput(
                                raw=action.raw,
                                finish=False,
                                error=trace_error(
                                    code="final_prediction_parse_error",
                                    message=str(exc),
                                ),
                            ),
                        )
                    )
                    messages.append(
                        error_feedback_message(
                            code="final_prediction_parse_error",
                            message=str(exc),
                            instruction=(
                                "Return exactly one valid prediction JSON object with "
                                "utterance_id, emotions, intensities, and analysis. "
                                "No more tools are available for this correction; use "
                                "only the transcript, memory state, and existing tool results."
                            ),
                        )
                    )
                    continue
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(raw=action.raw, final_prediction=record.prediction, finish=True),
                    )
                )
                if prediction_store is not None:
                    prediction_store.append(
                        {
                            "trace_id": resolved_trace_id,
                            "utterance_id": record.utterance_id,
                            "memory_version": record.memory_version,
                            "prediction": record.prediction,
                            "raw_prediction": raw_prediction,
                            "gold_meld": chunk.meld,
                        }
                    )
                return LabelLoopResult(
                    prediction_record=record,
                    final_message=action.message,
                    tool_results=tool_results,
                    llm_turns=llm_turns,
                    trace_id=resolved_trace_id,
                    raw_prediction=raw_prediction,
                )

            messages.append(self.structured_adapter.assistant_tool_call_message(action))
            for provider_call in action.tool_calls:
                tool_call = provider_call.tool_call
                decision = budget_guard.check(tool_call)
                if decision.allowed:
                    budget_guard.record_allowed(tool_call)
                    tool_result = self.executor.execute(workflow_input, tool_call)
                else:
                    tool_result = tool_budget_error_result(
                        tool_call,
                        code=decision.error_code,
                        message=decision.message,
                    )
                tool_results.append(tool_result)
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(
                            raw=action.raw,
                            tool_call=tool_call,
                            finish=False,
                        ),
                        tool_result=tool_result,
                    )
                )
                messages.append(self.structured_adapter.tool_result_message(provider_call, tool_result))
                if not decision.allowed:
                    force_final_without_tools = True
                    messages.append(
                        error_feedback_message(
                            code=decision.error_code,
                            message=decision.message,
                            instruction=(
                                "The requested tool call was not executed because a hard "
                                "tool budget was reached. No more tools are available. "
                                "Based only on the transcript, memory state, and tool "
                                "results already available, return the final prediction "
                                "JSON now. Do not call tools again."
                            ),
                        )
                    )

        llm_turns.append(
            LlmTurnTrace(
                input=LlmTurnInput(messages=list(messages), tools=[]),
                output=LlmTurnOutput(
                    raw={},
                    finish=False,
                    error=trace_error(
                        code="max_llm_turns_exhausted",
                        message=(
                            "LabelLoop reached max_llm_turns without a valid prediction; "
                            "no fallback prediction was generated."
                        ),
                    ),
                ),
            )
        )
        return LabelLoopResult(
            prediction_record=None,
            tool_results=tool_results,
            llm_turns=llm_turns,
            trace_id=resolved_trace_id,
            raw_prediction={},
        )

    def _initial_messages(
        self,
        chunk: TranscriptChunk,
        workflow_input: WorkflowInput,
    ) -> list[dict[str, Any]]:
        prompt = self.prompt_renderer.render_file(
            self.prompt_path,
            {
                "memory_version": workflow_input.memory.version_id,
                "target_utterance_id": workflow_input.current_utterance.utterance_id,
                "target_utterance": asdict(workflow_input.current_utterance),
                "speakers": chunk.speakers,
                "transcript": chunk.transcript,
                "character_states": {
                    speaker: asdict(workflow_input.memory.characters[speaker])
                    for speaker in chunk.speakers
                    if speaker in workflow_input.memory.characters
                },
                "current_event": asdict(workflow_input.memory.current_event),
                "retrieved_event_details": [
                    asdict(detail) for detail in workflow_input.retrieved_event_details
                ],
            },
        )
        return [
            {
                "role": "system",
                "content": "You are the AC Agent emotion labeling component. Use tools when needed.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self.registry.tools.values()
        ]


def _workflow_input_from_chunk(
    chunk: TranscriptChunk,
    memory: MemoryState,
    memory_store: Any | None,
) -> WorkflowInput:
    utterances = [_utterance_from_item(item) for item in chunk.items if item.kind == "utterance"]
    if not utterances:
        raise ValueError("Label chunks must contain at least one utterance")
    target_utterance_id = chunk.target_utterance_id or (utterances[-1].utterance_id if utterances else "")
    current_utterance = next(
        (utterance for utterance in utterances if utterance.utterance_id == target_utterance_id),
        utterances[-1],
    )
    return WorkflowInput(
        current_utterance=current_utterance,
        batch=utterances,
        local_context=[],
        memory=memory,
        is_eval_point=True,
        memory_store=memory_store,
        retrieved_event_details=[],
    )


def _prediction_record_from_message(
    *,
    message: dict[str, Any],
    fallback_utterance_id: str,
    memory_version: str,
    trace_id: str,
) -> tuple[PredictionRecord, dict[str, Any]]:
    data = _json_from_message(message)
    utterance_id = str(data.get("utterance_id") or fallback_utterance_id)
    try:
        prediction = EmotionPrediction(
            emotions=[Emotion(str(item)) for item in data.get("emotions", [])],
            intensities=[Intensity(str(item)) for item in data.get("intensities", [])],
            analysis=dict(data.get("analysis", {})),
        )
        prediction.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid final prediction: {exc}") from exc
    return (
        PredictionRecord(
            utterance_id=utterance_id,
            memory_version=memory_version,
            prediction=prediction,
            trace_id=trace_id,
        ),
        data,
    )


def _json_from_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, dict):
        return content
    data = json.loads(_json_text_from_content(str(content or "{}")))
    if not isinstance(data, dict):
        raise ValueError("Label final message must decode to a JSON object")
    return data


def _json_text_from_content(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text
