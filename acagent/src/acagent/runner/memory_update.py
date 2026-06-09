from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from acagent.llm import PromptRenderer, StructuredToolCallingAdapter
from acagent.memory import MemoryState
from acagent.runner.context import WorkflowInput
from acagent.runner.error_feedback import error_feedback_message, trace_error
from acagent.runner.tool_budget import ToolBudget, ToolBudgetGuard, tool_budget_error_result
from acagent.schemas import EventDetail, Utterance
from acagent.trace import LlmTurnInput, LlmTurnOutput, LlmTurnTrace
from acagent.transcript import TranscriptChunk, TranscriptItem
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
class MemoryUpdateLoopResult:
    final_message: dict[str, Any] = field(default_factory=dict)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_turns: list[LlmTurnTrace] = field(default_factory=list)

    @property
    def memory_updated(self) -> bool:
        return any(result.tool_name == "update_memory" and result.ok for result in self.tool_results)


class MemoryUpdateLoop:
    def __init__(
        self,
        llm_client: ChatCompletionClient,
        *,
        registry: ToolRegistry | None = None,
        prompt_path: str | Path = "acagent/prompts/memory_update.md",
        max_llm_turns: int = 8,
        max_read_events: int = 3,
        max_tool_calls: int = 8,
        update_mode: str = "unsupervised",
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry or build_default_registry()
        self.executor = ToolExecutor(self.registry)
        self.prompt_path = Path(prompt_path)
        self.max_llm_turns = max_llm_turns
        self.max_read_events = max_read_events
        self.max_tool_calls = max_tool_calls
        self.update_mode = update_mode
        self.prompt_renderer = PromptRenderer()
        self.structured_adapter = StructuredToolCallingAdapter()

    def run(
        self,
        *,
        chunk: TranscriptChunk,
        memory: MemoryState,
        memory_store: Any | None = None,
    ) -> MemoryUpdateLoopResult:
        workflow_input = _workflow_input_from_chunk(chunk, memory, memory_store)
        messages = self._initial_messages(chunk, workflow_input)
        tools = self._tool_schemas()
        tool_results: list[ToolResult] = []
        llm_turns: list[LlmTurnTrace] = []
        budget_guard = ToolBudgetGuard(
            ToolBudget(
                max_read_events=self.max_read_events,
                max_tool_calls=self.max_tool_calls,
            )
        )

        for _ in range(self.max_llm_turns):
            turn_input = LlmTurnInput(messages=list(messages), tools=tools)
            completion = self.llm_client.create(
                messages=messages,
                tools=tools,
                tool_choice="auto",
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
                            "or return the required final JSON if no tool is needed."
                        ),
                    )
                )
                continue

            if action.is_final:
                try:
                    _validate_final_message(action.message)
                except ValueError as exc:
                    llm_turns.append(
                        LlmTurnTrace(
                            input=turn_input,
                            output=LlmTurnOutput(
                                raw=action.raw,
                                finish=False,
                                error=trace_error(
                                    code="final_memory_update_parse_error",
                                    message=str(exc),
                                ),
                            ),
                        )
                    )
                    messages.append(
                        error_feedback_message(
                            code="final_memory_update_parse_error",
                            message=str(exc),
                            instruction=(
                                'Return exactly one JSON object such as '
                                '{"status":"memory_updated","notes":[]}.'
                            ),
                        )
                    )
                    continue
                llm_turns.append(
                    LlmTurnTrace(
                        input=turn_input,
                        output=LlmTurnOutput(raw=action.raw, finish=True),
                    )
                )
                return MemoryUpdateLoopResult(
                    final_message=action.message,
                    tool_results=tool_results,
                    llm_turns=llm_turns,
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
                    messages.append(
                        error_feedback_message(
                            code=decision.error_code,
                            message=decision.message,
                            instruction=(
                                "The requested tool call was not executed because a hard "
                                "tool budget was reached. Continue without that tool call."
                            ),
                        )
                    )

        return MemoryUpdateLoopResult(tool_results=tool_results, llm_turns=llm_turns)

    def _initial_messages(
        self,
        chunk: TranscriptChunk,
        workflow_input: WorkflowInput,
    ) -> list[dict[str, Any]]:
        prompt = self.prompt_renderer.render_file(
            self.prompt_path,
            {
                "update_mode": self.update_mode,
                "memory_version": workflow_input.memory.version_id,
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
                "content": "You are the AC Agent memory update component. Use tools when needed.",
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
    current_utterance = utterances[-1] if utterances else _empty_utterance()
    return WorkflowInput(
        current_utterance=current_utterance,
        batch=utterances,
        local_context=[],
        memory=memory,
        is_eval_point=False,
        memory_store=memory_store,
        retrieved_event_details=[],
    )


def _utterance_from_item(item: TranscriptItem) -> Utterance:
    return Utterance(
        episode_id=_episode_id_from_item_id(item.item_id),
        scene_id="",
        utterance_id=item.item_id,
        turn_index=_turn_index_from_item_id(item.item_id),
        speaker=item.speaker,
        text=item.text,
    )


def _empty_utterance() -> Utterance:
    return Utterance(
        episode_id="",
        scene_id="",
        utterance_id="",
        turn_index=0,
        speaker="",
        text="",
    )


def _episode_id_from_item_id(item_id: str) -> str:
    return item_id.split("_", 1)[0] if "_" in item_id else ""


def _turn_index_from_item_id(item_id: str) -> int:
    suffix = item_id.rsplit("_U", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def _validate_final_message(message: dict[str, Any]) -> None:
    content = message.get("content")
    if isinstance(content, dict):
        data = content
    else:
        data = json.loads(str(content or "{}"))
    if not isinstance(data, dict):
        raise ValueError("Memory update final message must decode to a JSON object")
