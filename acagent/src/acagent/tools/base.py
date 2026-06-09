from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from acagent.runner.context import WorkflowInput


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolError:
    message: str
    code: str = "tool_error"


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_name: str
    output: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    memory_version_before: str = ""
    memory_version_after: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


ToolHandler = Callable[[WorkflowInput, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class AgentTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: ToolHandler
    is_read_only: bool = True
    is_concurrency_safe: bool = False

    def call(self, workflow_input: WorkflowInput, arguments: dict[str, Any] | None = None) -> ToolResult:
        memory_version_before = workflow_input.memory.version_id
        try:
            output = self.handler(workflow_input, arguments or {})
            return ToolResult(
                tool_name=self.name,
                output=output,
                memory_version_before=memory_version_before,
                memory_version_after=workflow_input.memory.version_id,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                error=ToolError(message=str(exc)),
                memory_version_before=memory_version_before,
                memory_version_after=workflow_input.memory.version_id,
            )
