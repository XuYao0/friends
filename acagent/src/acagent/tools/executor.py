from __future__ import annotations

from acagent.runner.context import WorkflowInput
from acagent.tools.base import ToolCall, ToolError, ToolResult
from acagent.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, workflow_input: WorkflowInput, tool_call: ToolCall) -> ToolResult:
        if not self.registry.has(tool_call.name):
            return ToolResult(
                tool_name=tool_call.name,
                error=ToolError(message=f"Unknown tool: {tool_call.name}", code="unknown_tool"),
                memory_version_before=workflow_input.memory.version_id,
                memory_version_after=workflow_input.memory.version_id,
            )
        return self.registry.get(tool_call.name).call(workflow_input, tool_call.arguments)
