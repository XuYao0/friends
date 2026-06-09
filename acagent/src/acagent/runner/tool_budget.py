from __future__ import annotations

from dataclasses import dataclass

from acagent.tools import ToolCall, ToolError, ToolResult


@dataclass(slots=True)
class ToolBudget:
    max_read_events: int = 3
    max_tool_calls: int = 8
    read_events: int = 0
    tool_calls: int = 0


@dataclass(frozen=True, slots=True)
class ToolBudgetDecision:
    allowed: bool
    error_code: str = ""
    message: str = ""


class ToolBudgetGuard:
    def __init__(self, budget: ToolBudget | None = None) -> None:
        self.budget = budget or ToolBudget()

    def check(self, tool_call: ToolCall) -> ToolBudgetDecision:
        if self.budget.tool_calls >= self.budget.max_tool_calls:
            return ToolBudgetDecision(
                allowed=False,
                error_code="max_tool_calls_exceeded",
                message=(
                    f"Tool call budget exceeded: max_tool_calls={self.budget.max_tool_calls}. "
                    "Do not call more tools for this chunk; produce the best final answer from current context."
                ),
            )
        if (
            tool_call.name == "read_event"
            and self.budget.read_events >= self.budget.max_read_events
        ):
            return ToolBudgetDecision(
                allowed=False,
                error_code="max_read_events_exceeded",
                message=(
                    f"read_event budget exceeded: max_read_events={self.budget.max_read_events}. "
                    "Use the event information already available and produce the best final answer."
                ),
            )
        return ToolBudgetDecision(allowed=True)

    def record_allowed(self, tool_call: ToolCall) -> None:
        self.budget.tool_calls += 1
        if tool_call.name == "read_event":
            self.budget.read_events += 1


def tool_budget_error_result(tool_call: ToolCall, *, code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_call.name,
        error=ToolError(code=code, message=message),
    )
