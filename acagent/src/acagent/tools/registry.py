from __future__ import annotations

from dataclasses import dataclass, field

from acagent.tools.base import AgentTool
from acagent.tools.event_tools import build_read_event_tool, build_search_events_tool
from acagent.tools.memory_tools import build_update_memory_tool


@dataclass(slots=True)
class ToolRegistry:
    tools: dict[str, AgentTool] = field(default_factory=dict)

    def register(self, tool: AgentTool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        return self.tools[name]

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self) -> list[str]:
        return sorted(self.tools)


def build_default_registry(*, event_search_top_k: int = 5) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_search_events_tool(default_top_k=event_search_top_k))
    registry.register(build_read_event_tool())
    registry.register(build_update_memory_tool())
    return registry
