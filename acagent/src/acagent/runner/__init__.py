from __future__ import annotations

from typing import Any

__all__ = [
    "AgentAction",
    "AgentLoop",
    "AgentLoopResult",
    "EmptyAgentModel",
    "LabelLoop",
    "LabelLoopResult",
    "MemoryUpdateLoop",
    "MemoryUpdateLoopResult",
    "OnlineAgentRunner",
    "RunnerConfig",
    "WorkflowRunner",
    "WorkflowRunnerConfig",
    "WorkflowRunResult",
    "WorkflowInput",
    "build_default_workflow_runner",
    "build_workflow_runner_from_config",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentAction", "AgentLoop", "AgentLoopResult", "EmptyAgentModel"}:
        from acagent.runner.agent_loop import AgentAction, AgentLoop, AgentLoopResult, EmptyAgentModel

        return {
            "AgentAction": AgentAction,
            "AgentLoop": AgentLoop,
            "AgentLoopResult": AgentLoopResult,
            "EmptyAgentModel": EmptyAgentModel,
        }[name]
    if name == "WorkflowInput":
        from acagent.runner.context import WorkflowInput

        return WorkflowInput
    if name in {"LabelLoop", "LabelLoopResult"}:
        from acagent.runner.label import LabelLoop, LabelLoopResult

        return {"LabelLoop": LabelLoop, "LabelLoopResult": LabelLoopResult}[name]
    if name in {"MemoryUpdateLoop", "MemoryUpdateLoopResult"}:
        from acagent.runner.memory_update import MemoryUpdateLoop, MemoryUpdateLoopResult

        return {
            "MemoryUpdateLoop": MemoryUpdateLoop,
            "MemoryUpdateLoopResult": MemoryUpdateLoopResult,
        }[name]
    if name in {"OnlineAgentRunner", "RunnerConfig"}:
        from acagent.runner.online import OnlineAgentRunner, RunnerConfig

        return {"OnlineAgentRunner": OnlineAgentRunner, "RunnerConfig": RunnerConfig}[name]
    if name in {
        "WorkflowRunner",
        "WorkflowRunnerConfig",
        "WorkflowRunResult",
        "build_default_workflow_runner",
        "build_workflow_runner_from_config",
    }:
        from acagent.runner.workflow import (
            WorkflowRunner,
            WorkflowRunnerConfig,
            WorkflowRunResult,
            build_default_workflow_runner,
            build_workflow_runner_from_config,
        )

        return {
            "WorkflowRunner": WorkflowRunner,
            "WorkflowRunnerConfig": WorkflowRunnerConfig,
            "WorkflowRunResult": WorkflowRunResult,
            "build_default_workflow_runner": build_default_workflow_runner,
            "build_workflow_runner_from_config": build_workflow_runner_from_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
