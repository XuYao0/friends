from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from acagent.data_io import EvalPointLoader
from acagent.memory import MemoryState
from acagent.runner.agent_loop import AgentLoop
from acagent.runner.context import WorkflowInput
from acagent.schemas import Emotion, EmotionPrediction, Intensity, PredictionRecord, Utterance
from acagent.trace import TraceLogger


@dataclass
class RunnerConfig:
    batch_size: int = 20
    local_context_window: int = 12
    max_read_events: int = 3


@dataclass
class OnlineAgentRunner:
    eval_points: EvalPointLoader
    agent_loop: AgentLoop = field(default_factory=AgentLoop)
    config: RunnerConfig = field(default_factory=RunnerConfig)
    memory: MemoryState = field(default_factory=MemoryState)
    memory_store: Any | None = None
    trace_logger: TraceLogger = field(default_factory=TraceLogger)

    def run(self, utterances: Iterable[Utterance]) -> list[PredictionRecord]:
        if self.memory_store is not None:
            self.memory = self.memory_store.load()
        batch: list[Utterance] = []
        context: deque[Utterance] = deque(maxlen=self.config.local_context_window)
        predictions: list[PredictionRecord] = []

        for utterance in utterances:
            batch.append(utterance)
            context.append(utterance)
            is_eval_point = self.eval_points.is_eval_point(utterance.utterance_id)
            workflow_input = WorkflowInput(
                current_utterance=utterance,
                batch=list(batch),
                local_context=list(context),
                memory=self.memory,
                is_eval_point=is_eval_point,
                memory_store=self.memory_store,
            )
            trace = self.trace_logger.start(
                utterance_id=utterance.utterance_id,
                episode_id=utterance.episode_id,
                scene_id=utterance.scene_id,
                turn_index=utterance.turn_index,
                is_eval_point=is_eval_point,
                memory_version_before_agent=self.memory.version_id,
            )

            result = self.agent_loop.run(workflow_input)
            self.memory = workflow_input.memory
            trace.llm_turns = result.llm_turns
            trace.memory_version_after_agent = self.memory.version_id
            trace.final_prediction = result.prediction
            if result.memory_updated:
                batch = []

            if is_eval_point:
                prediction = result.prediction or _empty_prediction()
                predictions.append(
                    PredictionRecord(
                        utterance_id=utterance.utterance_id,
                        memory_version=trace.memory_version_after_agent,
                        prediction=prediction,
                        trace_id=trace.trace_id,
                    )
                )

        if batch:
            workflow_input = WorkflowInput(
                current_utterance=batch[-1],
                batch=list(batch),
                local_context=list(context),
                memory=self.memory,
                is_eval_point=False,
                memory_store=self.memory_store,
            )
            trace = self.trace_logger.start(
                utterance_id=batch[-1].utterance_id,
                episode_id=batch[-1].episode_id,
                scene_id=batch[-1].scene_id,
                turn_index=batch[-1].turn_index,
                is_eval_point=False,
                memory_version_before_agent=self.memory.version_id,
            )
            result = self.agent_loop.run(workflow_input)
            self.memory = workflow_input.memory
            trace.llm_turns = result.llm_turns
            trace.memory_version_after_agent = self.memory.version_id
            trace.final_prediction = result.prediction

        return predictions


def _empty_prediction() -> EmotionPrediction:
    return EmotionPrediction(
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
