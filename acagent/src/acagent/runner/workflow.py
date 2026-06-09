from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acagent.config import ExperimentConfig
from acagent.llm import DeepSeekChatCompletionClient, DeepSeekChatConfig
from acagent.memory import MemoryState
from acagent.runner.label import LabelLoop
from acagent.runner.memory_update import MemoryUpdateLoop
from acagent.schemas import PredictionRecord
from acagent.storage import JsonlPredictionStore, JsonMemoryStore
from acagent.trace import AgentTrace, JsonlTraceWriter, TraceLogger
from acagent.tools import build_default_registry
from acagent.transcript import FriendsTranscriptChunkSource, TranscriptChunk, TranscriptItem


def _default_progress_writer(message: str) -> None:
    print(message, file=sys.stderr)


@dataclass(frozen=True, slots=True)
class WorkflowRunnerConfig:
    transcript_path: str | Path = "screenplays/friends_records_renamed_with_selected.jsonl"
    batch_size: int = 20
    max_utterances: int | None = None
    start_chunk_index: int = 1


@dataclass(slots=True)
class WorkflowRunResult:
    predictions: list[PredictionRecord] = field(default_factory=list)
    chunks_processed: int = 0
    update_chunks: int = 0
    label_chunks: int = 0
    memory: MemoryState = field(default_factory=MemoryState)


@dataclass
class WorkflowRunner:
    memory_update_loop: MemoryUpdateLoop
    label_loop: LabelLoop
    config: WorkflowRunnerConfig = field(default_factory=WorkflowRunnerConfig)
    memory: MemoryState = field(default_factory=MemoryState)
    memory_store: JsonMemoryStore | None = None
    prediction_store: JsonlPredictionStore | None = None
    trace_logger: TraceLogger = field(default_factory=TraceLogger)
    trace_writer: JsonlTraceWriter | None = None
    progress_writer: Callable[[str], None] | None = _default_progress_writer

    def run(self) -> WorkflowRunResult:
        if self.memory_store is not None:
            self.memory = self.memory_store.load()

        predictions: list[PredictionRecord] = []
        chunks_processed = 0
        update_chunks = 0
        label_chunks = 0
        source = FriendsTranscriptChunkSource(
            path=self.config.transcript_path,
            batch_size=self.config.batch_size,
            max_utterances=self.config.max_utterances,
        )

        for absolute_chunk_index, chunk in enumerate(source.iter_chunks(), start=1):
            if absolute_chunk_index < self.config.start_chunk_index:
                continue
            chunks_processed += 1
            self._write_progress(absolute_chunk_index, chunk, phase="start")
            trace = self.trace_logger.start(
                utterance_id=_trace_utterance_id(chunk),
                episode_id=_episode_id_from_chunk(chunk),
                scene_id="",
                turn_index=_turn_index_from_chunk(chunk),
                is_eval_point=chunk.status == "label",
                memory_version_before_agent=self.memory.version_id,
            )

            if chunk.status == "update_memory":
                update_chunks += 1
                update_result = self.memory_update_loop.run(
                    chunk=chunk,
                    memory=self.memory,
                    memory_store=self.memory_store,
                )
                self._refresh_memory()
                trace.llm_turns = update_result.llm_turns
                trace.memory_version_after_agent = self.memory.version_id
                self._persist_trace(trace)
                self._write_progress(absolute_chunk_index, chunk, phase="done")
                continue

            label_chunks += 1
            label_result = self.label_loop.run(
                chunk=chunk,
                memory=self.memory,
                memory_store=self.memory_store,
                prediction_store=self.prediction_store,
                trace_id=trace.trace_id,
            )
            self._refresh_memory()
            trace.llm_turns = label_result.llm_turns
            trace.memory_version_after_agent = self.memory.version_id
            prediction = label_result.prediction_record
            if prediction is not None:
                trace.final_prediction = prediction.prediction
                predictions.append(
                    PredictionRecord(
                        utterance_id=prediction.utterance_id,
                        memory_version=trace.memory_version_after_agent,
                        prediction=prediction.prediction,
                        trace_id=trace.trace_id,
                    )
                )
            self._persist_trace(trace)
            self._write_progress(absolute_chunk_index, chunk, phase="done")

        return WorkflowRunResult(
            predictions=predictions,
            chunks_processed=chunks_processed,
            update_chunks=update_chunks,
            label_chunks=label_chunks,
            memory=self.memory,
        )

    def _refresh_memory(self) -> None:
        if self.memory_store is not None:
            self.memory = self.memory_store.load()

    def _persist_trace(self, trace: AgentTrace) -> None:
        if self.trace_writer is not None:
            self.trace_writer.append(trace)

    def _write_progress(self, chunk_index: int, chunk: TranscriptChunk, *, phase: str) -> None:
        if self.progress_writer is None:
            return
        self.progress_writer(
            " ".join(
                [
                    f"chunk={chunk_index}",
                    f"phase={phase}",
                    f"status={chunk.status}",
                    f"utterance_id={_trace_utterance_id(chunk)}",
                    f"memory_version={self.memory.version_id}",
                ]
            )
        )

def build_default_workflow_runner(
    *,
    transcript_path: str | Path = "screenplays/friends_records_renamed_with_selected.jsonl",
    batch_size: int = 20,
    memory_path: str | Path = "acagent_outputs/default/memory.json",
    prediction_path: str | Path = "acagent_outputs/default/predictions.jsonl",
    trace_path: str | Path = "acagent_outputs/default/traces.jsonl",
) -> WorkflowRunner:
    config = ExperimentConfig(
        transcript_path=str(transcript_path),
        batch_size=batch_size,
        output_dir=str(Path(memory_path).parent),
    )
    return build_workflow_runner_from_config(
        config,
        memory_path=memory_path,
        prediction_path=prediction_path,
        trace_path=trace_path,
    )


def build_workflow_runner_from_config(
    config: ExperimentConfig | str | Path,
    *,
    memory_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> WorkflowRunner:
    resolved_config = (
        ExperimentConfig.from_yaml(config)
        if isinstance(config, (str, Path))
        else config
    )
    output_dir = Path(resolved_config.output_dir)
    update_client = _deepseek_client(
        DeepSeekChatConfig(
            model=resolved_config.cheap_update_model,
            temperature=resolved_config.temperature,
            max_tokens=resolved_config.max_tokens,
        ),
        client_factory=client_factory,
    )
    label_client = _deepseek_client(
        DeepSeekChatConfig(
            model=resolved_config.reasoning_model,
            temperature=resolved_config.temperature,
            max_tokens=resolved_config.max_tokens,
        ),
        client_factory=client_factory,
    )
    registry = build_default_registry(event_search_top_k=resolved_config.event_search_top_k)
    return WorkflowRunner(
        memory_update_loop=MemoryUpdateLoop(
            llm_client=update_client,
            registry=registry,
            prompt_path=resolved_config.memory_update_prompt_path,
            max_read_events=resolved_config.max_read_events,
            max_tool_calls=resolved_config.max_tool_calls,
            update_mode=resolved_config.memory_update_mode,
        ),
        label_loop=LabelLoop(
            llm_client=label_client,
            registry=registry,
            prompt_path=resolved_config.emotion_labeling_prompt_path,
            max_read_events=resolved_config.max_read_events,
            max_tool_calls=resolved_config.max_tool_calls,
        ),
        config=WorkflowRunnerConfig(
            transcript_path=resolved_config.transcript_path,
            batch_size=resolved_config.batch_size,
            max_utterances=resolved_config.max_utterances,
            start_chunk_index=resolved_config.start_chunk_index,
        ),
        trace_logger=TraceLogger(start_counter=max(0, resolved_config.start_chunk_index - 1)),
        memory_store=JsonMemoryStore(memory_path or output_dir / "memory.json"),
        prediction_store=JsonlPredictionStore(prediction_path or output_dir / "predictions.jsonl"),
        trace_writer=JsonlTraceWriter(trace_path or output_dir / "traces.jsonl"),
    )


def _deepseek_client(
    config: DeepSeekChatConfig,
    *,
    client_factory: Callable[..., Any] | None,
) -> DeepSeekChatCompletionClient:
    if client_factory is None:
        return DeepSeekChatCompletionClient(config)
    return DeepSeekChatCompletionClient(config, client_factory=client_factory)


def _trace_utterance_id(chunk: TranscriptChunk) -> str:
    if chunk.target_utterance_id:
        return chunk.target_utterance_id
    utterance = _last_utterance(chunk)
    return utterance.item_id if utterance is not None else ""


def _episode_id_from_chunk(chunk: TranscriptChunk) -> str:
    utterance_id = _trace_utterance_id(chunk)
    return utterance_id.split("_", 1)[0] if "_" in utterance_id else ""


def _turn_index_from_chunk(chunk: TranscriptChunk) -> int:
    utterance = _last_utterance(chunk)
    if utterance is None:
        return 0
    suffix = utterance.item_id.rsplit("_U", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


def _last_utterance(chunk: TranscriptChunk) -> TranscriptItem | None:
    for item in reversed(chunk.items):
        if item.kind == "utterance":
            return item
    return None
