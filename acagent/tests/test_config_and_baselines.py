from pathlib import Path

from acagent.baselines import keyword_emotion_predictor
from acagent.config import ExperimentConfig
from acagent.memory import MemoryState
from acagent.runner import WorkflowInput, build_workflow_runner_from_config
from acagent.schemas import Emotion, EventIndex, EventUpdate, MemoryDelta, Utterance
from acagent.tools import ToolCall
from acagent.tools.executor import ToolExecutor


def test_experiment_config_reads_simple_yaml() -> None:
    config = ExperimentConfig.from_yaml(Path("acagent/configs/debug.yaml"))

    assert config.batch_size == 5
    assert config.reasoning_model == "deepseek-v4-pro"
    assert config.temperature == 0
    assert config.max_tokens == 4096
    assert config.max_tool_calls == 4


def test_workflow_runner_uses_experiment_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = ExperimentConfig.from_mapping(
        {
            "transcript_path": "custom.jsonl",
            "batch_size": 7,
            "max_utterances": 200,
            "start_chunk_index": 237,
            "cheap_update_model": "cheap-model",
            "reasoning_model": "reasoning-model",
            "temperature": 0.2,
            "max_tokens": 99,
            "event_search_top_k": 2,
            "max_read_events": 1,
            "max_tool_calls": 3,
            "memory_update_prompt_path": "acagent/prompts/memory_update.md",
            "emotion_labeling_prompt_path": "acagent/prompts/emotion_labeling.md",
            "output_dir": str(tmp_path / "outputs"),
        }
    )

    runner = build_workflow_runner_from_config(config, client_factory=FakeDeepSeekFactory)

    assert runner.config.transcript_path == "custom.jsonl"
    assert runner.config.batch_size == 7
    assert runner.config.max_utterances == 200
    assert runner.config.start_chunk_index == 237
    assert runner.memory_update_loop.max_read_events == 1
    assert runner.memory_update_loop.max_tool_calls == 3
    assert runner.label_loop.max_read_events == 1
    assert runner.label_loop.max_tool_calls == 3
    assert runner.memory_update_loop.llm_client.config.model == "cheap-model"
    assert runner.label_loop.llm_client.config.model == "reasoning-model"
    assert runner.label_loop.llm_client.config.temperature == 0.2
    assert runner.label_loop.llm_client.config.max_tokens == 99
    assert runner.memory_store.path == tmp_path / "outputs" / "memory.json"
    assert runner.prediction_store.path == tmp_path / "outputs" / "predictions.jsonl"
    assert runner.trace_writer.path == tmp_path / "outputs" / "traces.jsonl"


def test_configured_event_search_top_k_is_registry_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    config = ExperimentConfig.from_mapping(
        {
            "event_search_top_k": 1,
            "output_dir": str(tmp_path / "outputs"),
        }
    )
    runner = build_workflow_runner_from_config(config, client_factory=FakeDeepSeekFactory)
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            event_updates=[
                EventUpdate(index=EventIndex(event_id="E1", keywords=["date"], short_summary="one")),
                EventUpdate(index=EventIndex(event_id="E2", keywords=["date"], short_summary="two")),
            ]
        )
    )
    utterance = Utterance("S01E01", "SC1", "U1", 1, "Ross", "date")
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[],
        memory=memory,
        is_eval_point=False,
    )
    executor = ToolExecutor(runner.memory_update_loop.registry)

    result = executor.execute(
        workflow_input,
        ToolCall(name="search_events", arguments={"query": "date"}),
    )

    assert result.ok
    assert len(result.output["events"]) == 1


def test_keyword_baseline_predicts_surface_happiness() -> None:
    utterance = Utterance("S01E01", "SC1", "U1", 1, "Rachel", "This is great!")
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[utterance],
        memory=MemoryState(),
        is_eval_point=True,
    )

    prediction = keyword_emotion_predictor(workflow_input)

    assert prediction.emotions == [Emotion.HAPPINESS]


class FakeDeepSeekFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
