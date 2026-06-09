from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    transcript_path: str = "screenplays/friends_records_renamed_with_selected.jsonl"
    batch_size: int = 20
    max_utterances: int | None = None
    start_chunk_index: int = 1
    local_context_window: int = 12
    cheap_update_model: str = "deepseek-v4-pro"
    reasoning_model: str = "deepseek-v4-pro"
    judge_model: str = "deepseek-v4-pro"
    temperature: float | None = None
    max_tokens: int | None = None
    event_search_top_k: int = 5
    max_read_events: int = 3
    max_tool_calls: int = 8
    memory_update_mode: str = "unsupervised"
    prompt_version: str = "v0.1"
    memory_update_prompt_path: str = "acagent/prompts/memory_update.md"
    emotion_labeling_prompt_path: str = "acagent/prompts/emotion_labeling.md"
    output_dir: str = "acagent_outputs/default"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentConfig":
        fields = cls.__dataclass_fields__
        known = {key: value for key, value in data.items() if key in fields}
        return cls(**known)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value == "":
            continue
        data[key.strip()] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
