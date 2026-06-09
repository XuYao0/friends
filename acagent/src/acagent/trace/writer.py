from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from acagent.trace.logger import AgentTrace


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: AgentTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace_to_record(trace), ensure_ascii=False, sort_keys=True))
            file.write("\n")

    def read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    data = json.loads(stripped)
                    if not isinstance(data, dict):
                        raise ValueError("Trace JSONL line must decode to a JSON object")
                    records.append(data)
        return records


def trace_to_record(trace: AgentTrace) -> dict[str, Any]:
    data = _jsonable(trace)
    if not isinstance(data, dict):
        raise TypeError("Trace must serialize to a JSON object")
    return data


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
