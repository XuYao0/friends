from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acagent.memory import MemoryState
from acagent.schemas import (
    CharacterState,
    CurrentEventState,
    EventDetail,
    EventIndex,
    EvidenceRef,
    MemoryItem,
    RelationshipState,
)


class JsonMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> MemoryState:
        if not self.path.exists():
            return MemoryState()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return _memory_from_dict(data)

    def save(self, memory: MemoryState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(_memory_to_dict(memory), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _memory_to_dict(memory: MemoryState) -> dict[str, Any]:
    data = asdict(memory)
    data["relationships"] = [
        {
            "key": list(key),
            "state": asdict(value),
        }
        for key, value in memory.relationships.items()
    ]
    return data


def _memory_from_dict(data: dict[str, Any]) -> MemoryState:
    memory = MemoryState(version=int(data.get("version", 0)))
    memory.characters = {
        name: _character_state_from_dict(item)
        for name, item in data.get("characters", {}).items()
    }
    memory.relationships = {}
    relationships = data.get("relationships", [])
    if isinstance(relationships, dict):
        relationships = [{"key": key.split("||"), "state": value} for key, value in relationships.items()]
    for item in relationships:
        state = _relationship_state_from_dict(item["state"])
        memory.relationships[tuple(item["key"])] = state
    memory.event_index = {
        event_id: _event_index_from_dict(item)
        for event_id, item in data.get("event_index", {}).items()
    }
    memory.event_details = {
        detail_id: _event_detail_from_dict(item)
        for detail_id, item in data.get("event_details", {}).items()
    }
    memory.current_event = _current_event_from_dict(data.get("current_event", {}))
    memory.applied_sources = [list(item) for item in data.get("applied_sources", [])]
    return memory


def _character_state_from_dict(data: dict[str, Any]) -> CharacterState:
    return CharacterState(
        character=str(data.get("character", "")),
        version=int(data.get("version", 0)),
        recent_events=_items_from_list(data.get("recent_events", [])),
        short_term_traits=_items_from_list(data.get("short_term_traits", [])),
        long_term_traits=_items_from_list(data.get("long_term_traits", [])),
        relationships=_items_from_list(data.get("relationships", [])),
    )


def _relationship_state_from_dict(data: dict[str, Any]) -> RelationshipState:
    return RelationshipState(
        source=str(data.get("source", "")),
        target=str(data.get("target", "")),
        version=int(data.get("version", 0)),
        items=_items_from_list(data.get("items", [])),
    )


def _event_index_from_dict(data: dict[str, Any]) -> EventIndex:
    return EventIndex(
        event_id=str(data.get("event_id", "")),
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        characters=list(data.get("characters", [])),
        keywords=list(data.get("keywords", [])),
        short_summary=str(data.get("short_summary", "")),
        importance=int(data.get("importance", 1)),
        knowledge_scope=str(data.get("knowledge_scope", "")),
        detail_id=data.get("detail_id"),
    )


def _event_detail_from_dict(data: dict[str, Any]) -> EventDetail:
    return EventDetail(
        detail_id=str(data.get("detail_id", "")),
        event_id=str(data.get("event_id", "")),
        description=str(data.get("description", "")),
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        knowledge_scope=str(data.get("knowledge_scope", "")),
    )


def _current_event_from_dict(data: dict[str, Any]) -> CurrentEventState:
    return CurrentEventState(
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        summary=str(data.get("summary", "")),
        characters=list(data.get("characters", [])),
        knowledge_scope=str(data.get("knowledge_scope", "")),
    )


def _items_from_list(items: list[dict[str, Any]]) -> list[MemoryItem]:
    return [_memory_item_from_dict(item) for item in items]


def _memory_item_from_dict(data: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        text=str(data.get("text", "")),
        evidence_refs=[_evidence_ref_from_dict(item) for item in data.get("evidence_refs", [])],
        fact_type=_literal_or_default(data.get("fact_type"), "fact", ("fact", "inference", "uncertain")),
        status=str(data.get("status", "active")),
    )


def _evidence_ref_from_dict(data: dict[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        utterance_id=str(data.get("utterance_id", "")),
        note=str(data.get("note", "")),
    )


def _literal_or_default(value: Any, default: str, allowed: tuple[str, ...]) -> str:
    text = str(value or default)
    return text if text in allowed else default
