from __future__ import annotations

from typing import Any

from acagent.runner.context import WorkflowInput
from acagent.schemas import (
    CharacterUpdate,
    CurrentEventUpdate,
    EventDetail,
    EventIndex,
    EventUpdate,
    EvidenceRef,
    MemoryDelta,
    MemoryItem,
    RelationshipUpdate,
)
from acagent.tools.base import AgentTool


def build_update_memory_tool() -> AgentTool:
    return AgentTool(
        name="update_memory",
        description="Apply character, relationship, event, and current-event memory deltas for the current transcript.",
        input_schema=_UPDATE_MEMORY_INPUT_SCHEMA,
        output_schema={
            "type": "object",
            "properties": {
                "updated": {"type": "boolean"},
                "memory_version_before": {"type": "string"},
                "memory_version_after": {"type": "string"},
            },
            "required": ["updated", "memory_version_before", "memory_version_after"],
        },
        handler=_update_memory,
        is_read_only=False,
        is_concurrency_safe=False,
    )


def _update_memory(workflow_input: WorkflowInput, arguments: dict[str, Any]) -> dict[str, Any]:
    if workflow_input.memory_store is not None:
        workflow_input.memory = workflow_input.memory_store.load()
    delta = _memory_delta_from_dict(arguments)
    if not delta.source_utterance_ids:
        delta.source_utterance_ids = [item.utterance_id for item in workflow_input.batch]
    version_before = workflow_input.memory.version_id
    version_after = workflow_input.memory.apply_delta(delta)
    if workflow_input.memory_store is not None:
        workflow_input.memory_store.save(workflow_input.memory)
    return {
        "updated": True,
        "memory_version_before": version_before,
        "memory_version_after": version_after,
    }


def _memory_delta_from_dict(data: dict[str, Any]) -> MemoryDelta:
    return MemoryDelta(
        character_updates=[
            _character_update_from_dict(item)
            for item in data.get("character_updates", [])
        ],
        relationship_updates=[
            _relationship_update_from_dict(item)
            for item in data.get("relationship_updates", [])
        ],
        event_updates=[
            _event_update_from_dict(item)
            for item in data.get("event_updates", [])
        ],
        current_event_update=(
            _current_event_update_from_dict(data["current_event_update"])
            if data.get("current_event_update")
            else None
        ),
        uncertainties=[str(item) for item in data.get("uncertainties", [])],
        source_utterance_ids=[str(item) for item in data.get("source_utterance_ids", [])],
        mode=str(data.get("mode", "unsupervised")),
    )


def _character_update_from_dict(data: dict[str, Any]) -> CharacterUpdate:
    return CharacterUpdate(
        character=str(data["character"]),
        field=_character_field(data.get("field")),
        item=_memory_item_from_dict(data.get("item", {})),
        operation=_memory_item_operation(data.get("operation")),
        target_index=_optional_int(data.get("target_index")),
    )


def _relationship_update_from_dict(data: dict[str, Any]) -> RelationshipUpdate:
    return RelationshipUpdate(
        source=str(data["source"]),
        target=str(data["target"]),
        item=_memory_item_from_dict(data.get("item", {})),
        operation=_memory_item_operation(data.get("operation")),
        target_index=_optional_int(data.get("target_index")),
    )


def _event_update_from_dict(data: dict[str, Any]) -> EventUpdate:
    detail_data = data.get("detail")
    return EventUpdate(
        index=_event_index_from_dict(data.get("index", {})),
        detail=_event_detail_from_dict(detail_data) if detail_data else None,
    )


def _current_event_update_from_dict(data: dict[str, Any]) -> CurrentEventUpdate:
    return CurrentEventUpdate(
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        summary=str(data.get("summary", "")),
        characters=[str(item) for item in data.get("characters", [])],
        knowledge_scope=str(data.get("knowledge_scope", "")),
    )


def _memory_item_from_dict(data: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        text=str(data.get("text", "")),
        evidence_refs=[
            EvidenceRef(
                utterance_id=str(item.get("utterance_id", "")),
                note=str(item.get("note", "")),
            )
            for item in data.get("evidence_refs", [])
        ],
        fact_type=_literal_or_default(data.get("fact_type"), "fact", ("fact", "inference", "uncertain")),
        status=str(data.get("status", "active")),
    )


def _event_index_from_dict(data: dict[str, Any]) -> EventIndex:
    return EventIndex(
        event_id=str(data["event_id"]),
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        characters=[str(item) for item in data.get("characters", [])],
        keywords=[str(item) for item in data.get("keywords", [])],
        short_summary=str(data.get("short_summary", "")),
        importance=int(data.get("importance", 1)),
        knowledge_scope=str(data.get("knowledge_scope", "")),
        detail_id=data.get("detail_id"),
    )


def _event_detail_from_dict(data: dict[str, Any]) -> EventDetail:
    return EventDetail(
        detail_id=str(data["detail_id"]),
        event_id=str(data["event_id"]),
        description=str(data.get("description", "")),
        scene_id=str(data.get("scene_id", "")),
        time_label=str(data.get("time_label", "")),
        location=str(data.get("location", "")),
        knowledge_scope=str(data.get("knowledge_scope", "")),
    )


def _character_field(value: Any) -> str:
    return _literal_or_default(
        value,
        "recent_events",
        ("recent_events", "short_term_traits", "long_term_traits", "relationships"),
    )


def _literal_or_default(value: Any, default: str, allowed: tuple[str, ...]) -> str:
    text = str(value or default)
    return text if text in allowed else default


def _memory_item_operation(value: Any) -> str:
    return _literal_or_default(value, "append", ("append", "update", "supersede"))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


_MEMORY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "fact_type": {"type": "string", "enum": ["fact", "inference", "uncertain"]},
        "status": {"type": "string"},
        "evidence_refs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "utterance_id": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
    "required": ["text"],
}

_UPDATE_MEMORY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "character_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "field": {
                        "type": "string",
                        "enum": [
                            "recent_events",
                            "short_term_traits",
                            "long_term_traits",
                            "relationships",
                        ],
                    },
                    "item": _MEMORY_ITEM_SCHEMA,
                    "operation": {
                        "type": "string",
                        "enum": ["append", "update"],
                        "description": "append adds a new item; update replaces target_index.",
                    },
                    "target_index": {
                        "type": "integer",
                        "description": "Zero-based index in the selected character field; required for update.",
                    },
                },
                "required": ["character", "field", "item"],
            },
        },
        "relationship_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "item": _MEMORY_ITEM_SCHEMA,
                    "operation": {
                        "type": "string",
                        "enum": ["append", "update"],
                        "description": "append adds a new item; update replaces target_index.",
                    },
                    "target_index": {
                        "type": "integer",
                        "description": "Zero-based index in the relationship items list; required for update.",
                    },
                },
                "required": ["source", "target", "item"],
            },
        },
        "event_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "scene_id": {"type": "string"},
                            "time_label": {"type": "string"},
                            "location": {"type": "string"},
                            "characters": {"type": "array", "items": {"type": "string"}},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "short_summary": {"type": "string"},
                            "importance": {"type": "integer"},
                            "knowledge_scope": {"type": "string"},
                            "detail_id": {"type": "string"},
                        },
                        "required": ["event_id", "short_summary"],
                    },
                    "detail": {
                        "type": "object",
                        "properties": {
                            "detail_id": {"type": "string"},
                            "event_id": {"type": "string"},
                            "description": {"type": "string"},
                            "scene_id": {"type": "string"},
                            "time_label": {"type": "string"},
                            "location": {"type": "string"},
                            "knowledge_scope": {"type": "string"},
                        },
                        "required": ["detail_id", "event_id", "description"],
                    },
                },
                "required": ["index"],
            },
        },
        "current_event_update": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string"},
                "time_label": {"type": "string"},
                "location": {"type": "string"},
                "summary": {"type": "string"},
                "characters": {"type": "array", "items": {"type": "string"}},
                "knowledge_scope": {"type": "string"},
            },
            "required": ["scene_id", "summary"],
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "source_utterance_ids": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["unsupervised", "label_feedback"]},
    },
}
