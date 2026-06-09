from __future__ import annotations

from dataclasses import dataclass, field

from acagent.schemas import (
    CharacterState,
    CurrentEventState,
    EventDetail,
    EventIndex,
    MemoryDelta,
    MemoryItem,
    RelationshipState,
)


@dataclass
class MemoryState:
    version: int = 0
    characters: dict[str, CharacterState] = field(default_factory=dict)
    relationships: dict[tuple[str, str], RelationshipState] = field(default_factory=dict)
    event_index: dict[str, EventIndex] = field(default_factory=dict)
    event_details: dict[str, EventDetail] = field(default_factory=dict)
    current_event: CurrentEventState = field(default_factory=CurrentEventState)
    applied_sources: list[list[str]] = field(default_factory=list)

    @property
    def version_id(self) -> str:
        return f"mem_{self.version:05d}"

    def get_or_create_character(self, character: str) -> CharacterState:
        if character not in self.characters:
            self.characters[character] = CharacterState(character=character)
        return self.characters[character]

    def get_or_create_relationship(self, source: str, target: str) -> RelationshipState:
        key = tuple(sorted((source, target)))
        if key not in self.relationships:
            self.relationships[key] = RelationshipState(source=key[0], target=key[1])
        return self.relationships[key]

    def apply_delta(self, delta: MemoryDelta) -> str:
        for update in delta.character_updates:
            state = self.get_or_create_character(update.character)
            _apply_item_update(
                items=getattr(state, update.field),
                item=update.item,
                operation=update.operation,
                target_index=update.target_index,
            )
            state.version += 1

        for update in delta.relationship_updates:
            state = self.get_or_create_relationship(update.source, update.target)
            _apply_item_update(
                items=state.items,
                item=update.item,
                operation=update.operation,
                target_index=update.target_index,
            )
            state.version += 1

        for update in delta.event_updates:
            self.event_index[update.index.event_id] = update.index
            if update.detail is not None:
                self.event_details[update.detail.detail_id] = update.detail

        if delta.current_event_update is not None:
            self.current_event = CurrentEventState(
                scene_id=delta.current_event_update.scene_id,
                time_label=delta.current_event_update.time_label,
                location=delta.current_event_update.location,
                summary=delta.current_event_update.summary,
                characters=delta.current_event_update.characters,
                knowledge_scope=delta.current_event_update.knowledge_scope,
            )

        self.version += 1
        self.applied_sources.append(list(delta.source_utterance_ids))
        return self.version_id


def _apply_item_update(
    *,
    items: list[MemoryItem],
    item: MemoryItem,
    operation: str,
    target_index: int | None,
) -> None:
    if operation == "append":
        items.append(item)
        return
    index = _require_target_index(items, target_index, operation)
    if operation == "update":
        items[index] = item
        return
    if operation == "supersede":
        items[index].status = "superseded"
        items.append(item)
        return
    raise ValueError(f"Unsupported memory item operation: {operation}")


def _require_target_index(
    items: list[MemoryItem],
    target_index: int | None,
    operation: str,
) -> int:
    if target_index is None:
        raise ValueError(f"target_index is required for {operation} memory item updates")
    if target_index < 0 or target_index >= len(items):
        raise IndexError(
            f"target_index {target_index} is out of range for {operation} memory item update"
        )
    return target_index
