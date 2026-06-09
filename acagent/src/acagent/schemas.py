from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Emotion(StrEnum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    CONTENTMENT = "contentment"
    RELIEF = "relief"
    INTEREST = "interest"
    CONTEMPT = "contempt"
    SHAME = "shame"
    GUILT = "guilt"
    EMBARRASSMENT = "embarrassment"
    NEUTRAL = "neutral"


class Intensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NONE = "none"


FactType = Literal["fact", "inference", "uncertain"]
EventStatus = Literal["open", "resolved", "background"]
MemoryUpdateMode = Literal["unsupervised", "label_feedback"]
MemoryItemOperation = Literal["append", "update", "supersede"]


@dataclass(slots=True)
class Utterance:
    episode_id: str
    scene_id: str
    utterance_id: str
    turn_index: int
    speaker: str
    text: str
    stage_direction: str = ""
    scene_context: str = ""
    visible_characters: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceRef:
    utterance_id: str
    note: str = ""


@dataclass(slots=True)
class MemoryItem:
    text: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    fact_type: FactType = "fact"
    status: str = "active"


@dataclass(slots=True)
class CharacterState:
    character: str
    version: int = 0
    recent_events: list[MemoryItem] = field(default_factory=list)
    short_term_traits: list[MemoryItem] = field(default_factory=list)
    long_term_traits: list[MemoryItem] = field(default_factory=list)
    relationships: list[MemoryItem] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipState:
    source: str
    target: str
    version: int = 0
    items: list[MemoryItem] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.source, self.target)))


@dataclass(slots=True)
class EventIndex:
    event_id: str
    scene_id: str = ""
    time_label: str = ""
    location: str = ""
    characters: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    short_summary: str = ""
    importance: int = 1
    knowledge_scope: str = ""
    detail_id: str | None = None


@dataclass(slots=True)
class EventDetail:
    detail_id: str
    event_id: str
    description: str
    scene_id: str = ""
    time_label: str = ""
    location: str = ""
    knowledge_scope: str = ""


@dataclass(slots=True)
class CurrentEventState:
    scene_id: str = ""
    time_label: str = ""
    location: str = ""
    summary: str = ""
    characters: list[str] = field(default_factory=list)
    knowledge_scope: str = ""


@dataclass(slots=True)
class CharacterUpdate:
    character: str
    field: Literal[
        "recent_events",
        "short_term_traits",
        "long_term_traits",
        "relationships",
    ]
    item: MemoryItem
    operation: MemoryItemOperation = "append"
    target_index: int | None = None


@dataclass(slots=True)
class RelationshipUpdate:
    source: str
    target: str
    item: MemoryItem
    operation: MemoryItemOperation = "append"
    target_index: int | None = None


@dataclass(slots=True)
class EventUpdate:
    index: EventIndex
    detail: EventDetail | None = None


@dataclass(slots=True)
class CurrentEventUpdate:
    scene_id: str
    summary: str
    time_label: str = ""
    location: str = ""
    characters: list[str] = field(default_factory=list)
    knowledge_scope: str = ""


@dataclass(slots=True)
class MemoryDelta:
    character_updates: list[CharacterUpdate] = field(default_factory=list)
    relationship_updates: list[RelationshipUpdate] = field(default_factory=list)
    event_updates: list[EventUpdate] = field(default_factory=list)
    current_event_update: CurrentEventUpdate | None = None
    uncertainties: list[str] = field(default_factory=list)
    source_utterance_ids: list[str] = field(default_factory=list)
    mode: MemoryUpdateMode = "unsupervised"


@dataclass(slots=True)
class EmotionPrediction:
    emotions: list[Emotion]
    intensities: list[Intensity]
    analysis: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.emotions:
            raise ValueError("emotions must contain at least one label")
        if not self.intensities:
            raise ValueError("intensities must contain at least one label")
        if len(self.emotions) != len(self.intensities):
            raise ValueError("emotions and intensities must have the same length")
        if Emotion.NEUTRAL in self.emotions:
            if self.emotions != [Emotion.NEUTRAL] or self.intensities != [Intensity.NONE]:
                raise ValueError("neutral must be the only emotion and must use intensity none")
        for emotion, intensity in zip(self.emotions, self.intensities, strict=True):
            if emotion != Emotion.NEUTRAL and intensity == Intensity.NONE:
                raise ValueError("non-neutral emotions cannot use intensity none")


@dataclass(slots=True)
class SearchQuery:
    query: str
    characters: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    time_range: tuple[str, str] | None = None
    top_k: int = 5
    reason: str = ""


@dataclass(slots=True)
class PredictionRecord:
    utterance_id: str
    memory_version: str
    prediction: EmotionPrediction
    trace_id: str
