from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from acagent.memory import MemoryState
from acagent.schemas import EventDetail, Utterance


@dataclass(slots=True)
class WorkflowInput:
    current_utterance: Utterance
    batch: list[Utterance]
    local_context: list[Utterance]
    memory: MemoryState
    is_eval_point: bool
    memory_store: Any | None = None
    retrieved_event_details: list[EventDetail] = field(default_factory=list)
