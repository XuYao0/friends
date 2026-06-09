from __future__ import annotations

import re
from dataclasses import dataclass

from acagent.memory import MemoryState
from acagent.schemas import EventDetail, EventIndex, SearchQuery


@dataclass(slots=True)
class EventSearchResult:
    event: EventIndex
    score: int
    matched_terms: list[str]


class KeywordEventSearcher:
    def __init__(self, memory: MemoryState) -> None:
        self.memory = memory

    def search_events(self, search: SearchQuery) -> list[EventSearchResult]:
        terms = self._terms(search)
        results: list[EventSearchResult] = []
        for event in self.memory.event_index.values():
            score, matched_terms = self._score(event, terms, search.characters)
            if score > 0:
                results.append(EventSearchResult(event=event, score=score, matched_terms=matched_terms))
        results.sort(key=lambda item: (-item.score, -item.event.importance, item.event.event_id))
        return results[: search.top_k]

    def read_event(self, event_id: str) -> EventDetail | None:
        event = self.memory.event_index.get(event_id)
        if event is None or event.detail_id is None:
            return None
        return self.memory.event_details.get(event.detail_id)

    def _terms(self, search: SearchQuery) -> set[str]:
        text_terms = set(re.findall(r"[A-Za-z0-9_']+", search.query.lower()))
        keyword_terms = {item.lower() for item in search.keywords}
        character_terms = {item.lower() for item in search.characters}
        return text_terms | keyword_terms | character_terms

    def _score(
        self,
        event: EventIndex,
        terms: set[str],
        characters: list[str],
    ) -> tuple[int, list[str]]:
        haystack_terms = {
            *[item.lower() for item in event.characters],
            *[item.lower() for item in event.keywords],
            *re.findall(r"[A-Za-z0-9_']+", event.short_summary.lower()),
            event.location.lower(),
            event.scene_id.lower(),
            event.time_label.lower(),
        }
        matched = sorted(terms & haystack_terms)
        score = len(matched)
        character_hits = {item.lower() for item in characters} & {item.lower() for item in event.characters}
        score += 2 * len(character_hits)
        score += max(event.importance - 1, 0)
        return score, matched
