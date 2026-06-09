from acagent.memory import MemoryState
from acagent.retrieval import KeywordEventSearcher
from acagent.schemas import EventDetail, EventIndex, EventUpdate, MemoryDelta, SearchQuery


def test_keyword_event_search_returns_matching_events() -> None:
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            event_updates=[
                EventUpdate(
                    index=EventIndex(
                        event_id="EVT_1",
                        scene_id="SC1",
                        location="Central Perk",
                        characters=["Ross", "Rachel"],
                        keywords=["breakup", "jealousy"],
                        short_summary="Ross reacts to Rachel's date with jealousy.",
                        importance=3,
                        detail_id="D1",
                    ),
                    detail=EventDetail(
                        detail_id="D1",
                        event_id="EVT_1",
                        description="Ross is jealous when Rachel mentions a date.",
                    ),
                )
            ]
        )
    )
    searcher = KeywordEventSearcher(memory)

    results = searcher.search_events(
        SearchQuery(query="Rachel date", characters=["Ross"], keywords=["jealousy"], top_k=3)
    )

    assert [item.event.event_id for item in results] == ["EVT_1"]
    assert searcher.read_event("EVT_1").description == "Ross is jealous when Rachel mentions a date."
