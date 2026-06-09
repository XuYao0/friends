from acagent.memory import MemoryState
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


def test_memory_delta_merge_adds_character_relationship_and_event() -> None:
    memory = MemoryState()
    delta = MemoryDelta(
        character_updates=[
            CharacterUpdate(
                character="Monica",
                field="recent_events",
                item=MemoryItem(
                    text="Monica worries others are judging her date.",
                    evidence_refs=[EvidenceRef("S01E01_U001")],
                    fact_type="inference",
                ),
            )
        ],
        relationship_updates=[
            RelationshipUpdate(
                source="Monica",
                target="Rachel",
                item=MemoryItem(text="They treat each other as close friends."),
            )
        ],
        event_updates=[
            EventUpdate(
                index=EventIndex(
                    event_id="EVT_1",
                    scene_id="SC1",
                    location="Apartment",
                    characters=["Monica", "Rachel"],
                    keywords=["date", "judgment"],
                    short_summary="Monica discusses a date while Rachel listens.",
                    importance=2,
                    detail_id="EVTD_1",
                ),
                detail=EventDetail(
                    detail_id="EVTD_1",
                    event_id="EVT_1",
                    description="Monica discusses a date and appears self-conscious.",
                ),
            )
        ],
        source_utterance_ids=["S01E01_U001", "S01E01_U002"],
    )

    version = memory.apply_delta(delta)

    assert version == "mem_00001"
    assert memory.characters["Monica"].recent_events[0].fact_type == "inference"
    assert ("Monica", "Rachel") in memory.relationships
    assert memory.event_index["EVT_1"].detail_id == "EVTD_1"
    assert memory.event_details["EVTD_1"].description.startswith("Monica discusses")


def test_memory_schema_fields_cover_character_event_and_current_event() -> None:
    memory = MemoryState()
    delta = MemoryDelta(
        character_updates=[
            CharacterUpdate(
                character="Chandler",
                field="long_term_traits",
                item=MemoryItem(text="Uses sarcasm to deflect discomfort.", fact_type="inference"),
            )
        ],
        event_updates=[
            EventUpdate(
                index=EventIndex(
                    event_id="EVT_2",
                    scene_id="SC2",
                    time_label="early scene",
                    location="Central Perk",
                    characters=["Chandler", "Joey"],
                    keywords=["sarcasm"],
                    short_summary="Chandler jokes while avoiding a direct answer.",
                    detail_id="EVTD_2",
                ),
                detail=EventDetail(
                    detail_id="EVTD_2",
                    event_id="EVT_2",
                    description="Chandler avoids answering directly by making a joke.",
                    scene_id="SC2",
                    time_label="early scene",
                    location="Central Perk",
                ),
            )
        ],
        current_event_update=CurrentEventUpdate(
            scene_id="SC2",
            time_label="early scene",
            location="Central Perk",
            summary="The friends are discussing an awkward topic.",
            characters=["Chandler", "Joey"],
            knowledge_scope="Characters in Central Perk hear this.",
        ),
    )

    memory.apply_delta(delta)

    assert memory.characters["Chandler"].long_term_traits[0].text.startswith("Uses sarcasm")
    assert memory.event_index["EVT_2"].scene_id == "SC2"
    assert memory.event_details["EVTD_2"].location == "Central Perk"
    assert memory.current_event.time_label == "early scene"
    assert memory.current_event.characters == ["Chandler", "Joey"]


def test_memory_delta_can_update_existing_character_item() -> None:
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            character_updates=[
                CharacterUpdate(
                    character="Ross",
                    field="short_term_traits",
                    item=MemoryItem(text="Ross seems uncertain about Carol.", fact_type="inference"),
                )
            ]
        )
    )

    memory.apply_delta(
        MemoryDelta(
            character_updates=[
                CharacterUpdate(
                    character="Ross",
                    field="short_term_traits",
                    operation="update",
                    target_index=0,
                    item=MemoryItem(
                        text="Ross is distressed about Carol moving out.",
                        fact_type="fact",
                        evidence_refs=[EvidenceRef("S01E01_U002")],
                    ),
                )
            ]
        )
    )

    traits = memory.characters["Ross"].short_term_traits
    assert len(traits) == 1
    assert traits[0].text == "Ross is distressed about Carol moving out."
    assert traits[0].fact_type == "fact"
    assert memory.characters["Ross"].version == 2


def test_memory_delta_can_supersede_existing_relationship_item() -> None:
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            relationship_updates=[
                RelationshipUpdate(
                    source="Ross",
                    target="Carol",
                    item=MemoryItem(text="Ross may still expect things to improve.", fact_type="uncertain"),
                )
            ]
        )
    )

    memory.apply_delta(
        MemoryDelta(
            relationship_updates=[
                RelationshipUpdate(
                    source="Ross",
                    target="Carol",
                    operation="supersede",
                    target_index=0,
                    item=MemoryItem(
                        text="Ross is dealing with Carol having moved out.",
                        fact_type="fact",
                    ),
                )
            ]
        )
    )

    relationship = memory.relationships[("Carol", "Ross")]
    assert relationship.items[0].status == "superseded"
    assert relationship.items[1].status == "active"
    assert relationship.items[1].text == "Ross is dealing with Carol having moved out."
    assert relationship.version == 2
