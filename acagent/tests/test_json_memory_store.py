from acagent.memory import MemoryState
from acagent.schemas import (
    CharacterUpdate,
    EventDetail,
    EventIndex,
    EventUpdate,
    MemoryDelta,
    MemoryItem,
)
from acagent.storage import JsonMemoryStore


def test_json_memory_store_roundtrip(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            character_updates=[
                CharacterUpdate(
                    character="Phoebe",
                    field="short_term_traits",
                    item=MemoryItem(text="Curious about the strange situation."),
                )
            ],
            event_updates=[
                EventUpdate(
                    index=EventIndex(
                        event_id="EVT_1",
                        scene_id="SC1",
                        location="Apartment",
                        characters=["Phoebe"],
                        keywords=["curious"],
                        short_summary="Phoebe notices something strange.",
                        detail_id="D1",
                    ),
                    detail=EventDetail(
                        detail_id="D1",
                        event_id="EVT_1",
                        description="Phoebe notices something strange and asks about it.",
                    ),
                )
            ],
        )
    )

    store.save(memory)
    loaded = store.load()

    assert loaded.version_id == "mem_00001"
    assert loaded.characters["Phoebe"].short_term_traits[0].text.startswith("Curious")
    assert loaded.event_index["EVT_1"].detail_id == "D1"
    assert loaded.event_details["D1"].description.startswith("Phoebe notices")
