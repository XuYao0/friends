from acagent.memory import MemoryState
from acagent.runner import WorkflowInput
from acagent.schemas import (
    CharacterUpdate,
    EventDetail,
    EventIndex,
    EventUpdate,
    MemoryDelta,
    MemoryItem,
    Utterance,
)
from acagent.storage import JsonMemoryStore
from acagent.tools import ToolCall, build_default_registry
from acagent.tools.executor import ToolExecutor


def test_default_tools_accept_empty_arguments() -> None:
    utterance = Utterance("S01E01", "SC1", "U1", 1, "A", "one")
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[utterance],
        memory=MemoryState(),
        is_eval_point=False,
    )
    executor = ToolExecutor(build_default_registry())

    search_result = executor.execute(workflow_input, ToolCall(name="search_events"))
    update_result = executor.execute(workflow_input, ToolCall(name="update_memory"))

    assert search_result.ok
    assert search_result.output == {"events": []}
    assert update_result.ok
    assert update_result.output["memory_version_after"] == "mem_00001"


def test_unknown_tool_returns_structured_error() -> None:
    utterance = Utterance("S01E01", "SC1", "U1", 1, "A", "one")
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[utterance],
        memory=MemoryState(),
        is_eval_point=False,
    )
    executor = ToolExecutor(build_default_registry())

    result = executor.execute(workflow_input, ToolCall(name="missing_tool"))

    assert not result.ok
    assert result.error.code == "unknown_tool"


def test_tools_can_read_and_write_json_memory_store(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            event_updates=[
                EventUpdate(
                    index=EventIndex(
                        event_id="EVT_1",
                        location="Central Perk",
                        characters=["Ross"],
                        keywords=["jealousy"],
                        short_summary="Ross sounds jealous.",
                        detail_id="D1",
                    ),
                    detail=EventDetail(
                        detail_id="D1",
                        event_id="EVT_1",
                        description="Ross sounds jealous during the conversation.",
                    ),
                )
            ]
        )
    )
    store.save(memory)
    utterance = Utterance("S01E01", "SC1", "U2", 2, "Ross", "I'm fine.")
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[utterance],
        memory=MemoryState(),
        is_eval_point=True,
        memory_store=store,
    )
    executor = ToolExecutor(build_default_registry())

    search_result = executor.execute(
        workflow_input,
        ToolCall(name="search_events", arguments={"query": "jealousy", "top_k": 1}),
    )
    update_result = executor.execute(workflow_input, ToolCall(name="update_memory"))

    assert search_result.output["events"][0]["event_id"] == "EVT_1"
    assert update_result.output["memory_version_after"] == "mem_00002"
    assert store.load().version_id == "mem_00002"


def test_update_memory_tool_can_update_existing_character_item() -> None:
    utterance = Utterance("S01E01", "SC1", "U1", 1, "Ross", "Carol moved out.")
    memory = MemoryState()
    memory.apply_delta(
        MemoryDelta(
            character_updates=[
                CharacterUpdate(
                    character="Ross",
                    field="recent_events",
                    item=MemoryItem(text="Ross mentions Carol.", fact_type="fact"),
                )
            ]
        )
    )
    workflow_input = WorkflowInput(
        current_utterance=utterance,
        batch=[utterance],
        local_context=[utterance],
        memory=memory,
        is_eval_point=False,
    )
    executor = ToolExecutor(build_default_registry())

    result = executor.execute(
        workflow_input,
        ToolCall(
            name="update_memory",
            arguments={
                "character_updates": [
                    {
                        "character": "Ross",
                        "field": "recent_events",
                        "operation": "update",
                        "target_index": 0,
                        "item": {
                            "text": "Ross says Carol moved out.",
                            "fact_type": "fact",
                        },
                    }
                ]
            },
        ),
    )

    assert result.ok
    assert memory.characters["Ross"].recent_events[0].text == "Ross says Carol moved out."
    assert len(memory.characters["Ross"].recent_events) == 1


def test_update_memory_tool_schema_hides_internal_supersede_operation() -> None:
    registry = build_default_registry()
    schema = registry.get("update_memory").input_schema

    character_operation_enum = schema["properties"]["character_updates"]["items"]["properties"][
        "operation"
    ]["enum"]
    relationship_operation_enum = schema["properties"]["relationship_updates"]["items"]["properties"][
        "operation"
    ]["enum"]

    assert character_operation_enum == ["append", "update"]
    assert relationship_operation_enum == ["append", "update"]
