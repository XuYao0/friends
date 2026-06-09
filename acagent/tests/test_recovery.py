import json

from acagent.recovery import rebuild_memory_from_trace


def test_rebuild_memory_from_trace_replays_update_memory_calls(tmp_path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trace_id": "trace_00001",
                        "utterance_id": "U1",
                        "memory_version_after_agent": "mem_00001",
                        "llm_turns": [
                            {
                                "output": {
                                    "tool_call": {
                                        "name": "update_memory",
                                        "arguments": {
                                            "character_updates": [
                                                {
                                                    "character": "Monica",
                                                    "field": "short_term_traits",
                                                    "item": {"text": "Monica is worried."},
                                                }
                                            ],
                                            "source_utterance_ids": ["U1"],
                                        },
                                    }
                                },
                                "tool_result": {
                                    "tool_name": "update_memory",
                                    "ok": True,
                                    "error": None,
                                    "memory_version_before": "mem_00000",
                                    "memory_version_after": "mem_00001",
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "trace_id": "trace_00002",
                        "utterance_id": "U2",
                        "memory_version_after_agent": "mem_00002",
                        "llm_turns": [
                            {
                                "output": {
                                    "tool_call": {
                                        "name": "update_memory",
                                        "arguments": {
                                            "current_event_update": {
                                                "scene_id": "S01E01",
                                                "summary": "Ross reports a separation.",
                                            },
                                            "source_utterance_ids": ["U2"],
                                        },
                                    }
                                },
                                "tool_result": {
                                    "tool_name": "update_memory",
                                    "ok": True,
                                    "error": None,
                                    "memory_version_before": "mem_00001",
                                    "memory_version_after": "mem_00002",
                                },
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    memory = rebuild_memory_from_trace(trace_path, through_chunk=1)

    assert memory.version_id == "mem_00001"
    assert memory.characters["Monica"].short_term_traits[0].text == "Monica is worried."
    assert memory.current_event.summary == ""
