import json

from acagent.data_io import FriendsJsonlUtteranceSource
from acagent.transcript import render_transcript


def test_friends_jsonl_source_maps_descriptions_and_utterances(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "Scene: Central Perk, everyone is there."},
        },
        {
            "season": 1,
            "episode": 1,
            "type": "utterance",
            "content": {
                "utterance_id": 1,
                "global_utterance_id": 1,
                "speaker": "Monica",
                "utterance": "There's nothing to tell!",
                "inline_description": ["defensive"],
            },
        },
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "Monica looks away."},
        },
        {
            "season": 1,
            "episode": 1,
            "type": "utterance",
            "content": {
                "utterance_id": 2,
                "global_utterance_id": 2,
                "speaker": "Joey",
                "utterance": "C'mon!",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    utterances = list(FriendsJsonlUtteranceSource(path).iter_utterances())

    assert [utterance.utterance_id for utterance in utterances] == ["S01E01_U000001", "S01E01_U000002"]
    assert utterances[0].scene_id == "S01E01_SC001"
    assert utterances[0].scene_context == "Scene: Central Perk, everyone is there."
    assert utterances[0].stage_direction == "defensive"
    assert utterances[1].scene_context == "Monica looks away."


def test_default_friends_jsonl_source_reads_real_file() -> None:
    utterance = next(FriendsJsonlUtteranceSource().iter_utterances())

    assert utterance.episode_id == "S01E01"
    assert utterance.scene_id == "S01E01_SC001"
    assert utterance.speaker == "Olivia"
    assert utterance.text.startswith("There's nothing to tell")


def test_friends_jsonl_source_includes_descriptions_in_transcript_items(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "Scene: Central Perk."},
        },
        {
            "season": 1,
            "episode": 1,
            "type": "utterance",
            "content": {
                "utterance_id": 1,
                "global_utterance_id": 1,
                "speaker": "Monica",
                "utterance": "One.",
            },
        },
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "They all stare."},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    transcript_items = list(FriendsJsonlUtteranceSource(path).iter_transcript_items())

    assert render_transcript(transcript_items) == (
        "[S01E01_D000001] [description] Scene: Central Perk.\n"
        "[S01E01_U000001] Monica: One.\n"
        "[S01E01_D000002] [description] They all stare."
    )
