import json

from acagent.schemas import Utterance
from acagent.transcript import (
    FriendsTranscriptChunkSource,
    TranscriptBuilder,
    TranscriptItem,
    render_transcript,
)


def test_transcript_builder_formats_minimal_utterance_fields() -> None:
    utterance = Utterance(
        episode_id="S01E01",
        scene_id="S01E01_SC001",
        utterance_id="S01E01_U000001",
        turn_index=1,
        speaker="Monica",
        text="There's nothing to tell!",
        stage_direction="defensive",
        scene_context="Scene: Central Perk.",
        visible_characters=["Rachel", "Ross"],
    )

    transcript = TranscriptBuilder().render([utterance])

    assert transcript == "[S01E01_U000001] Monica: There's nothing to tell!"
    assert "defensive" not in transcript
    assert "Central Perk" not in transcript
    assert "Rachel" not in transcript
    assert "S01E01_SC001" not in transcript


def test_transcript_builder_formats_multiple_utterances_stably() -> None:
    utterances = [
        Utterance("S01E01", "SC1", "U1", 1, "Monica", "One."),
        Utterance("S01E01", "SC1", "U2", 2, "Joey", "Two."),
    ]

    transcript = render_transcript(utterances)

    assert transcript == "[U1] Monica: One.\n[U2] Joey: Two."


def test_transcript_builder_includes_description_items() -> None:
    transcript = render_transcript(
        [
            TranscriptItem.description("D1", "Scene: Central Perk."),
            Utterance("S01E01", "SC1", "U1", 1, "Monica", "One."),
        ]
    )

    assert transcript == "[D1] [description] Scene: Central Perk.\n[U1] Monica: One."


def test_transcript_builder_keeps_one_line_per_utterance() -> None:
    utterance = Utterance(
        episode_id="S01E01",
        scene_id="SC1",
        utterance_id="U1",
        turn_index=1,
        speaker="Monica\nGeller",
        text="There is\nnothing   to tell!",
    )

    transcript = TranscriptBuilder().render([utterance])

    assert transcript == "[U1] Monica Geller: There is nothing to tell!"


def test_friends_transcript_chunk_source_yields_update_chunk_at_batch_size(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "Scene: Central Perk."},
        },
        _utterance_record(1, "Monica", "One."),
        _utterance_record(2, "Joey", "Two."),
    ]
    _write_jsonl(path, records)

    chunks = list(FriendsTranscriptChunkSource(path, batch_size=2).iter_chunks())

    assert len(chunks) == 1
    assert chunks[0].status == "update_memory"
    assert chunks[0].utterance_count == 2
    assert chunks[0].speakers == ["Joey", "Monica"]
    assert chunks[0].target_utterance_id is None
    assert chunks[0].transcript == (
        "[S01E01_D000001] [description] Scene: Central Perk.\n"
        "[S01E01_U000001] Monica: One.\n"
        "[S01E01_U000002] Joey: Two."
    )


def test_friends_transcript_chunk_source_yields_label_chunk_at_selected_record(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "They all stare."},
        },
        _utterance_record(1, "Chandler", "Sounds like a date to me."),
        {
            **_utterance_record(2, "Ross", "Hi."),
            "long_context_selected": True,
            "meld": {"emotion": ["neutral"], "sentiment": ["neutral"]},
        },
        _utterance_record(3, "Joey", "Hello."),
    ]
    _write_jsonl(path, records)

    chunks = list(FriendsTranscriptChunkSource(path, batch_size=20).iter_chunks())

    assert chunks[0].status == "label"
    assert chunks[0].target_utterance_id == "S01E01_U000002"
    assert chunks[0].speakers == ["Chandler", "Ross"]
    assert chunks[0].meld == {"emotion": ["neutral"], "sentiment": ["neutral"]}
    assert chunks[0].transcript == (
        "[S01E01_D000001] [description] They all stare.\n"
        "[S01E01_U000001] Chandler: Sounds like a date to me.\n"
        "[S01E01_U000002] Ross: Hi."
    )
    assert chunks[1].status == "update_memory"
    assert chunks[1].speakers == ["Joey"]
    assert chunks[1].transcript == "[S01E01_U000003] Joey: Hello."


def test_friends_transcript_chunk_source_does_not_label_unselected_meld_record(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        _utterance_record(1, "Chandler", "One."),
        {
            **_utterance_record(2, "Ross", "Two."),
            "meld": {"emotion": ["neutral"], "sentiment": ["neutral"]},
        },
        _utterance_record(3, "Joey", "Three."),
    ]
    _write_jsonl(path, records)

    chunks = list(FriendsTranscriptChunkSource(path, batch_size=20).iter_chunks())

    assert len(chunks) == 1
    assert chunks[0].status == "update_memory"
    assert chunks[0].target_utterance_id is None
    assert chunks[0].utterance_count == 3


def test_friends_transcript_chunk_source_stops_at_max_utterances(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {
            "season": 1,
            "episode": 1,
            "type": "description",
            "content": {"description": "Scene: Central Perk."},
        },
        _utterance_record(1, "Monica", "One."),
        _utterance_record(2, "Joey", "Two."),
        _utterance_record(3, "Ross", "Three."),
    ]
    _write_jsonl(path, records)

    chunks = list(
        FriendsTranscriptChunkSource(path, batch_size=20, max_utterances=2).iter_chunks()
    )

    assert len(chunks) == 1
    assert chunks[0].status == "update_memory"
    assert chunks[0].utterance_count == 2
    assert chunks[0].transcript == (
        "[S01E01_D000001] [description] Scene: Central Perk.\n"
        "[S01E01_U000001] Monica: One.\n"
        "[S01E01_U000002] Joey: Two."
    )
    assert "Three" not in chunks[0].transcript


def _utterance_record(utterance_id: int, speaker: str, text: str) -> dict[str, object]:
    return {
        "season": 1,
        "episode": 1,
        "type": "utterance",
        "content": {
            "utterance_id": utterance_id,
            "global_utterance_id": utterance_id,
            "speaker": speaker,
            "utterance": text,
        },
    }


def _write_jsonl(path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
