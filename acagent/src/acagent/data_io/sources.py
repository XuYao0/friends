from __future__ import annotations

from pathlib import Path

from acagent.data_io.stream import FriendsJsonlUtteranceSource, UtteranceSource


DEFAULT_FRIENDS_RECORDS_PATH = Path("screenplays/friends_records_renamed_with_selected.jsonl")


def build_default_utterance_source() -> UtteranceSource:
    return FriendsJsonlUtteranceSource(DEFAULT_FRIENDS_RECORDS_PATH)
