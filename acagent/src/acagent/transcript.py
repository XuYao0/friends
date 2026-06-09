from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from acagent.schemas import Utterance


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    item_id: str
    kind: Literal["description", "utterance"]
    text: str
    speaker: str = ""

    @classmethod
    def from_utterance(cls, utterance: Utterance) -> "TranscriptItem":
        return cls(
            item_id=utterance.utterance_id,
            kind="utterance",
            speaker=utterance.speaker,
            text=utterance.text,
        )

    @classmethod
    def description(cls, item_id: str, text: str) -> "TranscriptItem":
        return cls(item_id=item_id, kind="description", text=text)


@dataclass(frozen=True, slots=True)
class TranscriptBuilder:
    """Formats transcript items as plain transcript text.

    This module intentionally does not add task instructions. Memory update,
    emotion labeling, and other prompt builders should append their own
    instructions after obtaining the transcript.
    """

    def render(self, items: Iterable[Utterance | TranscriptItem]) -> str:
        return "\n".join(self._render_line(item) for item in items)

    def _render_line(self, item: Utterance | TranscriptItem) -> str:
        transcript_item = _to_transcript_item(item)
        text = _single_line(transcript_item.text)
        if transcript_item.kind == "description":
            return f"[{transcript_item.item_id}] [description] {text}"
        speaker = _single_line(transcript_item.speaker)
        return f"[{transcript_item.item_id}] {speaker}: {text}"


def render_transcript(items: Iterable[Utterance | TranscriptItem]) -> str:
    return TranscriptBuilder().render(items)


ChunkStatus = Literal["update_memory", "label"]


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    status: ChunkStatus
    transcript: str
    items: list[TranscriptItem]
    utterance_count: int
    speakers: list[str]
    target_utterance_id: str | None = None
    meld: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FriendsTranscriptChunkSource:
    path: Path
    batch_size: int = 20
    max_utterances: int | None = None
    builder: TranscriptBuilder = TranscriptBuilder()

    def __init__(
        self,
        path: str | Path = "screenplays/friends_records_renamed_with_selected.jsonl",
        batch_size: int = 20,
        max_utterances: int | None = None,
        builder: TranscriptBuilder | None = None,
    ) -> None:
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "max_utterances", max_utterances)
        object.__setattr__(self, "builder", builder or TranscriptBuilder())

    def iter_chunks(self) -> Iterable[TranscriptChunk]:
        buffer: list[TranscriptItem] = []
        utterance_count = 0
        total_utterance_count = 0
        description_number_by_episode: dict[str, int] = {}

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                episode_id = _episode_id(record)
                item = self._record_to_item(record, episode_id, description_number_by_episode)
                if item is None:
                    continue

                buffer.append(item)
                stop_after_current_record = False
                if item.kind == "utterance":
                    utterance_count += 1
                    total_utterance_count += 1
                    stop_after_current_record = (
                        self.max_utterances is not None
                        and total_utterance_count >= self.max_utterances
                    )

                if item.kind == "utterance" and _is_label_record(record):
                    yield self._build_chunk(
                        status="label",
                        items=buffer,
                        utterance_count=utterance_count,
                        target_utterance_id=item.item_id,
                        meld=record.get("meld"),
                    )
                    buffer = []
                    utterance_count = 0
                    if stop_after_current_record:
                        break
                    continue

                if utterance_count >= self.batch_size:
                    yield self._build_chunk(
                        status="update_memory",
                        items=buffer,
                        utterance_count=utterance_count,
                    )
                    buffer = []
                    utterance_count = 0
                    if stop_after_current_record:
                        break

                if stop_after_current_record:
                    break

        if buffer:
            yield self._build_chunk(
                status="update_memory",
                items=buffer,
                utterance_count=utterance_count,
            )

    def _record_to_item(
        self,
        record: dict[str, Any],
        episode_id: str,
        description_number_by_episode: dict[str, int],
    ) -> TranscriptItem | None:
        if record.get("type") == "description":
            description = str(record.get("content", {}).get("description", ""))
            if not description:
                return None
            description_number_by_episode[episode_id] = (
                description_number_by_episode.get(episode_id, 0) + 1
            )
            return TranscriptItem.description(
                item_id=f"{episode_id}_D{description_number_by_episode[episode_id]:06d}",
                text=description,
            )

        if record.get("type") != "utterance":
            return None

        content = record.get("content", {})
        global_utterance_id = content.get("global_utterance_id", content.get("utterance_id"))
        return TranscriptItem(
            item_id=f"{episode_id}_U{int(global_utterance_id):06d}",
            kind="utterance",
            speaker=str(content.get("speaker", "")),
            text=str(content.get("utterance", "")),
        )

    def _build_chunk(
        self,
        *,
        status: ChunkStatus,
        items: list[TranscriptItem],
        utterance_count: int,
        target_utterance_id: str | None = None,
        meld: dict[str, Any] | None = None,
    ) -> TranscriptChunk:
        chunk_items = list(items)
        return TranscriptChunk(
            status=status,
            transcript=self.builder.render(chunk_items),
            items=chunk_items,
            utterance_count=utterance_count,
            speakers=_speakers(chunk_items),
            target_utterance_id=target_utterance_id,
            meld=meld,
        )


def _to_transcript_item(item: Utterance | TranscriptItem) -> TranscriptItem:
    if isinstance(item, TranscriptItem):
        return item
    return TranscriptItem.from_utterance(item)


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _speakers(items: Iterable[TranscriptItem]) -> list[str]:
    return sorted(
        {
            _single_line(item.speaker)
            for item in items
            if item.kind == "utterance" and _single_line(item.speaker)
        }
    )


def _episode_id(record: dict[str, Any]) -> str:
    return f"S{int(record.get('season', 0)):02d}E{int(record.get('episode', 0)):02d}"


def _is_label_record(record: dict[str, Any]) -> bool:
    return bool(record.get("long_context_selected"))
