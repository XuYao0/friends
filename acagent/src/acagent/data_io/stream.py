from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from acagent.schemas import Utterance
from acagent.transcript import TranscriptItem


class UtteranceSource(Protocol):
    def iter_utterances(self) -> Iterator[Utterance]:
        ...


class EpisodeLoader:
    """Loads utterances from JSON or JSONL files without reordering them."""

    def load_jsonl(self, path: str | Path) -> list[Utterance]:
        utterances: list[Utterance] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    utterances.append(self._from_mapping(json.loads(line)))
        return utterances

    def load_json(self, path: str | Path) -> list[Utterance]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("utterances", [])
        return [self._from_mapping(item) for item in data]

    def _from_mapping(self, item: dict[str, Any]) -> Utterance:
        return Utterance(
            episode_id=str(item["episode_id"]),
            scene_id=str(item["scene_id"]),
            utterance_id=str(item["utterance_id"]),
            turn_index=int(item["turn_index"]),
            speaker=str(item["speaker"]),
            text=str(item["text"]),
            stage_direction=str(item.get("stage_direction", "")),
            scene_context=str(item.get("scene_context", "")),
            visible_characters=list(item.get("visible_characters", [])),
        )


class FriendsJsonlUtteranceSource:
    """Streams utterances from the default selected Friends JSONL file."""

    def __init__(
        self,
        path: str | Path = "screenplays/friends_records_renamed_with_selected.jsonl",
    ) -> None:
        self.path = Path(path)

    def iter_utterances(self) -> Iterator[Utterance]:
        scene_context = ""
        scene_number_by_episode: dict[str, int] = {}
        current_scene_by_episode: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                episode_id = _episode_id(record)
                if record.get("type") == "description":
                    description = str(record.get("content", {}).get("description", ""))
                    if _starts_new_scene(description, episode_id not in current_scene_by_episode):
                        scene_number_by_episode[episode_id] = scene_number_by_episode.get(episode_id, 0) + 1
                        current_scene_by_episode[episode_id] = _scene_id(
                            episode_id,
                            scene_number_by_episode[episode_id],
                        )
                        scene_context = description
                    elif description:
                        scene_context = description
                    continue

                if record.get("type") != "utterance":
                    continue

                if episode_id not in current_scene_by_episode:
                    scene_number_by_episode[episode_id] = 1
                    current_scene_by_episode[episode_id] = _scene_id(episode_id, 1)

                content = record.get("content", {})
                global_utterance_id = content.get("global_utterance_id", content.get("utterance_id"))
                yield Utterance(
                    episode_id=episode_id,
                    scene_id=current_scene_by_episode[episode_id],
                    utterance_id=f"{episode_id}_U{int(global_utterance_id):06d}",
                    turn_index=int(global_utterance_id),
                    speaker=str(content.get("speaker", "")),
                    text=str(content.get("utterance", "")),
                    stage_direction=_inline_description(content),
                    scene_context=scene_context,
                    visible_characters=[],
                )

    def iter_transcript_items(self) -> Iterator[TranscriptItem]:
        description_number_by_episode: dict[str, int] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                episode_id = _episode_id(record)
                if record.get("type") == "description":
                    description = str(record.get("content", {}).get("description", ""))
                    if description:
                        description_number_by_episode[episode_id] = (
                            description_number_by_episode.get(episode_id, 0) + 1
                        )
                        yield TranscriptItem.description(
                            item_id=f"{episode_id}_D{description_number_by_episode[episode_id]:06d}",
                            text=description,
                        )
                    continue

                if record.get("type") != "utterance":
                    continue

                content = record.get("content", {})
                global_utterance_id = content.get("global_utterance_id", content.get("utterance_id"))
                yield TranscriptItem(
                    item_id=f"{episode_id}_U{int(global_utterance_id):06d}",
                    kind="utterance",
                    speaker=str(content.get("speaker", "")),
                    text=str(content.get("utterance", "")),
                )


class UtteranceStream:
    """Online iterator over utterances.

    The stream intentionally exposes only forward iteration. Consumers should
    keep their own bounded context windows instead of peeking ahead.
    """

    def __init__(self, utterances: Iterable[Utterance]) -> None:
        self._utterances = iter(utterances)
        self.seen_count = 0
        self.last_utterance_id: str | None = None

    def __iter__(self) -> Iterator[Utterance]:
        return self

    def __next__(self) -> Utterance:
        utterance = next(self._utterances)
        self.seen_count += 1
        self.last_utterance_id = utterance.utterance_id
        return utterance


class EvalPointLoader:
    def __init__(self, eval_utterance_ids: Iterable[str]) -> None:
        self.eval_utterance_ids = set(eval_utterance_ids)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalPointLoader":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            ids = data.get("eval_utterance_ids", data.get("utterance_ids", []))
        else:
            ids = data
        return cls(str(item) for item in ids)

    def is_eval_point(self, utterance_id: str) -> bool:
        return utterance_id in self.eval_utterance_ids


def _episode_id(record: dict[str, Any]) -> str:
    return f"S{int(record.get('season', 0)):02d}E{int(record.get('episode', 0)):02d}"


def _scene_id(episode_id: str, scene_number: int) -> str:
    return f"{episode_id}_SC{scene_number:03d}"


def _starts_new_scene(description: str, first_description_in_episode: bool) -> bool:
    return first_description_in_episode or description.strip().lower().startswith("scene:")


def _inline_description(content: dict[str, Any]) -> str:
    inline = content.get("inline_description", [])
    if isinstance(inline, list):
        return " ".join(str(item) for item in inline)
    return str(inline or "")
