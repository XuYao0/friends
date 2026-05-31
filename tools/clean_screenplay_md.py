#!/usr/bin/env python3
"""Parse Friends screenplay Markdown into JSON/JSONL event data."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = (
    Path("screenplays")
    / "converted_chunks"
    / "老友记_1-10季剧本_by_episode"
)
DEFAULT_OUTPUT_DIR = Path("screenplays") / "converted_chunks" / "cleaned"
EPISODE_RE = re.compile(r"^(?P<index>\d+)_S(?P<season>\d+)E(?P<episode>\d+)\.md$")
SPEAKER_AT_START_RE = re.compile(r"^([A-Z0-9][A-Za-z0-9 .,'’\"&/#-]{0,55}):\s*(.*)$")
SPEAKER_WITH_INLINE_DESCRIPTION_RE = re.compile(
    r"^([A-Z0-9][A-Za-z0-9 .,'’\"&/#-]{0,55})\s+\*\((.*?)\)\*:\s*(.*)$"
)
SPEAKER_BOUNDARY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z0-9][A-Za-z0-9 .,'’\"&/#-]{0,55}):")
BRACKET_DESCRIPTION_RE = re.compile(r"\*\[([^\]]+)\]\*")
INLINE_DESCRIPTION_RE = re.compile(r"\*\((.*?)\)\*|\*\[([^\]]+)\]\*")
PLAIN_SCENE_RE = re.compile(r"^\[([Ss]cene:[^\]]+)\]$")
MALFORMED_SCENE_RE = re.compile(r"^\[([Ss]cene:.*?)[\]\)]?$")
METADATA_PREFIXES = (
    "Written by:",
    "Transcribed by:",
    "Additional transcribing by:",
    "Teleplay by:",
    "Story by:",
    "Screenplay by",
    "Originally written by",
    "Trascribed by",
    "With Minor Adjustments by:",
    "Aired ",
    "Directed by:",
    "Transcript by:",
    "Produced by",
    "Final check by",
)
METADATA_CONTAINS = (
    " on the dvds / videos shown below",
    "written by",
    "story by",
    "teleplay by",
    "transcribed by",
    "trascribed by",
    "transcript by",
    "htmled by",
    "html by",
    "converted to html",
    "minor additions",
    "minor modifications",
    "minor adjustments",
    "special thanks",
    "friends series",
    "amazon",
    "dvd region",
    "pal dvds",
    "tv standard",
    "other tv shows",
    "friends tv show season guide",
    "season 1",
    "season 2",
    "season 3",
    "season 4",
    "season 5",
    "season 6",
    "season 7",
    "season 8",
    "season 9",
    "season 10",
    "visit friends pic gallery",
    "pre-order",
    "friends the complete",
    "friends trivia board game",
    "great present for friends fans",
    "for april, 2003",
    "dedicated to the great work",
)
CREDIT_STAGE_LINES = {"Opening Credits", "Commercial Break", "Closing Credits"}
REJECT_SPEAKERS = {
    "Written by",
    "Transcribed by",
    "Additional transcribing by",
    "Teleplay by",
    "Story by",
    "Note",
}


@dataclass
class WarningItem:
    file: str
    line_no: int
    type: str
    message: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_no": self.line_no,
            "type": self.type,
            "message": self.message,
            "text": self.text,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Friends episode Markdown into event JSONL and episode JSON."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jsonl-name", default="friends_records.jsonl")
    parser.add_argument("--json-name", default="friends_episodes.json")
    parser.add_argument("--warnings-name", default="friends_parse_warnings.jsonl")
    parser.add_argument(
        "--start-index",
        type=int,
        help="Only parse files whose leading NNN index is >= this value.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Only parse files whose leading NNN index is <= this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files instead of appending to existing JSONL outputs.",
    )
    parser.add_argument(
        "--include-front-matter",
        action="store_true",
        help="Also attempt to parse 000_front_matter.md. Off by default.",
    )
    return parser


def episode_files(input_dir: Path, include_front_matter: bool) -> list[Path]:
    paths = []
    for path in sorted(input_dir.glob("*.md")):
        if EPISODE_RE.match(path.name):
            paths.append(path)
        elif include_front_matter and path.name == "000_front_matter.md":
            paths.append(path)
    if not paths:
        raise SystemExit(f"No episode Markdown files found in {input_dir}")
    return paths


def select_episode_files(
    paths: list[Path],
    start_index: int | None,
    end_index: int | None,
) -> list[Path]:
    selected = []
    for path in paths:
        meta = episode_meta(path)
        index = meta["index"]
        if index is None:
            selected.append(path)
            continue
        if start_index is not None and index < start_index:
            continue
        if end_index is not None and index > end_index:
            continue
        selected.append(path)
    if not selected:
        raise SystemExit("No episode Markdown files matched the selected index range")
    return selected


def episode_meta(path: Path) -> dict[str, int | str | None]:
    match = EPISODE_RE.match(path.name)
    if not match:
        return {
            "index": None,
            "season": None,
            "episode": None,
            "filename": path.name,
        }
    return {
        "index": int(match.group("index")),
        "season": int(match.group("season")),
        "episode": int(match.group("episode")),
        "filename": path.name,
    }


def warn(
    warnings: list[WarningItem],
    path: Path,
    line_no: int,
    warning_type: str,
    message: str,
    text: str,
) -> None:
    warnings.append(
        WarningItem(
            file=path.as_posix(),
            line_no=line_no,
            type=warning_type,
            message=message,
            text=text,
        )
    )


def is_metadata(line: str) -> bool:
    lower = line.lower()
    return (
        line.startswith(METADATA_PREFIXES)
        or line.startswith("## ")
        or lower in {"new!", "结尾"}
        or lower.startswith("the one ")
        or re.fullmatch(r"[-_\\\s]{8,}", line)
        or re.fullmatch(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}",
            lower,
        )
        or any(marker in lower for marker in METADATA_CONTAINS)
    )


def is_end_marker(line: str) -> bool:
    normalized = line.strip().strip("*").strip().strip("[]()").strip().lower()
    return normalized in {"end", "the end"}


def is_footer_marker(line: str) -> bool:
    return line.startswith("# ![") or line.startswith("![")


def is_stage_context(content: str, before_text: str) -> bool:
    normalized = " ".join(content.strip().split())
    lower = normalized.lower()

    # If it appears inside a dialogue prefix, it is usually an inline action.
    if SPEAKER_AT_START_RE.match(before_text.strip()):
        if not lower.startswith(("scene:", "time lapse", "cut ", "cut to", "cut back")):
            return False

    stage_prefixes = (
        "scene:",
        "time lapse",
        "time-lapse",
        "cut to",
        "cut back",
        "cut ",
        "flashback",
        "flashback scene",
        "fade to",
        "fade in",
        "fade out",
        "opening",
        "closing",
        "commercial",
        "tag scene",
        "end credits",
        "later",
        "next is",
        "we get back",
        "back at",
        "back in",
    )
    if lower.startswith(stage_prefixes):
        return True

    # A standalone square-bracket direction is probably scene context.
    return not before_text.strip() and len(normalized.split()) >= 4


def split_description_fragments(line: str) -> list[tuple[str, str, str]]:
    """Return ordered tokens of ('text'|'description', value, raw)."""
    tokens: list[tuple[str, str, str]] = []
    pos = 0
    for match in BRACKET_DESCRIPTION_RE.finditer(line):
        before = line[pos : match.start()]
        after = line[match.end() :]
        content = match.group(1).strip()
        if (not before.strip() and not after.strip()) or is_stage_context(content, line[: match.start()]):
            if before:
                tokens.append(("text", before, before))
            tokens.append(("description", content, match.group(0)))
            pos = match.end()
    if pos < len(line):
        tokens.append(("text", line[pos:], line[pos:]))
    return tokens or [("text", line, line)]


def normalize_speaker(speaker: str) -> str:
    return " ".join(speaker.strip().split())


def looks_like_speaker(speaker: str, known_speakers: set[str]) -> bool:
    speaker = normalize_speaker(speaker)
    if speaker in REJECT_SPEAKERS:
        return False
    if speaker in known_speakers:
        return True
    if len(speaker) > 55 or len(speaker.split()) > 8:
        return False
    if not re.match(r"^[A-Z0-9][A-Za-z0-9 .,'’\"&/#-]+$", speaker):
        return False
    lower = speaker.lower()
    if lower.startswith(("http", "scene", "chapter", "question")):
        return False
    return True


def collect_known_speakers(lines: list[str]) -> set[str]:
    speakers: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if is_end_marker(stripped):
            break
        if is_metadata(stripped) or stripped.startswith(("*[", "*(")):
            continue
        match = SPEAKER_AT_START_RE.match(stripped)
        if not match:
            match = SPEAKER_WITH_INLINE_DESCRIPTION_RE.match(stripped)
        if not match:
            continue
        speaker = normalize_speaker(match.group(1))
        if looks_like_speaker(speaker, set()):
            speakers.add(speaker)
    return speakers


def accepted_boundary(text: str, start: int) -> bool:
    if start == 0:
        return True
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    return prefix[-1] in ".?!*)]\"'…"


def find_speaker_boundaries(text: str, known_speakers: set[str]) -> list[re.Match[str]]:
    matches = []
    for match in SPEAKER_BOUNDARY_RE.finditer(text):
        speaker = normalize_speaker(match.group(1))
        if not looks_like_speaker(speaker, known_speakers):
            continue
        if match.start() != 0:
            if speaker not in known_speakers:
                continue
            if not accepted_boundary(text, match.start()):
                continue
        matches.append(match)
    return matches


def extract_inline_descriptions(text: str) -> tuple[str, list[str]]:
    descriptions: list[str] = []

    def repl(match: re.Match[str]) -> str:
        content = (match.group(1) or match.group(2) or "").strip()
        if content:
            descriptions.append(content)
        return " "

    cleaned = INLINE_DESCRIPTION_RE.sub(repl, text)
    cleaned = " ".join(cleaned.split())
    return cleaned, descriptions


def parse_dialogue_text(
    text: str,
    path: Path,
    line_no: int,
    known_speakers: set[str],
    warnings: list[WarningItem],
) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    plain_scene = PLAIN_SCENE_RE.match(stripped)
    if plain_scene:
        return [
            {
                "type": "description",
                "description": plain_scene.group(1).strip(),
                "raw": stripped,
            }
        ]

    malformed_scene = MALFORMED_SCENE_RE.match(stripped)
    if malformed_scene:
        return [
            {
                "type": "description",
                "description": malformed_scene.group(1).strip(),
                "raw": stripped,
            }
        ]

    if stripped in CREDIT_STAGE_LINES:
        return [{"type": "description", "description": stripped, "raw": stripped}]

    if re.fullmatch(r"\*\(.*\)\*\.?", stripped):
        description_raw = stripped
        if description_raw.endswith(".") and description_raw.count("*(") == description_raw.count(")*"):
            description_raw = description_raw[:-1]
        return [
            {
                "type": "description",
                "description": description_raw[2:-2].strip(),
                "raw": stripped,
            }
        ]

    inline_speaker = SPEAKER_WITH_INLINE_DESCRIPTION_RE.match(stripped)
    if inline_speaker:
        speaker = normalize_speaker(inline_speaker.group(1))
        inline_description = inline_speaker.group(2).strip()
        utterance_raw = inline_speaker.group(3).strip()
        if not looks_like_speaker(speaker, known_speakers):
            return []
        if not utterance_raw:
            warn(
                warnings,
                path,
                line_no,
                "empty_utterance",
                "Speaker boundary had no utterance text.",
                stripped,
            )
            return []
        utterance, inline_descriptions = extract_inline_descriptions(
            f"*({inline_description})* {utterance_raw}"
        )
        return [
            {
                "type": "utterance",
                "speaker": speaker,
                "utterance": utterance,
                "inline_descriptions": inline_descriptions,
                "raw": stripped,
            }
        ]

    boundaries = find_speaker_boundaries(stripped, known_speakers)
    if not boundaries:
        if stripped and not is_metadata(stripped):
            warn(
                warnings,
                path,
                line_no,
                "unparsed_text",
                "Non-empty text was not recognized as scene, action, or dialogue.",
                stripped,
            )
        return []

    events: list[dict[str, Any]] = []
    if boundaries[0].start() > 0 and stripped[: boundaries[0].start()].strip():
        warn(
            warnings,
            path,
            line_no,
            "leading_text_before_speaker",
            "Text before the first recognized speaker boundary was ignored.",
            stripped[: boundaries[0].start()].strip(),
        )

    for index, match in enumerate(boundaries):
        next_start = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(stripped)
        speaker = normalize_speaker(match.group(1))
        utterance_raw = stripped[match.end() : next_start].strip()
        if not utterance_raw:
            warn(
                warnings,
                path,
                line_no,
                "empty_utterance",
                "Speaker boundary had no utterance text.",
                stripped[match.start() : next_start],
            )
            continue
        utterance, inline_descriptions = extract_inline_descriptions(utterance_raw)
        events.append(
            {
                "type": "utterance",
                "speaker": speaker,
                "utterance": utterance,
                "inline_descriptions": inline_descriptions,
                "raw": f"{speaker}: {utterance_raw}",
            }
        )

    if len(events) > 1:
        warn(
            warnings,
            path,
            line_no,
            "multiple_dialogues_one_line",
            "Multiple speaker turns were parsed from one physical line.",
            stripped,
        )
    return events


def parse_episode(
    path: Path,
    global_counter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[WarningItem], int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    known_speakers = collect_known_speakers(lines)
    meta = episode_meta(path)
    metadata: dict[str, Any] = {"raw_header": []}
    warnings: list[WarningItem] = []
    records: list[dict[str, Any]] = []

    utterance_id = 0
    started = False
    ended = False

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ended:
            continue
        if is_end_marker(line) or is_footer_marker(line):
            ended = True
            continue

        if not started and is_metadata(line):
            metadata["raw_header"].append(line)
            continue

        tokens = split_description_fragments(line)
        for token_type, value, raw_value in tokens:
            value = value.strip()
            if not value:
                continue

            if token_type == "description":
                started = True
                records.append(
                    {
                        "season": meta["season"],
                        "episode": meta["episode"],
                        "type": "description",
                        "content": {"description": value},
                        "raw": raw_value.strip(),
                        "emotion": None,
                        "reason": None,
                    }
                )
                continue

            parsed_events = parse_dialogue_text(value, path, line_no, known_speakers, warnings)
            for event in parsed_events:
                if not started:
                    if is_metadata(value):
                        metadata["raw_header"].append(value)
                        continue
                    started = True

                if event["type"] == "description":
                    records.append(
                        {
                            "season": meta["season"],
                            "episode": meta["episode"],
                            "type": "description",
                            "content": {"description": event["description"]},
                            "raw": event["raw"],
                            "emotion": None,
                            "reason": None,
                        }
                    )
                    continue

                if event["type"] != "utterance":
                    continue

                utterance_id += 1
                global_counter += 1
                content = {
                    "utterance_id": utterance_id,
                    "global_utterance_id": global_counter,
                    "speaker": event["speaker"],
                    "utterance": event["utterance"],
                }
                if event["inline_descriptions"]:
                    content["inline_description"] = event["inline_descriptions"]
                record = {
                    "season": meta["season"],
                    "episode": meta["episode"],
                    "type": "utterance",
                    "content": content,
                    "raw": event["raw"],
                    "emotion": None,
                    "reason": None,
                }
                records.append(record)

    if not ended:
        warn(warnings, path, len(lines), "missing_end", "No standalone End marker was found.", "")
    if utterance_id == 0:
        warn(warnings, path, 0, "no_utterances", "No dialogue utterances were parsed.", "")

    episode = {
        **meta,
        "metadata": metadata,
        "known_speakers": sorted(known_speakers),
        "record_count": len(records),
        "utterance_count": utterance_id,
        "description_count": sum(1 for record in records if record["type"] == "description"),
        "records": records,
    }
    return episode, records, warnings, global_counter


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_global_counter(path: Path) -> int:
    if not path.exists():
        return 0
    counter = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "utterance":
                continue
            content = row.get("content")
            if not isinstance(content, dict):
                continue
            value = content.get("global_utterance_id")
            if isinstance(value, int):
                counter = max(counter, value)
    return counter


def merge_episode_payload(
    path: Path,
    source_dir: str,
    episodes: list[dict[str, Any]],
    records: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    append: bool,
) -> dict[str, Any]:
    if append and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    merged_episodes = list(payload.get("episodes", [])) if append else []
    merged_episodes.extend(episodes)

    record_count = int(payload.get("record_count", 0)) if append else 0
    utterance_count = int(payload.get("utterance_count", 0)) if append else 0
    description_count = int(payload.get("description_count", 0)) if append else 0
    warning_count = int(payload.get("warning_count", 0)) if append else 0

    return {
        "source_dir": source_dir,
        "episode_count": len(merged_episodes),
        "record_count": record_count + len(records),
        "utterance_count": utterance_count
        + sum(1 for record in records if record["type"] == "utterance"),
        "description_count": description_count
        + sum(1 for record in records if record["type"] == "description"),
        "warning_count": warning_count + len(warnings),
        "episodes": merged_episodes,
    }


def main() -> None:
    args = build_parser().parse_args()
    paths = select_episode_files(
        episode_files(args.input_dir, args.include_front_matter),
        args.start_index,
        args.end_index,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / args.jsonl_name
    json_path = args.output_dir / args.json_name
    warnings_path = args.output_dir / args.warnings_name

    episodes: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    append = not args.overwrite
    global_counter = existing_global_counter(jsonl_path) if append else 0

    for path in paths:
        episode, records, warnings, global_counter = parse_episode(path, global_counter)
        episodes.append(episode)
        all_records.extend(records)
        all_warnings.extend(warning.to_dict() for warning in warnings)

    write_jsonl(jsonl_path, all_records, append)
    write_jsonl(warnings_path, all_warnings, append)
    json_path.write_text(
        json.dumps(
            merge_episode_payload(
                json_path,
                args.input_dir.as_posix(),
                episodes,
                all_records,
                all_warnings,
                append,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"mode: {'append' if append else 'overwrite'}")
    print(f"files: {paths[0].name} .. {paths[-1].name}")
    print(f"episodes_this_run: {len(episodes)}")
    print(f"records_this_run: {len(all_records)}")
    print(f"utterances_this_run: {sum(1 for record in all_records if record['type'] == 'utterance')}")
    print(f"descriptions_this_run: {sum(1 for record in all_records if record['type'] == 'description')}")
    print(f"warnings_this_run: {len(all_warnings)}")
    print(f"jsonl: {jsonl_path}")
    print(f"json: {json_path}")
    print(f"warnings_jsonl: {warnings_path}")


if __name__ == "__main__":
    main()
