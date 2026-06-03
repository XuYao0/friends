#!/usr/bin/env python3
"""用途：把 MELD 台词标签对齐到本项目清洗后的 Friends 台词。

这个脚本读取 MELD CSV 和 `friends_records.jsonl`，按季/集、说话人、台词文本和
局部顺序进行匹配。它会做文本规范化、相似度匹配，并允许把连续同一说话人的
MELD 行合并后再匹配 Friends 的单条 utterance。输出是 mapped CSV，里面记录
每条 MELD 行是否匹配、匹配分数、对应的 Friends utterance_id/global_utterance_id
等信息，供 `apply_meld_to_friends_records.py` 继续写回 JSONL。

Map MELD utterance labels onto cleaned Friends utterance records.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_RECORDS = Path("screenplays/converted_chunks/cleaned/friends_records.jsonl")
DEFAULT_OUTPUT_DIR = Path("workzone/meld_mapping")
DEFAULT_MELD_FILES = [Path("screenplays/MELD/all_sent_emo_sorted.csv")]

STAGE_RE = re.compile(r"\*\([^)]*\)\*|\([^)]*\)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align MELD CSV rows to cleaned Friends utterance records by episode, "
            "speaker, utterance text, and local order."
        )
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score", type=float, default=0.86)
    parser.add_argument(
        "--max-merge",
        type=int,
        default=3,
        help="Maximum number of consecutive same-speaker MELD rows to merge for matching.",
    )
    parser.add_argument(
        "--max-a-scan",
        type=int,
        default=300,
        help="Maximum number of Friends utterances to scan before marking current MELD row unmatched.",
    )
    parser.add_argument("meld_csv", nargs="*", type=Path, default=DEFAULT_MELD_FILES)
    return parser


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u0091", "'")
        .replace("\u0092", "'")
        .replace("\u0093", '"')
        .replace("\u0094", '"')
        .replace("\u0096", "-")
        .replace("\u0097", "-")
    )
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("…", "...")
    text = STAGE_RE.sub(" ", text)
    text = text.lower()
    text = text.replace("n't", "nt")
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_speaker(speaker: str) -> str:
    return normalize_text(speaker)


def read_records(path: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_episode: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "utterance":
                continue
            content = row.get("content")
            if not isinstance(content, dict):
                continue
            season = int(row["season"])
            episode = int(row["episode"])
            utterance = str(content.get("utterance", ""))
            speaker = str(content.get("speaker", ""))
            by_episode[(season, episode)].append(
                {
                    "season": season,
                    "episode": episode,
                    "utterance_id": content.get("utterance_id"),
                    "global_utterance_id": content.get("global_utterance_id"),
                    "speaker": speaker,
                    "utterance": utterance,
                    "norm_speaker": normalize_speaker(speaker),
                    "norm_utterance": normalize_text(utterance),
                }
            )

    for utterances in by_episode.values():
        utterances.sort(key=lambda item: int(item["utterance_id"]))
    return dict(by_episode)


def time_key(value: str) -> tuple[int, int, int, int]:
    value = (value or "").strip().replace(",", ":")
    parts = value.split(":")
    try:
        if len(parts) == 4:
            hour, minute, second, millis = (int(part) for part in parts)
            return hour, minute, second, millis
        if len(parts) == 3:
            minute, second, millis = (int(part) for part in parts)
            return 0, minute, second, millis
    except ValueError:
        pass
    return 99, 99, 99, 999


def text_similarity(left: str, right: str) -> float:
    text_score = SequenceMatcher(None, left, right).ratio()
    if left and right:
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        length_ratio = shorter / longer if longer else 0.0
        if shorter >= 12 and length_ratio >= 0.25 and (
            left in right or right in left
        ):
            text_score = max(text_score, 0.94)
    if not left or not right:
        return 0.0
    return text_score


def meld_group_options(
    rows: list[dict[str, str]],
    start: int,
    max_merge: int,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if start >= len(rows):
        return options

    speaker = normalize_speaker(rows[start].get("Speaker", ""))
    utterances: list[str] = []
    for offset in range(max_merge):
        index = start + offset
        if index >= len(rows):
            break
        row_speaker = normalize_speaker(rows[index].get("Speaker", ""))
        if row_speaker != speaker:
            break
        utterances.append(rows[index].get("Utterance", ""))
        combined = " ".join(utterances)
        options.append(
            {
                "start": start,
                "size": offset + 1,
                "speaker": speaker,
                "utterance": combined,
                "norm_utterance": normalize_text(combined),
            }
        )
    return options


def best_group_match(
    record: dict[str, Any],
    group_options: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, float, float]:
    exact_matches = [
        option
        for option in group_options
        if option["speaker"] == record["norm_speaker"]
        and option["norm_utterance"] == record["norm_utterance"]
    ]
    if exact_matches:
        match = max(exact_matches, key=lambda option: option["size"])
        return match, "exact", 1.0, 1.0

    best: tuple[dict[str, Any], float, float] | None = None
    for option in group_options:
        text_score = text_similarity(option["norm_utterance"], record["norm_utterance"])
        speaker_bonus = 0.04 if option["speaker"] == record["norm_speaker"] else -0.12
        score = min(1.0, max(0.0, text_score + speaker_bonus))
        if best is None or (score, text_score, option["size"]) > (
            best[1],
            best[2],
            best[0]["size"],
        ):
            best = option, score, text_score

    if best is None:
        return None, "unmatched", 0.0, 0.0
    return best[0], "fuzzy", best[1], best[2]


def map_file(
    meld_path: Path,
    records_by_episode: dict[tuple[int, int], list[dict[str, Any]]],
    output_dir: Path,
    min_score: float,
    max_merge: int,
    max_a_scan: int,
) -> dict[str, int]:
    with meld_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows.sort(
        key=lambda row: (
            int(row["Season"]),
            int(row["Episode"]),
            time_key(row.get("StartTime", "")),
            int(row.get("Dialogue_ID") or 0),
            int(row.get("Utterance_ID") or 0),
        )
    )

    rows_by_episode: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_episode[(int(row["Season"]), int(row["Episode"]))].append(row)

    mapped_rows: list[dict[str, Any]] = []
    stats = {"total": 0, "matched": 0, "exact": 0, "fuzzy": 0, "unmatched": 0}

    def append_mapped_rows(
        meld_rows: list[dict[str, str]],
        status: str,
        score: float,
        text_score: float,
        match: dict[str, Any] | None,
        group_utterance: str = "",
        a_scan_count: int = 0,
    ) -> None:
        for row in meld_rows:
            stats["total"] += 1
            if status == "unmatched":
                stats["unmatched"] += 1
            else:
                stats["matched"] += 1
                stats[status] += 1

            mapped = dict(row)
            mapped.update(
                {
                    "match_status": status,
                    "match_score": f"{score:.4f}",
                    "match_text_score": f"{text_score:.4f}",
                    "meld_group_size": len(meld_rows),
                    "meld_group_utterance": group_utterance,
                    "a_scan_count": a_scan_count,
                    "friends_utterance_id": "" if match is None else match["utterance_id"],
                    "friends_global_utterance_id": ""
                    if match is None
                    else match["global_utterance_id"],
                    "friends_speaker": "" if match is None else match["speaker"],
                    "friends_utterance": "" if match is None else match["utterance"],
                }
            )
            mapped_rows.append(mapped)

    episode_keys = sorted(set(rows_by_episode) | set(records_by_episode))
    for episode_key in episode_keys:
        meld_episode_rows = rows_by_episode.get(episode_key, [])
        records = records_by_episode.get(episode_key, [])
        a_index = 0
        meld_index = 0

        while meld_index < len(meld_episode_rows):
            if a_index >= len(records):
                append_mapped_rows(
                    [meld_episode_rows[meld_index]],
                    "unmatched",
                    0.0,
                    0.0,
                    None,
                    a_scan_count=0,
                )
                meld_index += 1
                continue

            options = meld_group_options(meld_episode_rows, meld_index, max_merge)
            best: tuple[int, dict[str, Any], dict[str, Any], str, float, float] | None = None
            scan_limit = min(len(records), a_index + max_a_scan)
            for candidate_index in range(a_index, scan_limit):
                record = records[candidate_index]
                match_option, status, score, text_score = best_group_match(record, options)
                if match_option is None:
                    continue
                if status == "exact":
                    best = candidate_index, record, match_option, status, score, text_score
                    break
                if best is None or (score, text_score, -candidate_index) > (
                    best[4],
                    best[5],
                    -best[0],
                ):
                    best = candidate_index, record, match_option, status, score, text_score

            if best is None or best[4] < min_score:
                append_mapped_rows(
                    [meld_episode_rows[meld_index]],
                    "unmatched",
                    0.0 if best is None else best[4],
                    0.0 if best is None else best[5],
                    None,
                    group_utterance="" if best is None else str(best[2]["utterance"]),
                    a_scan_count=scan_limit - a_index,
                )
                meld_index += 1
                continue

            matched_a_index, record, match_option, status, score, text_score = best
            group_size = int(match_option["size"])
            matched_meld_rows = meld_episode_rows[meld_index : meld_index + group_size]
            append_mapped_rows(
                matched_meld_rows,
                status,
                score,
                text_score,
                record,
                group_utterance=str(match_option["utterance"]),
                a_scan_count=matched_a_index - a_index,
            )
            a_index = matched_a_index + 1
            meld_index += group_size

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{meld_path.stem}_mapped.csv"
    fieldnames = list(mapped_rows[0].keys()) if mapped_rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped_rows)

    stats["output"] = str(output_path)  # type: ignore[assignment]
    return stats


def main() -> None:
    args = build_parser().parse_args()
    records_by_episode = read_records(args.records)
    summaries = [
        map_file(
            path,
            records_by_episode,
            args.output_dir,
            args.min_score,
            args.max_merge,
            args.max_a_scan,
        )
        for path in args.meld_csv
    ]

    for path, summary in zip(args.meld_csv, summaries, strict=True):
        total = summary["total"]
        matched = summary["matched"]
        ratio = matched / total if total else 0.0
        print(
            f"{path}: matched {matched}/{total} ({ratio:.1%}), "
            f"exact={summary['exact']}, fuzzy={summary['fuzzy']}, "
            f"unmatched={summary['unmatched']} -> {summary['output']}"
        )


if __name__ == "__main__":
    main()
