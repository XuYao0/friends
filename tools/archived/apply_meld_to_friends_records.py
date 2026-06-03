#!/usr/bin/env python3
"""用途：把 MELD 对齐结果写回 Friends 清洗记录。

这个脚本读取 `map_meld_to_friends.py` 生成的 mapped CSV，根据
`friends_global_utterance_id` 找到对应的 Friends utterance，并在输出 JSONL 中
增加 `meld` 字段。它不会修改原始 `friends_records.jsonl`，而是生成一个带 MELD
标签副本，供后续对照、人工核查或标注评估使用。

Add matched MELD labels to Friends records without modifying the source JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_RECORDS = Path("screenplays/converted_chunks/cleaned/friends_records.jsonl")
DEFAULT_MAPPING = Path("workzone/meld_mapping_max10/all_sent_emo_sorted_mapped.csv")
DEFAULT_OUTPUT = Path("workzone/friends_records_with_meld.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy Friends records and add a meld field to matched utterances."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_meld_mapping(path: Path) -> OrderedDict[str, dict[str, Any]]:
    by_global_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("match_status") == "unmatched":
                continue
            global_id = (row.get("friends_global_utterance_id") or "").strip()
            if not global_id:
                continue

            label = by_global_id.get(global_id)
            if label is None:
                label = {
                    "emotion": [],
                    "sentiment": [],
                    "match_status": row.get("match_status", ""),
                    "match_score": parse_float(row.get("match_score", "")),
                    "match_text_score": parse_float(row.get("match_text_score", "")),
                    "meld_group_size": parse_int(row.get("meld_group_size", "")),
                }
                by_global_id[global_id] = label

            label["emotion"].append(row.get("Emotion", ""))
            label["sentiment"].append(row.get("Sentiment", ""))

    return by_global_id


def copy_records_with_meld(records_path: Path, output_path: Path, labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    stats = {
        "records": 0,
        "utterances": 0,
        "matched_utterances": 0,
        "description_records": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with records_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stats["records"] += 1

            if record.get("type") == "description":
                stats["description_records"] += 1

            content = record.get("content")
            if record.get("type") == "utterance" and isinstance(content, dict):
                stats["utterances"] += 1
                global_id = str(content.get("global_utterance_id", ""))
                label = labels.get(global_id)
                if label is not None:
                    record["meld"] = label
                    stats["matched_utterances"] += 1

            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    args = build_parser().parse_args()
    if args.output.resolve() == args.records.resolve():
        raise SystemExit(f"Refusing to overwrite input file: {args.records}")

    labels = read_meld_mapping(args.mapping)
    stats = copy_records_with_meld(args.records, args.output, labels)
    unused_labels = len(labels) - stats["matched_utterances"]

    print(
        f"Wrote {stats['records']} records -> {args.output}\n"
        f"utterances={stats['utterances']}, matched_utterances={stats['matched_utterances']}, "
        f"description_records={stats['description_records']}, unused_meld_matches={unused_labels}"
    )


if __name__ == "__main__":
    main()
