#!/usr/bin/env python3
"""Add long-context candidate selection flags to Friends records JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RECORDS = Path("screenplays/friends_records_renamed.jsonl")
DEFAULT_CANDIDATES = Path("workzone/long_context_candidates_merged.json")
DEFAULT_OUTPUT = Path("screenplays/friends_records_renamed_with_selected.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mark selected long-context candidate utterances in a records JSONL file."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--flag-field",
        default="long_context_selected",
        help="Top-level boolean field to write on every record.",
    )
    parser.add_argument(
        "--reason-field",
        default="",
        help="Optional top-level field for selected candidate reason. Empty disables it.",
    )
    return parser


def selected_candidate_reasons(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")

    selected: dict[int, str] = {}
    for batch in data.get("batches", []):
        if not isinstance(batch, dict) or batch.get("status") != "ok":
            continue
        parsed = batch.get("parsed")
        if not isinstance(parsed, dict):
            continue
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            if not isinstance(item, dict):
                continue
            global_id = item.get("global_utterance_id")
            if not isinstance(global_id, int):
                continue
            reason = item.get("reason")
            selected[global_id] = reason if isinstance(reason, str) else ""
    return selected


def record_global_id(record: dict[str, Any]) -> int | None:
    content = record.get("content")
    if not isinstance(content, dict):
        return None
    global_id = content.get("global_utterance_id")
    return global_id if isinstance(global_id, int) else None


def main() -> None:
    args = build_parser().parse_args()
    selected = selected_candidate_reasons(args.candidates)
    seen_selected: set[int] = set()
    total_records = 0
    utterance_records = 0
    description_records = 0
    marked_records = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("r", encoding="utf-8") as input_handle, args.output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            if not line.strip():
                output_handle.write(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {args.records}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"Invalid record at {args.records}:{line_no}: expected object")

            total_records += 1
            record_type = record.get("type")
            if record_type == "utterance":
                utterance_records += 1
            elif record_type == "description":
                description_records += 1

            global_id = record_global_id(record)
            is_selected = global_id in selected if global_id is not None else False
            record[args.flag_field] = is_selected
            if is_selected:
                marked_records += 1
                seen_selected.add(global_id)  # type: ignore[arg-type]
                if args.reason_field:
                    record[args.reason_field] = selected[global_id]  # type: ignore[index]
            elif args.reason_field:
                record[args.reason_field] = None

            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    missing_ids = sorted(set(selected) - seen_selected)
    report = {
        "records": str(args.records),
        "candidates": str(args.candidates),
        "output": str(args.output),
        "flag_field": args.flag_field,
        "reason_field": args.reason_field or None,
        "candidate_selected_ids": len(selected),
        "total_records": total_records,
        "utterance_records": utterance_records,
        "description_records": description_records,
        "marked_records": marked_records,
        "missing_selected_ids": missing_ids,
        "missing_selected_id_count": len(missing_ids),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
