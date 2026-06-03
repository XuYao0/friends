#!/usr/bin/env python3
"""用途：合并 MELD 的 train/dev/test 三个 split。

这个脚本读取 MELD 官方的 train、dev、test CSV，给每行补充来源 split 和原始行号，
然后按 Season、Episode、StartTime、EndTime、Dialogue_ID、Utterance_ID 排序，
输出一个统一的 `all_sent_emo_sorted.csv`。这个合并文件曾用于后续把 MELD 标签
对齐到 Friends 清洗台词。

Merge MELD train/dev/test CSV files into one episode-time ordered CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    Path("screenplays/MELD/train_sent_emo_sorted.csv"),
    Path("screenplays/MELD/dev_sent_emo_sorted.csv"),
    Path("screenplays/MELD/test_sent_emo_sorted.csv"),
]
DEFAULT_OUTPUT = Path("screenplays/MELD/all_sent_emo_sorted.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge MELD train/dev/test files and sort by Season, Episode, StartTime."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    return parser


def split_name(path: Path) -> str:
    name = path.name
    if name.startswith("train_"):
        return "train"
    if name.startswith("dev_"):
        return "dev"
    if name.startswith("test_"):
        return "test"
    return path.stem


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_time(value: str) -> tuple[int, int, int, int]:
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


def read_rows(paths: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    base_fieldnames: list[str] | None = None

    for path in paths:
        source_split = split_name(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"No CSV header found in {path}")
            if base_fieldnames is None:
                base_fieldnames = list(reader.fieldnames)
            elif base_fieldnames != list(reader.fieldnames):
                raise SystemExit(f"CSV header differs in {path}")

            for row_index, row in enumerate(reader, start=1):
                row["MELD_Split"] = source_split
                row["MELD_Row_ID"] = str(row_index)
                rows.append(row)

    if base_fieldnames is None:
        raise SystemExit("No input files were provided")
    return rows, ["MELD_Split", "MELD_Row_ID", *base_fieldnames]


def validate_paths(inputs: list[Path], output: Path) -> None:
    output_path = output.resolve()
    input_paths = {path.resolve() for path in inputs}
    if output_path in input_paths:
        raise SystemExit(f"Refusing to overwrite input file: {output}")


def sort_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        parse_int(row.get("Season", "")),
        parse_int(row.get("Episode", "")),
        parse_time(row.get("StartTime", "")),
        parse_time(row.get("EndTime", "")),
        parse_int(row.get("Dialogue_ID", "")),
        parse_int(row.get("Utterance_ID", "")),
        row.get("MELD_Split", ""),
        parse_int(row.get("MELD_Row_ID", "")),
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_paths(args.inputs, args.output)
    rows, fieldnames = read_rows(args.inputs)
    rows.sort(key=sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
