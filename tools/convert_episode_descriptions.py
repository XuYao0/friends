#!/usr/bin/env python3
"""Convert Friends episode descriptions Markdown into JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("workzone") / "各集简介.md"
DEFAULT_OUTPUT = Path("workzone") / "episode_descriptions.jsonl"
HEADING_RE = re.compile(r"^S(?P<season>\d{2})E(?P<episode>\d{2})[:：]\s*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SxxExx episode descriptions Markdown to JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on duplicate season/episode keys.",
    )
    return parser


def normalize_description(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_descriptions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: tuple[int, int] | None = None
    buffer: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = HEADING_RE.match(line.strip())
        if match:
            if current is not None:
                rows.append(
                    {
                        "season": current[0],
                        "episode": current[1],
                        "description": normalize_description(buffer),
                    }
                )
            current = (int(match.group("season")), int(match.group("episode")))
            buffer = []
            continue

        if current is None:
            if line.strip():
                raise SystemExit(f"Unexpected text before first episode at line {line_no}: {line}")
            continue

        buffer.append(line)

    if current is not None:
        rows.append(
            {
                "season": current[0],
                "episode": current[1],
                "description": normalize_description(buffer),
            }
        )

    if not rows:
        raise SystemExit(f"No episode descriptions found in {path}")
    return rows


def corrected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply known source typo fixes without changing description text."""
    fixed = [dict(row) for row in rows]
    if len(fixed) >= 2 and fixed[0]["season"] == 1 and fixed[0]["episode"] == 1:
        second = fixed[1]
        if second["season"] == 2 and second["episode"] == 2:
            description = second["description"].lower()
            if "carol" in description and "pregnant" in description and "barry" in description:
                second["season"] = 1
                second["episode"] = 2
    return fixed


def duplicate_keys(rows: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (row["season"], row["episode"])
        counts[key] = counts.get(key, 0) + 1
    return {key: count for key, count in counts.items() if count > 1}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    rows = corrected_rows(parse_descriptions(args.input))
    duplicates = duplicate_keys(rows)
    if args.strict and duplicates:
        formatted = ", ".join(f"S{s:02d}E{e:02d} x{count}" for (s, e), count in duplicates.items())
        raise SystemExit(f"Duplicate episode keys: {formatted}")

    write_jsonl(args.output, rows)
    print(f"rows: {len(rows)}")
    print(f"output: {args.output}")
    if duplicates:
        formatted = ", ".join(f"S{s:02d}E{e:02d} x{count}" for (s, e), count in duplicates.items())
        print(f"duplicates: {formatted}")


if __name__ == "__main__":
    main()
