#!/usr/bin/env python3
"""Replace Friends main-character names in cleaned episode JSON.

The script is intentionally non-destructive: it reads the source JSON and
writes a new JSON file unless --in-place is explicitly provided.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("screenplays/converted_chunks/cleaned/friends_episodes.json")
DEFAULT_OUTPUT = Path("screenplays/converted_chunks/cleaned/friends_episodes_renamed.json")


@dataclass(frozen=True)
class NameReplacement:
    source: str
    replacement: str
    replace_lowercase: bool = False


DEFAULT_REPLACEMENTS: tuple[NameReplacement, ...] = (
    # Rachel Green
    NameReplacement("Rachel Green", "Natalie Brooks"),
    NameReplacement("Rachel", "Natalie"),
    NameReplacement("Rach", "Nat"),
    NameReplacement("Green", "Brooks"),
    # Monica Geller
    NameReplacement("Monica Geller", "Olivia Bennett"),
    NameReplacement("Monica", "Olivia"),
    NameReplacement("Mon", "Liv"),
    NameReplacement("MNCA", "Olivia"),
    NameReplacement("Geller", "Bennett"),
    # Phoebe Buffay
    NameReplacement("Phoebe Buffay", "Zoe Parker"),
    NameReplacement("Phoebe", "Zoe"),
    NameReplacement("Pheebs", "Zo"),
    NameReplacement("PHOE", "Zoe"),
    NameReplacement("Buffay", "Parker"),
    # Chandler Bing
    NameReplacement("Chandler Bing", "Ethan Hayes"),
    NameReplacement("Chandler", "Ethan"),
    NameReplacement("Chan", "Eth"),
    NameReplacement("Bing", "Hayes"),
    # Joey Tribbiani
    NameReplacement("Joey Tribbiani", "Leo Romano"),
    NameReplacement("Joey", "Leo"),
    NameReplacement("Joe", "Leo"),
    NameReplacement("Tribbiani", "Romano"),
    # Ross Geller. Monica and Ross intentionally share the new surname.
    NameReplacement("Ross Geller", "Noah Bennett"),
    NameReplacement("Ross", "Noah"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Friends main-character names throughout cleaned episode JSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mapping-json",
        type=Path,
        help=(
            "Optional mapping JSON. Format: either {\"Rachel\":\"Alice\"} or "
            "[{\"source\":\"Rachel\",\"replacement\":\"Alice\",\"replace_lowercase\":false}]."
        ),
    )
    parser.add_argument(
        "--replace-lowercase",
        action="store_true",
        help="Also replace all-lowercase matches such as 'green', 'mon', or 'joe'.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print detected default-name counts; do not write output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print replacement counts without writing output.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input. A .bak file is written first.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for a JSON report containing replacement counts.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def load_replacements(path: Path | None) -> list[NameReplacement]:
    if path is None:
        return list(DEFAULT_REPLACEMENTS)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    replacements: list[NameReplacement] = []
    if isinstance(raw, dict):
        for source, replacement in raw.items():
            replacements.append(NameReplacement(str(source), str(replacement)))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or "source" not in item or "replacement" not in item:
                raise ValueError("Each mapping list item must contain source and replacement.")
            replacements.append(
                NameReplacement(
                    str(item["source"]),
                    str(item["replacement"]),
                    bool(item.get("replace_lowercase", False)),
                )
            )
    else:
        raise ValueError("--mapping-json must contain either an object or a list.")

    if not replacements:
        raise ValueError("No replacements loaded.")
    return replacements


def has_uppercase(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def apply_case(source_text: str, replacement: str) -> str:
    if source_text.isupper():
        return replacement.upper()
    if source_text.islower():
        return replacement.lower()
    if source_text.istitle():
        return replacement.title()
    if source_text[:1].isupper():
        return replacement
    return replacement


def compile_pattern(replacements: list[NameReplacement]) -> re.Pattern[str]:
    sources = sorted({r.source for r in replacements}, key=len, reverse=True)
    escaped = [re.escape(source) for source in sources]
    # Avoid replacing inside words: "Mon" should not match "Monday" or "Monica".
    return re.compile(r"(?<![A-Za-z])(" + "|".join(escaped) + r")(?![A-Za-z])", re.IGNORECASE)


def replace_text(
    text: str,
    pattern: re.Pattern[str],
    replacements_by_lower: dict[str, NameReplacement],
    counts: Counter[str],
    replace_lowercase: bool,
) -> str:
    def repl(match: re.Match[str]) -> str:
        matched = match.group(0)
        rule = replacements_by_lower[matched.lower()]
        if not replace_lowercase and not rule.replace_lowercase and not has_uppercase(matched):
            return matched
        counts[rule.source] += 1
        return apply_case(matched, rule.replacement)

    return pattern.sub(repl, text)


def replace_recursive(
    value: Any,
    pattern: re.Pattern[str],
    replacements_by_lower: dict[str, NameReplacement],
    counts: Counter[str],
    replace_lowercase: bool,
) -> Any:
    if isinstance(value, str):
        return replace_text(value, pattern, replacements_by_lower, counts, replace_lowercase)
    if isinstance(value, list):
        return [
            replace_recursive(item, pattern, replacements_by_lower, counts, replace_lowercase)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: replace_recursive(val, pattern, replacements_by_lower, counts, replace_lowercase)
            for key, val in value.items()
        }
    return value


def collect_string_counts(
    value: Any,
    pattern: re.Pattern[str],
    replacements_by_lower: dict[str, NameReplacement],
    replace_lowercase: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()

    def walk(item: Any) -> None:
        if isinstance(item, str):
            for match in pattern.finditer(item):
                matched = match.group(0)
                rule = replacements_by_lower[matched.lower()]
                if replace_lowercase or rule.replace_lowercase or has_uppercase(matched):
                    counts[rule.source] += 1
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)

    walk(value)
    return counts


def print_counts(title: str, counts: Counter[str], replacements: list[NameReplacement]) -> None:
    print(title)
    for rule in replacements:
        count = counts.get(rule.source, 0)
        if count:
            print(f"{count:7d}  {rule.source} -> {rule.replacement}")


def write_json(path: Path, data: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def main() -> None:
    args = parse_args()
    replacements = load_replacements(args.mapping_json)
    replacements_by_lower = {rule.source.lower(): rule for rule in replacements}
    if len(replacements_by_lower) != len(replacements):
        raise ValueError("Duplicate replacement sources after case-folding.")

    pattern = compile_pattern(replacements)
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    before_counts = collect_string_counts(
        data, pattern, replacements_by_lower, args.replace_lowercase
    )
    print_counts("Detected name occurrences:", before_counts, replacements)

    if args.stats_only:
        return

    output_path = args.input if args.in_place else args.output
    if args.in_place:
        backup_path = args.input.with_suffix(args.input.suffix + ".bak")
        write_json(backup_path, data, args.indent)
        print(f"Backup written: {backup_path}")

    changed_counts: Counter[str] = Counter()
    replaced = replace_recursive(
        copy.deepcopy(data),
        pattern,
        replacements_by_lower,
        changed_counts,
        args.replace_lowercase,
    )
    print_counts("Applied replacements:", changed_counts, replacements)

    report = {
        "input": str(args.input),
        "output": str(output_path),
        "mapping_json": str(args.mapping_json) if args.mapping_json else None,
        "replace_lowercase": args.replace_lowercase,
        "counts": dict(changed_counts),
        "total_replacements": sum(changed_counts.values()),
    }
    if args.report:
        write_json(args.report, report, args.indent)
        print(f"Report written: {args.report}")

    if args.dry_run:
        print("Dry run: output JSON was not written.")
        return

    write_json(output_path, replaced, args.indent)
    print(f"Output written: {output_path}")
    print(f"Total replacements: {sum(changed_counts.values())}")


if __name__ == "__main__":
    main()
