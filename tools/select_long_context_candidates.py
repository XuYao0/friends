#!/usr/bin/env python3
"""Select long-context emotion-dependency candidates from Friends transcripts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_RECORDS = Path("screenplays/friends_records_with_meld.jsonl")
DEFAULT_DESCRIPTIONS = Path("screenplays/converted_chunks/episode_descriptions.jsonl")
DEFAULT_OUTPUT = Path("workzone/long_context_candidates.json")

DEPENDENCY_TYPES = {
    "relationship_history",
    "character_trait_or_habit",
    "past_event_reference",
    "surface_true_emotion_mismatch",
}
RISK_TYPES = {
    "current_scene_only",
    "overinterpretation",
    "pretraining_or_fame_bias",
    "none",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use DeepSeek to find utterances whose emotion labels may require long-range context."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="max", choices=["high", "max"])
    parser.add_argument("--start-global-utterance-id", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on a failed batch. Default records the error and continues.",
    )
    parser.add_argument("--no-timestamp-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable DeepSeek thinking mode. Default is enabled.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def row_utterance(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("type") != "utterance":
        return None
    content = row.get("content")
    if not isinstance(content, dict):
        return None
    return content


def utterance_global_id(row: dict[str, Any]) -> int | None:
    utterance = row_utterance(row)
    if utterance is None:
        return None
    global_id = utterance.get("global_utterance_id")
    return global_id if isinstance(global_id, int) else None


def load_episode_descriptions(path: Path) -> dict[tuple[int, int], str]:
    descriptions: dict[tuple[int, int], str] = {}
    for row in read_jsonl(path):
        season = row.get("season")
        episode = row.get("episode")
        description = row.get("description")
        if isinstance(season, int) and isinstance(episode, int) and isinstance(description, str):
            descriptions[(season, episode)] = description.strip()
    return descriptions


def batch_episode_keys(rows: list[dict[str, Any]], batch: dict[str, Any]) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows[batch["row_start"] : batch["row_end"] + 1]:
        season = row.get("season")
        episode = row.get("episode")
        if not isinstance(season, int) or not isinstance(episode, int):
            continue
        key = (season, episode)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def format_episode_synopses(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    descriptions: dict[tuple[int, int], str],
) -> str:
    lines: list[str] = []
    for season, episode in batch_episode_keys(rows, batch):
        description = descriptions.get((season, episode), "")
        if description:
            lines.append(f"S{season:02d}E{episode:02d}: {description}")
        else:
            lines.append(f"S{season:02d}E{episode:02d}: [missing synopsis]")
    return "\n".join(lines)


def format_row(row: dict[str, Any], *, target: bool) -> str:
    marker = "TARGET" if target else "context"
    season = row.get("season")
    episode = row.get("episode")
    prefix = f"[{marker} S{season:02d}E{episode:02d}]"
    content = row.get("content")
    if row.get("type") == "description" and isinstance(content, dict):
        return f'{prefix} description: {content.get("description", "")}'

    utterance = row_utterance(row)
    if utterance is None:
        return f"{prefix} {row.get('raw', '')}"

    line = (
        f'{prefix} gid={utterance.get("global_utterance_id")} '
        f'{utterance.get("speaker")}: '
        f'{utterance.get("utterance")}'
    )
    inline = utterance.get("inline_description")
    if isinstance(inline, list) and inline:
        line += " [" + "; ".join(str(item) for item in inline) + "]"
    return line


def make_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    overlap: int,
    start_global_utterance_id: int | None,
) -> list[dict[str, Any]]:
    utterance_row_indices = [
        idx for idx, row in enumerate(rows) if row_utterance(row) is not None and utterance_global_id(row) is not None
    ]
    if not utterance_row_indices:
        raise SystemExit("No utterances with global_utterance_id found.")

    start_pos = 0
    if start_global_utterance_id is not None:
        for pos, row_idx in enumerate(utterance_row_indices):
            global_id = utterance_global_id(rows[row_idx])
            if global_id is not None and global_id >= start_global_utterance_id:
                start_pos = pos
                break
        else:
            raise SystemExit(f"No utterance found at or after global id {start_global_utterance_id}")

    batches: list[dict[str, Any]] = []
    batch_no = 1
    pos = start_pos
    while pos < len(utterance_row_indices):
        target_start_pos = pos
        target_end_pos = min(pos + batch_size, len(utterance_row_indices))
        context_start_pos = max(0, target_start_pos - overlap)

        first_context_row = utterance_row_indices[context_start_pos]
        last_target_row = utterance_row_indices[target_end_pos - 1]
        target_row_indices = set(utterance_row_indices[target_start_pos:target_end_pos])
        target_ids = [
            utterance_global_id(rows[row_idx])
            for row_idx in utterance_row_indices[target_start_pos:target_end_pos]
        ]
        batches.append(
            {
                "batch_id": batch_no,
                "row_start": first_context_row,
                "row_end": last_target_row,
                "target_row_indices": target_row_indices,
                "target_global_ids": [global_id for global_id in target_ids if global_id is not None],
            }
        )
        batch_no += 1
        pos = target_end_pos
    return batches


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are a careful data curation assistant for a long-dialogue emotion-recognition benchmark built from the sitcom Friends (老友记) screenplay transcripts.",
            "Return only valid JSON. Do not include Markdown, comments, or extra text.",
            "Your goal is high-recall candidate discovery: find target utterances whose true speaker emotion may require long-range dialogue history.",
            "Treat utterance lines as character dialogue and description lines as screenplay stage directions, including scene changes, facial expressions, actions, entrances, and exits.",
            "Examine every TARGET utterance one by one before deciding which ones to select.",
            "Output one review item for every TARGET utterance, with a selection label and a concise evidence-based reason.",
        ]
    )


def build_user_prompt(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    descriptions: dict[tuple[int, int], str],
    *,
    max_candidates: int,
) -> str:
    target_row_indices = batch["target_row_indices"]
    transcript = "\n".join(
        format_row(row, target=idx in target_row_indices)
        for idx, row in enumerate(rows[batch["row_start"] : batch["row_end"] + 1], start=batch["row_start"])
    )
    target_ids = batch["target_global_ids"]
    synopses = format_episode_synopses(rows, batch, descriptions)
    return "\n".join(
        [
            "Task:",
            "The input is from Friends (老友记) screenplay transcripts. Utterance lines are character dialogue; description lines are stage directions that may include setting, actions, expressions, entrances, and exits.",
            "For every TARGET utterance, decide whether it is a strong candidate for requiring long-range historical context to identify the speaker's true emotion correctly.",
            "Context lines are included only to stabilize local reading. Do not select context-only utterances.",
            "Inspect the TARGET utterances one by one in order. Do not only pick the most dramatic or most famous lines.",
            "",
            "Definition of long-range dependency:",
            "A target utterance is a candidate only if its emotional interpretation likely depends on information beyond the immediate line or very short local exchange, such as earlier scenes, earlier episodes, relationship history, recurring character patterns, or unresolved past events.",
            "",
            "Four useful long-range dependency types:",
            "1. relationship_history: emotion depends on relationship history, past romance/conflict/family dynamics, breakups, betrayals, promises, jealousy, or role expectations.",
            "2. character_trait_or_habit: emotion depends on stable character traits or recurring habits, such as sarcasm, defensiveness, control needs, avoidance, insecurity, or unusual expression style.",
            "3. past_event_reference: the line weakly refers to older events through terms such as again, that thing, after what happened, you know why, or another indirect allusion.",
            "4. surface_true_emotion_mismatch: surface wording differs from the speaker's likely true emotion because of sarcasm, pretending, suppression, face-saving, awkwardness, or joking to hide distress.",
            "",
            "Common AI mistakes to avoid:",
            "1. Do not confuse current-scene context with long-range dependency. If the emotion is clear from the current scene or nearby lines, do not select it.",
            "2. Do not overinterpret ordinary jokes, factual replies, or functional transitions as deep emotion. If historical evidence would be decorative rather than necessary, skip it.",
            "",
            "Selection policy:",
            f"- Mark at most {max_candidates} TARGET utterances as is_selected=true.",
            "- Prefer recall over precision, but every selected item must have a concrete short-context failure mode.",
            "- Do not use future lines after the target utterance as evidence.",
            "- Return one review item for every TARGET utterance, even when is_selected=false.",
            "- For each review item, write reason in no more than 5 sentences.",
            "- If is_selected=true, the reason must explain both the likely long-range dependency and why short context may fail.",
            "- If is_selected=false, the reason should briefly say why local context is enough or why the line is not a good long-context candidate.",
            "",
            "Return this JSON object shape:",
            json.dumps(
                {
                    "batch_id": batch["batch_id"],
                    "utterance_reviews": [
                        {
                            "global_utterance_id": 123,
                            "season": 1,
                            "episode": 1,
                            "speaker": "Rachel",
                            "is_selected": True,
                            "reason": "no more than 5 sentences explaining the decision for this utterance"
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            "",
            f"Allowed target global_utterance_id range: {target_ids[0]}..{target_ids[-1]}",
            "",
            "Episode synopsis for every episode covered by this batch context:",
            synopses,
            "",
            "Transcript:",
            transcript,
        ]
    )


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {
            "prompt_tokens": None,
            "prompt_cache_hit_tokens": None,
            "prompt_cache_miss_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
        }
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {name: getattr(usage, name, None) for name in dir(usage) if not name.startswith("_")}

    details = raw.get("completion_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    return {
        "prompt_tokens": raw.get("prompt_tokens"),
        "prompt_cache_hit_tokens": raw.get("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": raw.get("prompt_cache_miss_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
        "total_tokens": raw.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
    }


def object_to_plain_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: object_to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [object_to_plain_dict(item) for item in value]
    return value


def parse_candidates(content: str, batch: dict[str, Any]) -> dict[str, Any]:
    if not content.strip():
        raise ValueError("Model returned empty content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model returned non-object JSON")

    parsed["batch_id"] = batch["batch_id"]
    reviews = parsed.get("utterance_reviews")
    if not isinstance(reviews, list):
        reviews = parsed.get("candidates")
    if not isinstance(reviews, list):
        raise ValueError("Missing utterance_reviews list")

    allowed_ids = set(batch["target_global_ids"])
    clean_reviews: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    for item in reviews:
        if not isinstance(item, dict):
            continue
        validation_errors: list[str] = []
        global_id = item.get("global_utterance_id")
        if global_id not in allowed_ids:
            item["validation_errors"] = ["global_utterance_id is outside this batch's target range"]
            continue
        if not isinstance(item.get("is_selected"), bool):
            validation_errors.append("missing or invalid is_selected; defaulted to false")
            item["is_selected"] = False
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            validation_errors.append("missing or empty reason")
        if validation_errors:
            item["validation_errors"] = validation_errors
        clean_reviews.append(item)
        if item.get("is_selected") is True:
            selected_candidates.append(item)
    parsed["utterance_reviews"] = clean_reviews
    parsed["candidates"] = selected_candidates
    return parsed


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def request_raw_batch(client: Any, request_kwargs: dict[str, Any]) -> Any:
    return client.chat.completions.create(**request_kwargs)


def timestamped_output_path(path: Path, timestamp: str) -> Path:
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_{timestamp}{suffix}")


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_output(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metadata": {
            "model": args.model,
            "base_url": args.base_url,
            "records": str(args.records),
            "descriptions": str(args.descriptions),
            "output": str(args.output),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "batch_size": args.batch_size,
            "overlap": args.overlap,
            "max_candidates": args.max_candidates,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "thinking": "disabled" if args.disable_thinking else "enabled",
            "start_global_utterance_id": args.start_global_utterance_id,
            "strategy": "Batch target utterances with preceding overlap context; select only target lines likely requiring long-range history for true-emotion recognition.",
            "dependency_types": sorted(DEPENDENCY_TYPES),
            "known_ai_error_types": [
                "confusing current-scene context with long-range dependency",
                "overinterpreting ordinary or functional lines",
                "relying on pretraining/fame bias instead of evidence need",
            ],
        },
        "batches": [],
    }


def load_or_create_output(args: argparse.Namespace) -> tuple[dict[str, Any], set[int]]:
    if not args.resume or not args.output.exists():
        return build_output(args), set()
    try:
        output = json.loads(args.output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cannot resume from invalid JSON output {args.output}: {exc}") from exc
    if not isinstance(output, dict) or not isinstance(output.get("batches"), list):
        raise SystemExit(f"Cannot resume: {args.output} does not contain a batches list")
    output.setdefault("metadata", {})["resumed_at"] = datetime.now(timezone.utc).isoformat()
    completed = {
        batch.get("batch_id")
        for batch in output["batches"]
        if isinstance(batch, dict) and isinstance(batch.get("batch_id"), int)
    }
    return output, completed


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.overlap < 0:
        raise SystemExit("--overlap cannot be negative")
    if not args.no_timestamp_output and not args.resume:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = timestamped_output_path(args.output, timestamp)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: openai. This UV project already declares it; run with `uv run`.") from exc
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("Missing dependency: tqdm. This UV project already declares it; run with `uv run`.") from exc

    rows = read_jsonl(args.records)
    descriptions = load_episode_descriptions(args.descriptions)
    batches = make_batches(
        rows,
        batch_size=args.batch_size,
        overlap=args.overlap,
        start_global_utterance_id=args.start_global_utterance_id,
    )
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    output, completed = load_or_create_output(args)
    system_prompt = build_system_prompt()
    output["metadata"]["started_at"] = datetime.now(timezone.utc).isoformat()
    write_output(args.output, output)

    processed = 0
    pending_batches = [batch for batch in batches if batch["batch_id"] not in completed]
    if args.limit_batches is not None:
        pending_batches = pending_batches[: args.limit_batches]
    progress = tqdm(pending_batches, desc="Selecting long-context candidates", unit="batch")
    total_candidates = 0
    failed_batches = 0
    for batch in progress:
        batch_id = batch["batch_id"]
        user_prompt = build_user_prompt(
            rows,
            batch,
            descriptions,
            max_candidates=args.max_candidates,
        )
        request_kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if args.disable_thinking:
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": args.reasoning_effort,
            }

        request_record = {
            "model": request_kwargs["model"],
            "messages": request_kwargs["messages"],
            "response_format": request_kwargs["response_format"],
            "max_tokens": request_kwargs["max_tokens"],
            "stream": request_kwargs["stream"],
            "extra_body": request_kwargs.get("extra_body"),
        }
        output["metadata"]["last_started_batch"] = {
            "batch_id": batch_id,
            "target_global_id_start": batch["target_global_ids"][0],
            "target_global_id_end": batch["target_global_ids"][-1],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": request_record,
        }
        write_output(args.output, output)

        try:
            response = request_raw_batch(client, request_kwargs)
        except Exception as exc:
            error_result = {
                "batch_id": batch_id,
                "target_global_id_start": batch["target_global_ids"][0],
                "target_global_id_end": batch["target_global_ids"][-1],
                "candidate_count": 0,
                "error": str(exc),
                "status": "failed",
                "request": request_record,
            }
            output["batches"].append(error_result)
            completed.add(batch_id)
            processed += 1
            failed_batches += 1
            progress.set_postfix(
                candidates=total_candidates,
                failed=failed_batches,
                last=f"{error_result['target_global_id_start']}..{error_result['target_global_id_end']}",
            )
            write_output(args.output, output)
            if args.stop_on_error:
                raise
            continue

        message = response.choices[0].message
        raw_content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        finish_reason = response.choices[0].finish_reason
        usage = usage_to_dict(response.usage)
        raw_response = object_to_plain_dict(response)

        try:
            parsed = parse_candidates(raw_content, batch)
            status = "ok"
            parse_error = None
            candidates = parsed["candidates"]
        except ValueError as exc:
            parsed = None
            status = "parse_failed"
            parse_error = str(exc)
            candidates = []

        result = {
            "batch_id": batch_id,
            "target_global_id_start": batch["target_global_ids"][0],
            "target_global_id_end": batch["target_global_ids"][-1],
            "candidate_count": len(candidates),
            "status": status,
            "parse_error": parse_error,
            "request": request_record,
            "parsed": parsed,
            "raw_content": raw_content,
            "cot": reasoning_content,
            "raw_response": raw_response,
            "usage": usage,
            "finish_reason": finish_reason,
        }
        output["batches"].append(result)
        completed.add(batch_id)
        processed += 1
        if status != "ok":
            failed_batches += 1
        total_candidates += result["candidate_count"]
        progress.set_postfix(
            candidates=total_candidates,
            failed=failed_batches,
            last=f"{result['target_global_id_start']}..{result['target_global_id_end']}",
        )
        write_output(args.output, output)

    output["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    output["metadata"]["processed_batches_this_run"] = processed
    write_output(args.output, output)
    print(f"Completed {processed} batches; wrote {args.output}")


if __name__ == "__main__":
    main()
