#!/usr/bin/env python3
"""Label all Friends utterances that have MELD matches, preserving full episode context."""

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
DEFAULT_OUTPUT = Path("workzone/meld_labeled_results.json")

EMOTION_LABELS: dict[str, str] = {
    "anger": "A confrontational emotion caused by being offended, blocked, treated unfairly, misunderstood, controlled, or having an important goal harmed. Low anger may appear as annoyance, dissatisfaction, resistance, or irritation.",
    "disgust": "Aversion, repulsion, rejection, or a desire to distance oneself from an object, behavior, idea, bodily/moral contamination, or situation.",
    "fear": "A defensive emotion in response to threat, danger, punishment, loss, relationship damage, failure, or an anticipated bad outcome. Includes worry, nervousness, concern, and anxiety.",
    "happiness": "A positive emotion such as pleasure, joy, amusement, satisfaction, optimism, or positive engagement. Use it for jokes only when the speaker is personally amused or enjoying the interaction.",
    "surprise": "An orienting reaction when reality suddenly violates expectations. Includes shock, disbelief, being caught off guard, or sudden realization.",
    "sadness": "A low-energy negative emotion caused by loss, failure, separation, rejection, disappointment, damaged relationships, frustrated wishes, or sympathy for another person's pain.",
    "contentment": "A calm positive emotion when needs are met, the situation feels comfortable and stable, relationships feel secure, or there is no urgent unmet need.",
    "relief": "A relaxing positive emotion after a prior threat, pressure, worry, uncertain bad outcome, or embarrassment risk is removed or reduced.",
    "interest": "Attention is drawn to information, a person, object, activity, or possibility, with a desire to learn more, confirm, observe, or participate.",
    "contempt": "A devaluing judgment that someone or something is stupid, inferior, ridiculous, unworthy of respect, or laughable.",
    "shame": "Painful self-evaluation when one's identity, image, worth, or socially visible self seems flawed or exposed, often with a desire to hide.",
    "guilt": "Self-blame from believing one has done something wrong, hurt someone, violated a duty, broken a promise, or failed a moral standard.",
    "embarrassment": "Mild to moderate discomfort from a social mistake, awkward exposure, being noticed, private information being revealed, or an inappropriate situation.",
    "neutral": "No clear emotion, or not enough evidence to identify a specific emotion. Use it for factual statements, ordinary confirmations, functional transitions, or lines with no clear appraisal.",
}

INTENSITIES: dict[str, str] = {
    "low": "Weak: the emotion is present but mild.",
    "medium": "Medium: the emotion is clearly identifiable but not an intense outburst or extreme state.",
    "high": "High: the emotion is strong, usually with obvious escalation, overwhelming reaction, or strong action tendency.",
    "none": "None: only used for neutral.",
}

MELD_EMOTION_MAP: dict[str, str] = {
    "joy": "happiness",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label every Friends utterance with a MELD match, using MELD as a weak reference. "
            "For target utterance n, the request includes the whole episode prefix through n."
        )
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-timestamp-output",
        action="store_true",
        help="Write exactly to --output instead of appending a timestamp before the file suffix.",
    )
    parser.add_argument("--start-season", type=int, default=1)
    parser.add_argument("--start-episode", type=int, default=1)
    parser.add_argument("--start-utterance-id", type=int, default=1)
    parser.add_argument(
        "--start-global-utterance-id",
        type=int,
        default=None,
        help="Optional global utterance id resume point. If set, it overrides season/episode/utterance start.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of MELD-matched utterances to label in this run.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode. Disabled by default to reduce labeling cost.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Deprecated compatibility flag; thinking is disabled unless --enable-thinking is set.",
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


def load_episode_description(path: Path, season: int, episode: int) -> str:
    for row in read_jsonl(path):
        if row.get("season") == season and row.get("episode") == episode:
            description = row.get("description")
            if isinstance(description, str) and description.strip():
                return description.strip()
            raise SystemExit(f"Empty description for S{season:02d}E{episode:02d}")
    raise SystemExit(f"Description not found for S{season:02d}E{episode:02d} in {path}")


def load_events_by_episode(path: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_episode: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        content = row.get("content")
        if not isinstance(content, dict):
            continue
        season = row.get("season")
        episode = row.get("episode")
        if not isinstance(season, int) or not isinstance(episode, int):
            continue

        event: dict[str, Any] | None = None
        if row.get("type") == "description":
            description = content.get("description")
            if isinstance(description, str) and description.strip():
                event = {"type": "description", "description": description.strip()}
        elif row.get("type") == "utterance":
            event = {
                "type": "utterance",
                "utterance_id": content["utterance_id"],
                "global_utterance_id": content.get("global_utterance_id"),
                "speaker": content["speaker"],
                "utterance": content["utterance"],
                "inline_description": content.get("inline_description", []),
            }
            meld = row.get("meld")
            if isinstance(meld, dict):
                event["meld"] = meld

        if event is not None:
            by_episode.setdefault((season, episode), []).append(event)

    return by_episode


def event_utterances(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == "utterance"]


def format_event(event: dict[str, Any]) -> str:
    if event.get("type") == "description":
        return f'description: {event["description"]}'
    line = f'{event["utterance_id"]}. {event["speaker"]}: {event["utterance"]}'
    inline = event.get("inline_description")
    if isinstance(inline, list) and inline:
        line += " [" + "; ".join(str(item) for item in inline) + "]"
    return line


def labels_text() -> str:
    label_lines = [f"- {name}: {description}" for name, description in EMOTION_LABELS.items()]
    intensity_lines = [f"- {name}: {description}" for name, description in INTENSITIES.items()]
    return "\n".join(
        [
            "Available emotion labels:",
            *label_lines,
            "",
            "Available intensity labels:",
            *intensity_lines,
        ]
    )


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are a careful dialogue emotion labeling assistant.",
            "Return only valid JSON. Do not include Markdown, comments, or extra text.",
            "Base the label on the target utterance, its speaker, the episode synopsis, the MELD weak reference, and prior dialogue context.",
        ]
    )


def format_meld_reference(meld: dict[str, Any]) -> str:
    mapped_emotions = [
        MELD_EMOTION_MAP[emotion]
        for emotion in meld.get("emotion", [])
        if isinstance(emotion, str) and emotion in MELD_EMOTION_MAP
    ]
    return json.dumps(
        {
            "emotion": meld.get("emotion"),
            "mapped_emotion": mapped_emotions,
            "sentiment": meld.get("sentiment"),
            "match_status": meld.get("match_status"),
            "match_score": meld.get("match_score"),
            "match_text_score": meld.get("match_text_score"),
            "meld_group_size": meld.get("meld_group_size"),
        },
        ensure_ascii=False,
    )


def build_meld_user_prompt(
    season: int,
    episode: int,
    description: str,
    events: list[dict[str, Any]],
    meld: dict[str, Any],
) -> str:
    transcript = "\n".join(format_event(event) for event in events)
    target = next(event for event in reversed(events) if event.get("type") == "utterance")
    target_json = json.dumps(
        {
            "utterance_id": target["utterance_id"],
            "speaker": target["speaker"],
            "utterance": target["utterance"],
            "inline_description": target.get("inline_description", []),
        },
        ensure_ascii=False,
    )
    return "\n".join(
        [
            f"Episode: S{season:02d}E{episode:02d}",
            "",
            "Task: Label the emotion of the final utterance in the transcript below.",
            "The transcript contains all previous events plus the current target utterance.",
            "description lines are stage or scene context, not utterances to label.",
            "Only label the final utterance. Use earlier events only as context.",
            "",
            "MELD reference label for the target utterance:",
            format_meld_reference(meld),
            "",
            "Use the MELD label as a weak reference, not as a hard answer.",
            "Use mapped_emotion when comparing MELD to this project's label set.",
            "If the dialogue evidence supports MELD, keep the aligned label and explain why.",
            "If the dialogue evidence clearly conflicts with MELD, choose the better label and explain the conflict briefly.",
            "",
            labels_text(),
            "",
            "Rules:",
            "- emotions must be selected from the emotion labels above. You **can** choose more than one emotion, not must",
            "- intensities must align one-to-one with emotions.",
            "- non-neutral emotions use low, medium, or high.",
            "- neutral must be exactly emotions=[\"neutral\"] and intensities=[\"none\"].",
            "- Keep reason concise and grounded in the dialogue context.",
            "- The reason should be useful for later human review, not a long chain of thought.",
            "- MELD label only contains 7 kinds of emotions. That's one reason why you use them as a weak reference.",
            "",
            "Return this JSON object shape:",
            '{"utterance_id": 1, "speaker": "Monica", "emotions": ["neutral"], '
            '"intensities": ["none"], "reason": "brief evidence-based reason", '
            '"meld_agreement": "agree|partial|disagree"}',
            "",
            "Episode synopsis:",
            description,
            "",
            "Transcript; the final line is the current target utterance:",
            transcript,
            "",
            "Current target utterance. Label this exact utterance, not any previous line:",
            target_json,
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


def validate_label(label: dict[str, Any], utterance: dict[str, Any]) -> None:
    emotions = label.get("emotions")
    intensities = label.get("intensities")
    if label.get("utterance_id") != utterance["utterance_id"]:
        raise ValueError(
            f"Utterance id mismatch: expected {utterance['utterance_id']}, got {label.get('utterance_id')}"
        )
    if label.get("speaker") != utterance["speaker"]:
        raise ValueError(
            f"Speaker mismatch for utterance {utterance['utterance_id']}: "
            f"expected {utterance['speaker']}, got {label.get('speaker')}"
        )
    if not isinstance(emotions, list) or not emotions:
        raise ValueError(f"Missing emotions list for utterance {utterance['utterance_id']}")
    if not isinstance(intensities, list) or len(intensities) != len(emotions):
        raise ValueError(f"Invalid intensities for utterance {utterance['utterance_id']}")
    if any(emotion not in EMOTION_LABELS for emotion in emotions):
        raise ValueError(f"Invalid emotion in utterance {utterance['utterance_id']}: {emotions}")
    if any(intensity not in INTENSITIES for intensity in intensities):
        raise ValueError(f"Invalid intensity in utterance {utterance['utterance_id']}: {intensities}")
    if "neutral" in emotions and emotions != ["neutral"]:
        raise ValueError(f"Neutral cannot be mixed for utterance {utterance['utterance_id']}")
    if emotions == ["neutral"] and intensities != ["none"]:
        raise ValueError(f"Neutral must use none intensity for utterance {utterance['utterance_id']}")
    if emotions != ["neutral"] and any(intensity == "none" for intensity in intensities):
        raise ValueError(f"Only neutral may use none intensity for utterance {utterance['utterance_id']}")
    if not isinstance(label.get("reason"), str) or not label["reason"].strip():
        raise ValueError(f"Missing reason for utterance {utterance['utterance_id']}")
    if label.get("meld_agreement") not in {"agree", "partial", "disagree"}:
        raise ValueError(f"Invalid meld_agreement for utterance {utterance['utterance_id']}")


def parse_label_json(content: str, utterance: dict[str, Any]) -> dict[str, Any]:
    if not content.strip():
        raise ValueError(f"Model returned empty content for utterance {utterance['utterance_id']}")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON for utterance {utterance['utterance_id']}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Model returned non-object JSON for utterance {utterance['utterance_id']}")

    model_utterance_id = parsed.get("utterance_id")
    model_speaker = parsed.get("speaker")
    if model_utterance_id != utterance["utterance_id"]:
        parsed["model_utterance_id"] = model_utterance_id
    if model_speaker != utterance["speaker"]:
        parsed["model_speaker"] = model_speaker
    parsed["utterance_id"] = utterance["utterance_id"]
    parsed["speaker"] = utterance["speaker"]
    validate_label(parsed, utterance)
    return parsed


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def request_parsed_label(
    client: Any,
    request_kwargs: dict[str, Any],
    utterance: dict[str, Any],
) -> tuple[Any, dict[str, Any], str, str]:
    response = client.chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", None) or ""
    try:
        label = parse_label_json(content, utterance)
    except ValueError as exc:
        finish_reason = response.choices[0].finish_reason
        usage = usage_to_dict(response.usage)
        content_preview = content[:200].replace("\n", "\\n")
        raise ValueError(
            f"{exc}; finish_reason={finish_reason}; usage={usage}; content_preview={content_preview!r}"
        ) from exc
    return response, label, content, reasoning_content


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def timestamped_output_path(path: Path, timestamp: str) -> Path:
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_{timestamp}{suffix}")


def prepare_output_path(args: argparse.Namespace) -> None:
    if args.no_timestamp_output:
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output = timestamped_output_path(args.output, timestamp)
    args.output_timestamp = timestamp


def start_key(args: argparse.Namespace) -> tuple[int, int, int]:
    return args.start_season, args.start_episode, args.start_utterance_id


def should_consider_target(args: argparse.Namespace, season: int, episode: int, utterance: dict[str, Any]) -> bool:
    global_id = utterance.get("global_utterance_id")
    if args.start_global_utterance_id is not None:
        return isinstance(global_id, int) and global_id >= args.start_global_utterance_id
    return (season, episode, int(utterance["utterance_id"])) >= start_key(args)


def result_key(result: dict[str, Any]) -> tuple[int, int, int] | None:
    utterance = result.get("utterance")
    if not isinstance(utterance, dict):
        return None
    season = result.get("season")
    episode = result.get("episode")
    global_id = utterance.get("global_utterance_id")
    if isinstance(season, int) and isinstance(episode, int) and isinstance(global_id, int):
        return season, episode, global_id
    return None


def build_output(args: argparse.Namespace) -> dict[str, Any]:
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "records": str(args.records),
        "descriptions": str(args.descriptions),
        "output": str(args.output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_season": args.start_season,
        "start_episode": args.start_episode,
        "start_utterance_id": args.start_utterance_id,
        "start_global_utterance_id": args.start_global_utterance_id,
        "target_filter": "Only utterances with a meld field are labeled.",
        "request_strategy": "For target utterance n, send the fixed instruction, MELD weak reference, episode synopsis, and episode events 1..n; the newest utterance is appended at the end for cache locality.",
    }
    if getattr(args, "output_timestamp", None):
        metadata["output_timestamp"] = args.output_timestamp
    return {
        "metadata": {
            **metadata,
        },
        "results": [],
    }


def load_resume_output(args: argparse.Namespace) -> tuple[dict[str, Any], set[tuple[int, int, int]]]:
    output = build_output(args)
    if not args.output.exists():
        return output, set()

    try:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cannot resume from invalid JSON output {args.output}: {exc}") from exc
    if not isinstance(existing, dict) or not isinstance(existing.get("results"), list):
        raise SystemExit(f"Cannot resume: {args.output} does not contain a results list")

    output["metadata"].update(existing.get("metadata", {}))
    output["metadata"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
    output["results"] = existing["results"]

    completed: set[tuple[int, int, int]] = set()
    for result in output["results"]:
        if not isinstance(result, dict):
            continue
        key = result_key(result)
        if key is not None:
            completed.add(key)
    return output, completed


def prefix_events_for_target(events: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    prefix_events: list[dict[str, Any]] = []
    for event in events:
        prefix_events.append(event)
        if event.get("type") == "utterance" and event["utterance_id"] == target["utterance_id"]:
            return prefix_events
    raise SystemExit(f"Could not build transcript prefix for utterance {target['utterance_id']}")


def main() -> None:
    args = build_parser().parse_args()
    prepare_output_path(args)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: openai. Install it with `uv add openai`.") from exc

    events_by_episode = load_events_by_episode(args.records)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    output, completed = load_resume_output(args)
    system_prompt = build_system_prompt()

    labeled_this_run = 0
    total_targets_seen = 0
    for season, episode in sorted(events_by_episode):
        events = events_by_episode[(season, episode)]
        utterances = event_utterances(events)
        if not utterances:
            continue
        description = load_episode_description(args.descriptions, season, episode)

        for utterance in utterances:
            if "meld" not in utterance:
                continue
            if not should_consider_target(args, season, episode, utterance):
                continue
            total_targets_seen += 1

            global_id = utterance.get("global_utterance_id")
            key = (season, episode, global_id) if isinstance(global_id, int) else None
            if key is not None and key in completed:
                continue
            if args.limit is not None and labeled_this_run >= args.limit:
                write_output(args.output, output)
                print(f"Reached --limit {args.limit}; wrote {args.output}")
                return

            prefix_events = prefix_events_for_target(events, utterance)
            request_kwargs: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": build_meld_user_prompt(
                            season,
                            episode,
                            description,
                            prefix_events,
                            utterance["meld"],
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            if args.enable_thinking and not args.disable_thinking:
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": args.reasoning_effort,
                }
            else:
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response, label, content, reasoning_content = request_parsed_label(
                client,
                request_kwargs,
                utterance,
            )
            output["results"].append(
                {
                    "season": season,
                    "episode": episode,
                    "utterance": utterance,
                    "meld": utterance.get("meld"),
                    "label": label,
                    "cot": reasoning_content,
                    "usage": usage_to_dict(response.usage),
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
            if key is not None:
                completed.add(key)
            labeled_this_run += 1
            print(
                f"S{season:02d}E{episode:02d} utterance {utterance['utterance_id']} "
                f"global {utterance.get('global_utterance_id')} ({labeled_this_run} this run)"
            )
            write_output(args.output, output)

    output["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    output["metadata"]["targets_seen_this_run"] = total_targets_seen
    output["metadata"]["labeled_this_run"] = labeled_this_run
    write_output(args.output, output)


if __name__ == "__main__":
    main()
