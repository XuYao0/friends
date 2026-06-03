#!/usr/bin/env python3
"""Label S01E01 utterance emotions with DeepSeek and save token usage."""

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
DEFAULT_RECORDS = Path("screenplays/converted_chunks/cleaned/friends_records.jsonl")
DEFAULT_DESCRIPTIONS = Path("screenplays/converted_chunks/episode_descriptions.jsonl")
DEFAULT_OUTPUT = Path("workzone/tmp.json")
DEFAULT_SEASON = 1
DEFAULT_EPISODE = 1

EMOTION_LABELS: list[tuple[str, str, str]] = [
    ("anger", "愤怒", "被冒犯、被阻碍、不公对待或目标受损时的对抗性情绪。"),
    ("disgust", "厌恶", "对对象、行为、想法或情境产生反感、排斥或想远离。"),
    ("fear", "恐惧", "面对威胁、危险、惩罚、损失或坏结果预期时的防御性情绪。"),
    ("happiness", "快乐", "一般正向愉悦、高兴、开心或愿望满足。"),
    ("surprise", "惊讶", "现实突然偏离预期时的定向反应，可正向、负向或中性。"),
    ("sadness", "悲伤", "失去、失败、分离、拒绝、失望或关系受损引发的低落。"),
    ("contentment", "满足", "需求被满足、处境舒适安稳、没有急迫未满足需求时的平静正向情绪。"),
    ("relief", "如释重负", "先前威胁、压力、担忧或坏结果预期解除后的放松性正向情绪。"),
    ("interest", "兴趣", "注意被对象、信息、人物或可能性吸引，并想继续了解、观察或参与。"),
    ("contempt", "轻蔑", "对对象产生贬低性评价，认为其愚蠢、低劣、不值得尊重或可笑。"),
    ("shame", "羞耻", "自我形象、身份或价值暴露出缺陷时产生的自我否定和想隐藏的情绪。"),
    ("guilt", "内疚", "认为自己做错事、伤害别人、违反责任或承诺时产生的自责。"),
    ("embarrassment", "尴尬", "轻度社交失误、失态、被撞见、私人信息暴露或场面不合适引发的不适。"),
    ("neutral", "中性", "无明显情绪，或证据不足以判断具体情绪。"),
]
INTENSITIES = {
    "low": "弱：情绪存在但轻微。",
    "medium": "中：情绪明确可识别，但未达到强烈爆发或极端状态。",
    "high": "强：情绪强烈，通常伴随明显爆发、压倒性反应或强烈行动倾向。",
    "none": "无：仅用于 neutral。",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use DeepSeek to label every utterance in Friends S01E01."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--start-utterance-id",
        type=int,
        default=1,
        help="Resume from this utterance id, preserving earlier results from --output.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Do not send extra_body thinking={type: enabled}.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
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


def load_events(path: Path, season: int, episode: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    utterances: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if row.get("season") != season or row.get("episode") != episode:
            continue
        content = row.get("content")
        if not isinstance(content, dict):
            continue
        if row.get("type") == "description":
            description = content.get("description")
            if isinstance(description, str) and description.strip():
                events.append({"type": "description", "description": description.strip()})
            continue
        if row.get("type") == "utterance":
            utterance = {
                "type": "utterance",
                "utterance_id": content["utterance_id"],
                "global_utterance_id": content.get("global_utterance_id"),
                "speaker": content["speaker"],
                "utterance": content["utterance"],
                "inline_description": content.get("inline_description", []),
            }
            events.append(utterance)
            utterances.append(utterance)

    if not utterances:
        raise SystemExit(f"No utterances found for S{season:02d}E{episode:02d} in {path}")
    return events


def event_utterances(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == "utterance"]


def labels_text() -> str:
    label_lines = [f"- {name} ({zh}): {description}" for name, zh, description in EMOTION_LABELS]
    intensity_lines = [f"- {name}: {description}" for name, description in INTENSITIES.items()]
    return "\n".join(
        [
            "可用 emotion 标签：",
            *label_lines,
            "",
            "可用 intensity 标签：",
            *intensity_lines,
        ]
    )


def format_event(event: dict[str, Any]) -> str:
    if event.get("type") == "description":
        return f'description: {event["description"]}'
    utterance = event
    line = f'{utterance["utterance_id"]}. {utterance["speaker"]}: {utterance["utterance"]}'
    inline = utterance.get("inline_description")
    if isinstance(inline, list) and inline:
        line += " [" + "; ".join(str(item) for item in inline) + "]"
    return line


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are a careful dialogue emotion labeling assistant.",
            "Return only valid JSON. Do not include Markdown, comments, or extra text.",
            "Base the label on the target utterance, its speaker, the episode synopsis, and prior dialogue context.",
        ]
    )


def build_user_prompt(
    season: int,
    episode: int,
    description: str,
    events: list[dict[str, Any]],
) -> str:
    transcript = "\n".join(format_event(event) for event in events)
    target = next(event for event in reversed(events) if event.get("type") == "utterance")
    target_json = json.dumps(
        {
            "utterance_id": target["utterance_id"],
            "speaker": target["speaker"],
            "utterance": target["utterance"],
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
            labels_text(),
            "",
            "Rules:",
            "- emotions must be selected from the emotion labels above.",
            "- intensities must align one-to-one with emotions.",
            "- non-neutral emotions use low, medium, or high.",
            "- neutral must be exactly emotions=[\"neutral\"] and intensities=[\"none\"].",
            "- Keep reason concise and grounded in the dialogue context.",
            "",
            "skills:",
            "- First infer the speaker's current inner state at the moment of speaking.",
            "- Put yourself in the speaker's position and ask what the speaker wants, fears, hides, resists, seeks, or reacts to.",
            "- Use a simple human-level interpretation. Most utterances do not require deep hidden motives.",
            "- Do not over-interpret jokes, filler lines, brief confirmations, or casual narration.",
            "- Distinguish the speaker's current feeling from the audience's amusement, the comedic function of the line.",
            "- A funny line is not happiness unless the speaker is personally amused.",
            "- When describing an event, actually the event happening now is not the described event, it's the 'describing' itself. Put yourself in the describing, feel the emotions in the describing, instead of the described event.",
            "- If the direct reading is neutral or low-intensity, keep it neutral or low unless there is clear textual or contextual evidence for a stronger emotion.",
            "- For short clarification questions, use interest when the speaker mainly seeks information; use surprise only when the utterance reacts to something unexpected or logically strange.",
            "",
            "Return this JSON object shape:",
            '{"utterance_id": 1, "speaker": "Monica", "emotions": ["neutral"], '
            '"intensities": ["none"], "reason": "brief reason"}',
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


def validate_label(label: dict[str, Any], utterance: dict[str, Any]) -> None:
    valid_emotions = {name for name, _, _ in EMOTION_LABELS}
    valid_intensities = set(INTENSITIES)
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
    if any(emotion not in valid_emotions for emotion in emotions):
        raise ValueError(f"Invalid emotion in utterance {utterance['utterance_id']}: {emotions}")
    if any(intensity not in valid_intensities for intensity in intensities):
        raise ValueError(f"Invalid intensity in utterance {utterance['utterance_id']}: {intensities}")
    if "neutral" in emotions and emotions != ["neutral"]:
        raise ValueError(f"Neutral cannot be mixed for utterance {utterance['utterance_id']}")
    if emotions == ["neutral"] and intensities != ["none"]:
        raise ValueError(f"Neutral must use none intensity for utterance {utterance['utterance_id']}")
    if emotions != ["neutral"] and any(intensity == "none" for intensity in intensities):
        raise ValueError(f"Only neutral may use none intensity for utterance {utterance['utterance_id']}")
    if not isinstance(label.get("reason"), str) or not label["reason"].strip():
        raise ValueError(f"Missing reason for utterance {utterance['utterance_id']}")


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def request_label(client: Any, request_kwargs: dict[str, Any]) -> Any:
    return client.chat.completions.create(**request_kwargs)


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


def build_output(args: argparse.Namespace, description: str) -> dict[str, Any]:
    return {
        "metadata": {
            "season": args.season,
            "episode": args.episode,
            "model": args.model,
            "base_url": args.base_url,
            "records": str(args.records),
            "descriptions": str(args.descriptions),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_strategy": "For utterance n, send the fixed instruction, episode synopsis, and utterances 1..n; the newest utterance is appended at the end for cache locality.",
        },
        "episode_description": description,
        "results": [],
    }


def load_resume_output(
    args: argparse.Namespace,
    description: str,
    utterance_ids: set[int],
) -> dict[str, Any]:
    output = build_output(args, description)
    if args.start_utterance_id <= 1:
        return output
    if not args.output.exists():
        raise SystemExit(
            f"Cannot resume from utterance {args.start_utterance_id}: output file does not exist: {args.output}"
        )

    try:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cannot resume from invalid JSON output {args.output}: {exc}") from exc
    if not isinstance(existing, dict) or not isinstance(existing.get("results"), list):
        raise SystemExit(f"Cannot resume: {args.output} does not contain a results list")

    kept_results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for result in existing["results"]:
        if not isinstance(result, dict):
            continue
        utterance = result.get("utterance")
        if not isinstance(utterance, dict):
            continue
        utterance_id = utterance.get("utterance_id")
        if not isinstance(utterance_id, int):
            continue
        if utterance_id >= args.start_utterance_id:
            continue
        if utterance_id not in utterance_ids:
            continue
        kept_results.append(result)
        seen_ids.add(utterance_id)

    missing = [
        utterance_id
        for utterance_id in range(1, args.start_utterance_id)
        if utterance_id in utterance_ids and utterance_id not in seen_ids
    ]
    if missing:
        formatted = ", ".join(str(utterance_id) for utterance_id in missing)
        raise SystemExit(f"Cannot resume: existing output is missing earlier utterance ids: {formatted}")

    output["metadata"].update(existing.get("metadata", {}))
    output["metadata"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
    output["metadata"]["start_utterance_id"] = args.start_utterance_id
    output["results"] = kept_results
    return output


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: openai. Install it with `uv add openai`.") from exc

    description = load_episode_description(args.descriptions, args.season, args.episode)
    events = load_events(args.records, args.season, args.episode)
    utterances = event_utterances(events)
    utterance_ids = {utterance["utterance_id"] for utterance in utterances}
    if args.start_utterance_id not in utterance_ids:
        raise SystemExit(
            f"Start utterance id {args.start_utterance_id} was not found in S{args.season:02d}E{args.episode:02d}"
        )
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    output = load_resume_output(args, description, utterance_ids)

    system_prompt = build_system_prompt()
    for index, utterance in enumerate(utterances, start=1):
        if utterance["utterance_id"] < args.start_utterance_id:
            continue
        prefix_events = []
        seen_target = False
        for event in events:
            prefix_events.append(event)
            if event.get("type") == "utterance" and event["utterance_id"] == utterance["utterance_id"]:
                seen_target = True
                break
        if not seen_target:
            raise SystemExit(f"Could not build transcript prefix for utterance {utterance['utterance_id']}")
        request_kwargs: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        args.season,
                        args.episode,
                        description,
                        prefix_events,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if not args.disable_thinking:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}, "reasoning_effort": args.reasoning_effort}
        else:
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response, label, content, reasoning_content = request_parsed_label(
            client,
            request_kwargs,
            utterance,
        )
        output["results"].append(
            {
                "utterance": utterance,
                "label": label,
                "cot": reasoning_content,
                "usage": usage_to_dict(response.usage),
                "finish_reason": response.choices[0].finish_reason,
            }
        )
        print(index)
        write_output(args.output, output)

    write_output(args.output, output)


if __name__ == "__main__":
    main()
