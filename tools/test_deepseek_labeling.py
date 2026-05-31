#!/usr/bin/env python3
"""Smoke test for DeepSeek labeling calls via the OpenAI SDK."""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_SYSTEM_PROMPT = (
    "You are a data labeling assistant."
)
DEFAULT_PROMPT = (
    "讲讲老友记里的joey是个什么样的人"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test DeepSeek API calls before running data labeling."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Do not send extra_body thinking={type: enabled}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it with `pip3 install openai`."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    request_kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "stream": False,
        "reasoning_effort": args.reasoning_effort,
    }
    if not args.disable_thinking:
        request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    response = client.chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    print(message.content or "", file=sys.stdout)


if __name__ == "__main__":
    main()
