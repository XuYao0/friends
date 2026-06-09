from __future__ import annotations

import json
from typing import Any


def error_feedback_message(*, code: str, message: str, instruction: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": json.dumps(
            {
                "type": "acagent_error_feedback",
                "error": {
                    "code": code,
                    "message": message,
                },
                "instruction": instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def trace_error(*, code: str, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
    }
