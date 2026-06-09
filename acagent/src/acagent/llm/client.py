from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from string import Template
from typing import Any


class PromptRenderer:
    def render_text(self, template_text: str, variables: dict[str, object]) -> str:
        template = Template(template_text)
        return template.safe_substitute(
            {key: self._stringify(value) for key, value in variables.items()}
        )

    def render_file(self, path: str | Path, variables: dict[str, object]) -> str:
        return self.render_text(Path(path).read_text(encoding="utf-8"), variables)

    def _stringify(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bool | int | float):
            return str(value)
        normalized = self._normalize(value)
        try:
            return json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
        except TypeError:
            return str(value)

    def _normalize(self, value: object) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._normalize(item) for item in value]
        return value


class LLMClient:
    """Interface placeholder for future model calls."""

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, object]:
        raise NotImplementedError("LLM integration is intentionally not wired in the scaffold.")
