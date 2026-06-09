from __future__ import annotations

from typing import Any

from acagent.retrieval import KeywordEventSearcher
from acagent.runner.context import WorkflowInput
from acagent.schemas import SearchQuery
from acagent.tools.base import AgentTool


def build_search_events_tool(default_top_k: int = 5) -> AgentTool:
    return AgentTool(
        name="search_events",
        description="Search visible online event memory. Empty arguments return no results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "characters": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "top_k": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "short_summary": {"type": "string"},
                            "score": {"type": "integer"},
                            "matched_terms": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                }
            },
            "required": ["events"],
        },
        handler=_search_events_handler(default_top_k),
        is_read_only=True,
        is_concurrency_safe=True,
    )


def build_read_event_tool() -> AgentTool:
    return AgentTool(
        name="read_event",
        description="Read one event detail by event_id. Empty arguments return an empty object.",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "detail_id": {"type": "string"},
                "event_id": {"type": "string"},
                "description": {"type": "string"},
            },
        },
        handler=_read_event,
        is_read_only=True,
        is_concurrency_safe=False,
    )


def _search_events_handler(default_top_k: int):
    def _search_events(workflow_input: WorkflowInput, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments:
            return {"events": []}
        memory = _load_memory(workflow_input)
        query = SearchQuery(
            query=str(arguments.get("query", "")),
            characters=list(arguments.get("characters", [])),
            keywords=list(arguments.get("keywords", [])),
            top_k=int(arguments.get("top_k", default_top_k)),
        )
        searcher = KeywordEventSearcher(memory)
        results = searcher.search_events(query)
        return {
            "events": [
                {
                    "event_id": item.event.event_id,
                    "short_summary": item.event.short_summary,
                    "score": item.score,
                    "matched_terms": item.matched_terms,
                }
                for item in results
            ]
        }

    return _search_events


def _read_event(workflow_input: WorkflowInput, arguments: dict[str, Any]) -> dict[str, Any]:
    event_id = str(arguments.get("event_id", ""))
    if not event_id:
        return {}
    memory = _load_memory(workflow_input)
    detail = KeywordEventSearcher(memory).read_event(event_id)
    if detail is None:
        return {}
    workflow_input.retrieved_event_details.append(detail)
    return {
        "detail_id": detail.detail_id,
        "event_id": detail.event_id,
        "description": detail.description,
    }


def _load_memory(workflow_input: WorkflowInput):
    if workflow_input.memory_store is None:
        return workflow_input.memory
    workflow_input.memory = workflow_input.memory_store.load()
    return workflow_input.memory
