from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from acagent.memory import MemoryState
from acagent.storage import JsonMemoryStore
from acagent.tools.memory_tools import _memory_delta_from_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover an AC Agent run from persisted trace.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--base-config", type=Path, default=Path("acagent/configs/default.yaml"))
    parser.add_argument("--through-chunk", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-prefix",
        action="store_true",
        help="Copy trace and prediction records up to --through-chunk into the output dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    memory = rebuild_memory_from_trace(args.trace, through_chunk=args.through_chunk)
    JsonMemoryStore(output_dir / "memory.json").save(memory)

    if args.copy_prefix:
        copy_trace_prefix(args.trace, output_dir / "traces.jsonl", through_chunk=args.through_chunk)
        if args.predictions is not None:
            copy_prediction_prefix(
                args.predictions,
                output_dir / "predictions.jsonl",
                through_chunk=args.through_chunk,
            )

    write_resume_config(
        base_config=args.base_config,
        output_path=output_dir / "resume.yaml",
        output_dir=output_dir,
        start_chunk_index=args.through_chunk + 1,
    )
    print(f"memory_version={memory.version_id}")
    print(f"resume_config={output_dir / 'resume.yaml'}")


def rebuild_memory_from_trace(trace_path: Path, *, through_chunk: int) -> MemoryState:
    memory = MemoryState()
    last_record: dict[str, Any] | None = None
    for chunk_index, record in _iter_trace_records(trace_path):
        if chunk_index > through_chunk:
            break
        last_record = record
        for turn in record.get("llm_turns", []):
            tool_result = turn.get("tool_result") or {}
            output = turn.get("output") or {}
            tool_call = output.get("tool_call") or {}
            if tool_call.get("name") != "update_memory":
                continue
            if tool_result.get("error") is not None or tool_result.get("tool_name") != "update_memory":
                continue
            expected_before = tool_result.get("memory_version_before")
            if expected_before and expected_before != memory.version_id:
                raise ValueError(
                    f"Memory version mismatch before chunk {chunk_index}: "
                    f"trace expects {expected_before}, replay has {memory.version_id}"
                )
            memory.apply_delta(_memory_delta_from_dict(dict(tool_call.get("arguments") or {})))
            expected_after = tool_result.get("memory_version_after")
            if expected_after and expected_after != memory.version_id:
                raise ValueError(
                    f"Memory version mismatch after chunk {chunk_index}: "
                    f"trace expects {expected_after}, replay has {memory.version_id}"
                )

    if last_record is None:
        raise ValueError(f"No trace records found in {trace_path}")
    expected_final = last_record.get("memory_version_after_agent")
    if expected_final and expected_final != memory.version_id:
        raise ValueError(
            f"Final memory version mismatch at chunk {through_chunk}: "
            f"trace expects {expected_final}, replay has {memory.version_id}"
        )
    return memory


def copy_trace_prefix(trace_path: Path, output_path: Path, *, through_chunk: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for chunk_index, line in enumerate(source, start=1):
            if chunk_index > through_chunk:
                break
            target.write(line)


def copy_prediction_prefix(
    prediction_path: Path,
    output_path: Path,
    *,
    through_chunk: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            trace_id = str(record.get("trace_id", ""))
            trace_number = _trace_number(trace_id)
            if trace_number is not None and trace_number <= through_chunk:
                target.write(line)


def write_resume_config(
    *,
    base_config: Path,
    output_path: Path,
    output_dir: Path,
    start_chunk_index: int,
) -> None:
    values = _read_simple_yaml_lines(base_config)
    values["start_chunk_index"] = str(start_chunk_index)
    values["output_dir"] = str(output_dir)
    output_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _iter_trace_records(trace_path: Path) -> Any:
    with trace_path.open("r", encoding="utf-8") as file:
        for chunk_index, line in enumerate(file, start=1):
            if line.strip():
                yield chunk_index, json.loads(line)


def _trace_number(trace_id: str) -> int | None:
    if not trace_id.startswith("trace_"):
        return None
    try:
        return int(trace_id.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _read_simple_yaml_lines(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#") or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


if __name__ == "__main__":
    main()
