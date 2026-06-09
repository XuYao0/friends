# AC Agent Run Recovery

This document records how to resume a failed or polluted AC Agent run from a
known-good chunk boundary.

## When To Use

Use this recovery path when:

1. `traces.jsonl` exists and contains complete tool calls and tool results.
2. The latest `memory.json` is no longer trustworthy.
3. You know the last good chunk index, such as `236`.
4. You want to keep the original output directory unchanged.

This method rebuilds memory by replaying successful `update_memory` tool calls
from trace records up to the chosen chunk.

## What It Recovers

The recovery tool reconstructs:

1. `memory.json` at the end of `--through-chunk`.
2. A `resume.yaml` config with `start_chunk_index = through_chunk + 1`.

It can optionally copy trace and prediction prefixes into the new output
directory, but the default recommended path is to keep the original run
unchanged and write new outputs to a fresh directory.

## Example

Recover the current default run from chunk 236:

```bash
PYTHONPATH=acagent/src python -m acagent.recovery \
  --trace acagent_outputs/default/traces.jsonl \
  --predictions acagent_outputs/default/predictions.jsonl \
  --through-chunk 236 \
  --output-dir acagent_outputs/resume_from_chunk_236
```

Expected output:

```text
memory_version=mem_00182
resume_config=acagent_outputs/resume_from_chunk_236/resume.yaml
```

Then continue the run:

```bash
PYTHONPATH=acagent/src python -c 'from acagent.runner import build_workflow_runner_from_config; build_workflow_runner_from_config("acagent_outputs/resume_from_chunk_236/resume.yaml").run()'
```

## How It Works

`acagent.recovery` reads each trace record up to `--through-chunk`.

For each LLM turn, it checks whether the recorded tool call is:

```text
update_memory
```

and whether the tool result has no error. For old trace formats, success is
recognized by:

```json
{"tool_name": "update_memory", "error": null}
```

The tool arguments are converted back into a memory delta and applied to a fresh
`MemoryState`. After replay, the recovered memory version must match the trace
record's `memory_version_after_agent`; otherwise recovery fails instead of
silently writing a bad memory file.

## Resume Config

The generated `resume.yaml` is copied from the base config and overrides:

```yaml
output_dir: acagent_outputs/resume_from_chunk_236
start_chunk_index: 237
```

`WorkflowRunner` uses `start_chunk_index` to skip earlier transcript chunks:

```text
chunk 1 ... chunk 236 skipped
chunk 237 processed first
```

The trace logger also starts counting from `start_chunk_index - 1`, so resumed
trace ids continue from the original chunk number.

## Notes

1. The original output directory is not modified.
2. The recovered memory depends on successful `update_memory` calls being present
   in trace records.
3. Failed tool calls, read-only tools, and fallback predictions are not replayed.
4. If a historic tool call omitted `source_utterance_ids`, replay may not recover
   that bookkeeping field exactly. Core character, relationship, event, and
   current-event memory contents are recovered from the stored tool arguments.
5. After recovery, run tests before relying on the resumed path:

```bash
PYTHONPATH=acagent/src uv run --with pytest pytest acagent/tests
```
