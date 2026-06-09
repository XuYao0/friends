---
name: acagent-run-recovery
description: Use when an AC Agent run under acagent_outputs is polluted or interrupted and must be resumed from a known-good chunk boundary by rebuilding memory from persisted traces and generating a resume config.
---

# AC Agent Run Recovery

Use this skill for AC Agent recovery tasks where a run has bad predictions,
fallback outputs, or an untrustworthy final `memory.json`, but `traces.jsonl`
still contains complete LLM turns, tool calls, and tool results.

## Goal

Recover a trustworthy memory state at the end of a known-good chunk, then resume
the run from the next chunk in a fresh output directory.

Do not overwrite the original output directory unless the user explicitly asks.

## Workflow

1. Inspect output files:

```bash
ls -lh acagent_outputs/default
wc -l acagent_outputs/default/traces.jsonl acagent_outputs/default/predictions.jsonl
```

2. Identify the last good chunk boundary.

Use JSON parsing, not raw `sed`, because trace records are large:

```bash
PYTHONPATH=acagent/src python - <<'PY'
import json
p = "acagent_outputs/default/traces.jsonl"
for i, line in enumerate(open(p, encoding="utf-8"), start=1):
    r = json.loads(line)
    if 232 <= i <= 241:
        print(
            i,
            r["trace_id"],
            r["utterance_id"],
            r["is_eval_point"],
            r["memory_version_before_agent"],
            r["memory_version_after_agent"],
            bool(r.get("final_prediction")),
        )
PY
```

3. Scan later predictions for pollution.

Common pollution indicators:

- `Fallback neutral prediction because no valid model prediction was available.`
- `max_llm_turns_exhausted`
- empty `emotions` / `intensities`
- repeated parse errors in trace

4. Rebuild memory from trace with `acagent.recovery`.

Example for chunk 236:

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

5. Confirm the generated config.

```bash
sed -n '1,80p' acagent_outputs/resume_from_chunk_236/resume.yaml
```

It should contain:

```yaml
output_dir: acagent_outputs/resume_from_chunk_236
start_chunk_index: 237
```

6. Count remaining chunks without calling the LLM.

```bash
PYTHONPATH=acagent/src python - <<'PY'
from collections import Counter
from acagent.config import ExperimentConfig
from acagent.transcript import FriendsTranscriptChunkSource

c = ExperimentConfig.from_yaml("acagent_outputs/resume_from_chunk_236/resume.yaml")
chunks = list(
    FriendsTranscriptChunkSource(
        c.transcript_path,
        batch_size=c.batch_size,
        max_utterances=c.max_utterances,
    ).iter_chunks()
)
remaining = chunks[c.start_chunk_index - 1 :]
counts = Counter(chunk.status for chunk in remaining)
print("total", len(chunks))
print("start_chunk_index", c.start_chunk_index)
print("remaining", len(remaining))
print("remaining_update_memory", counts["update_memory"])
print("remaining_label", counts["label"])
print("first_remaining", remaining[0].status, remaining[0].target_utterance_id)
PY
```

7. Resume the run.

```bash
PYTHONPATH=acagent/src python -c 'from acagent.runner import build_workflow_runner_from_config; build_workflow_runner_from_config("acagent_outputs/resume_from_chunk_236/resume.yaml").run()'
```

## How Recovery Works

`acagent.recovery` starts from an empty `MemoryState`, reads trace records up to
`--through-chunk`, finds successful `update_memory` tool calls, converts their
arguments back to memory deltas, and applies them in order.

It verifies replayed memory versions against:

- each tool result's `memory_version_before`
- each tool result's `memory_version_after`
- the chunk trace's `memory_version_after_agent`

If the versions do not match, stop and investigate instead of continuing.

## Important Notes

- Old trace format may use `{"tool_name": "update_memory", "error": null}`
  instead of `ok: true`; treat `error: null` as success.
- Do not replay read-only tools or failed tool calls.
- Do not replay predictions into memory.
- If the original trace omitted `source_utterance_ids`, replay may not recover
  that bookkeeping field exactly. Core memory content is still recovered from
  stored tool arguments.
- Prefer a new output directory such as `resume_from_chunk_236_v2` if an earlier
  resume directory already contains bad predictions.

## Validation

Run tests after recovery code or config changes:

```bash
PYTHONPATH=acagent/src uv run --with pytest pytest acagent/tests
```
