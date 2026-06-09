---
name: acagent-prediction-audit
description: Use when inspecting AC Agent prediction JSONL and trace JSONL files to find hard-coded fallback labels, empty labels, parse failures, tool budget exhaustion, or root causes of invalid emotion predictions.
---

# AC Agent Prediction Audit

Use this skill to audit `predictions.jsonl` and corresponding `traces.jsonl`
for invalid, hard-coded, or suspicious AC Agent label outputs.

## Goal

Find whether bad predictions came from:

- program-generated fallback neutral
- model output parse errors
- Markdown fenced JSON
- empty assistant content
- empty `emotions` / `intensities`
- invalid labels or intensities
- neutral mixed with other emotions
- tool budget or turn budget exhaustion

The priority is to identify the root cause, not only count bad rows.

## Quick Scan

Run this from the project root:

```bash
PYTHONPATH=acagent/src python - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path("acagent_outputs/default/predictions.jsonl")
allowed_emotions = {
    "anger", "disgust", "fear", "happiness", "surprise", "sadness",
    "contentment", "relief", "interest", "contempt", "shame", "guilt",
    "embarrassment", "neutral",
}
allowed_intensities = {"low", "medium", "high", "none"}

counts = Counter()
examples = defaultdict(list)
total = 0

for line_no, line in enumerate(p.open(encoding="utf-8"), start=1):
    if not line.strip():
        continue
    total += 1
    try:
        row = json.loads(line)
    except Exception as exc:
        counts["invalid_json_line"] += 1
        examples["invalid_json_line"].append((line_no, str(exc)))
        continue

    pred = row.get("prediction") or {}
    raw = row.get("raw_prediction") or {}
    analysis = pred.get("analysis") or {}
    emotions = pred.get("emotions")
    intensities = pred.get("intensities")
    uncertainties = analysis.get("uncertainties") or []
    final_reason = analysis.get("final_reason", "")

    def add(kind, detail=None):
        counts[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(
                (
                    line_no,
                    row.get("trace_id"),
                    row.get("utterance_id"),
                    detail
                    or {
                        "emotions": emotions,
                        "intensities": intensities,
                        "uncertainties": uncertainties,
                        "final_reason": final_reason[:160],
                    },
                )
            )

    if emotions == ["neutral"] and intensities == ["none"] and "Fallback neutral prediction" in final_reason:
        add("hardcoded_fallback_neutral")
    if "max_llm_turns_exhausted" in uncertainties:
        add("max_llm_turns_exhausted_marker")
    if not isinstance(emotions, list) or not emotions:
        add("missing_or_empty_emotions")
    if not isinstance(intensities, list) or not intensities:
        add("missing_or_empty_intensities")
    if isinstance(emotions, list) and isinstance(intensities, list) and len(emotions) != len(intensities):
        add("emotion_intensity_length_mismatch")
    if isinstance(emotions, list):
        bad = [item for item in emotions if item not in allowed_emotions]
        if bad:
            add("unknown_emotion_label", bad)
    if isinstance(intensities, list):
        bad = [item for item in intensities if item not in allowed_intensities]
        if bad:
            add("unknown_intensity_label", bad)
    if isinstance(emotions, list) and "neutral" in emotions and not (
        emotions == ["neutral"] and intensities == ["none"]
    ):
        add("neutral_mixed_or_bad_intensity")
    if raw == {}:
        add("empty_raw_prediction")

print("total", total)
for key, value in counts.most_common():
    print(key, value)
print("\\nEXAMPLES")
for key, rows in examples.items():
    print("\\n##", key)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False)[:1000])
PY
```

## Trace Root-Cause Inspection

For suspicious trace ids, inspect the raw assistant content and parser errors.

```bash
PYTHONPATH=acagent/src python - <<'PY'
import json

trace_ids = {"trace_00287", "trace_00298", "trace_00328", "trace_00342"}

for line in open("acagent_outputs/default/traces.jsonl", encoding="utf-8"):
    record = json.loads(line)
    if record.get("trace_id") not in trace_ids:
        continue
    print("\\n===", record["trace_id"], record["utterance_id"], "final=", record.get("final_prediction"))
    turns = record.get("llm_turns", [])
    for index, turn in enumerate(turns, start=1):
        output = turn.get("output") or {}
        error = (output.get("error") or {}).get("code")
        raw = output.get("raw") or {}
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        final_prediction = output.get("final_prediction")
        if error or final_prediction is not None or index >= len(turns) - 2:
            print("TURN", index, "finish_reason=", choice.get("finish_reason"), "err=", error)
            print("fp=", final_prediction)
            print("content=", repr((message.get("content") or "")[:1200]))
            print("reasoning_head=", repr((message.get("reasoning_content") or "")[:400]))
            print("tool=", (output.get("tool_call") or {}).get("name"))
            print("---")
PY
```

## Known Root Causes

### Markdown Fenced JSON

Symptom:

```text
final_prediction_parse_error: Expecting value: line 1 column 1 (char 0)
content starts with ```json
```

Cause: parser calls `json.loads(content)` and the first character is a
backtick, not `{`.

Fix: strip Markdown code fences before JSON parsing, and prompt the model to
return raw JSON only.

### Empty Labels

Symptom:

```json
{"emotions": [], "intensities": [], "analysis": {}}
```

Typical trace pattern:

1. Earlier turns returned valid fenced JSON and failed parser.
2. Later turn returned `content=""` with non-empty `reasoning_content`.
3. Old parser used `json.loads(content or "{}")`.
4. Empty `{}` became `emotions=[]`, `intensities=[]`.
5. Old validation allowed empty lists because their lengths matched.

Fix: reject empty `emotions` and empty `intensities` in
`EmotionPrediction.validate()`.

### Hard-Coded Fallback Neutral

Symptom:

```json
{
  "emotions": ["neutral"],
  "intensities": ["none"],
  "analysis": {
    "uncertainties": ["max_llm_turns_exhausted"],
    "final_reason": "Fallback neutral prediction because no valid model prediction was available."
  }
}
```

Cause: old `LabelLoop` generated a neutral prediction after `max_llm_turns`
instead of leaving the prediction missing.

Fix: on `max_llm_turns_exhausted`, write trace error only and return
`prediction_record=None`; do not append to `predictions.jsonl`.

### Tool Budget Exhaustion

Symptom:

Trace contains tool result errors such as:

```text
max_tool_calls_exceeded
max_read_events_exceeded
```

Expected behavior: after a hard tool limit, the next model call should use:

```python
tools = []
tool_choice = "none"
```

and should ask the model to return final prediction JSON from existing context
and tool results.

## Validation

After fixes, run:

```bash
PYTHONPATH=acagent/src uv run --with pytest pytest acagent/tests
```

Then rerun the quick scan on new output directories. Historical polluted files
will still show old errors unless regenerated or excluded.
