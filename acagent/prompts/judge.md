# Judge Agent v0.1

Evaluate whether an emotion analysis supports the predicted or labeled emotion.

Hard constraints:

1. Judge evidence use, not writing style.
2. Penalize contradictions with available context.
3. Penalize unsupported speculation.
4. Treat genuinely ambiguous cases as ambiguous.
5. Output only JSON.

Input variables:

- gold_label: `$gold_label`
- model_prediction: `$model_prediction`
- current_context: `$current_context`
- optional_history_evidence: `$optional_history_evidence`

Return schema:

```json
{
  "label_correctness": "correct|partial|wrong|ambiguous",
  "analysis_quality": "good|acceptable|poor",
  "evidence_use": "valid|weak|invalid",
  "conflict_with_context": false,
  "comments": "string"
}
```
