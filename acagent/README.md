# AC Agent

AC Agent is a prototype framework for online long-context emotion recognition over screenplay dialogue.

The current implementation focuses on a deterministic outer workflow with
LLM tool-calling inner loops:

1. Transcript chunking from `screenplays/friends_records_renamed_with_selected.jsonl`.
2. Structured memory state.
3. Rule-based memory delta merge with optional local JSON persistence.
4. Keyword event retrieval.
5. Memory update and emotion labeling tool-calling loops.
6. Program-enforced tool budgets and recoverable LLM output errors.
7. JSONL prediction and trace persistence.

The default workflow processes the complete Friends JSONL file and ends at EOF.

```python
from acagent.runner import build_default_workflow_runner

runner = build_default_workflow_runner()
result = runner.run()
```

Configuration files can build the same workflow:

```python
from acagent.runner import build_workflow_runner_from_config

runner = build_workflow_runner_from_config("acagent/configs/default.yaml")
result = runner.run()
```

Run with UV from the repository root:

```bash
PYTHONPATH=acagent/src uv run python -c 'from acagent.runner import build_workflow_runner_from_config; build_workflow_runner_from_config("acagent/configs/default.yaml").run()'
```

The default DeepSeek client reads `DEEPSEEK_API_KEY`. Local outputs are:

1. `acagent_outputs/default/memory.json`
2. `acagent_outputs/default/predictions.jsonl`
3. `acagent_outputs/default/traces.jsonl`

If a run needs to resume from a known-good trace boundary, see
[`RUN_RECOVERY.md`](RUN_RECOVERY.md).

The implemented workflow is ready for small real-model runs, subject to API key
and cost limits. Current tests:

```text
55 passed
```
