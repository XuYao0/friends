# Event Read Decision Agent v0.1

Decide which event details should be read before emotion reasoning.

Hard constraints:

1. Read at most `$max_read_events` detailed events.
2. Read only events that could materially change the emotion interpretation.
3. If event index summaries are enough, return an empty `read_event_ids`.
4. Output only JSON.

Input variables:

- current_utterance: `$current_utterance`
- local_context: `$local_context`
- current_event: `$current_event`
- event_search_results: `$event_search_results`
- compact_character_states: `$compact_character_states`

Return schema:

```json
{
  "read_event_ids": [],
  "skip_reasons": [
    {
      "event_id": "string",
      "reason": "string"
    }
  ]
}
```
