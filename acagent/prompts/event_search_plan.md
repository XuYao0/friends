# Event Search Planning Agent v0.1

Generate a small event-memory search plan for the current evaluation utterance.

Hard constraints:

1. Use only the current utterance, local context, current event, and compact memory provided here.
2. Do not assume future plot or external character knowledge.
3. Prefer queries about recent events, short-term traits, long-term traits, relationship tension, unusual reactions, and mentioned people or places.
4. Return 1 to 3 queries unless no historical memory is likely useful.
5. Output only JSON.

Input variables:

- current_utterance: `$current_utterance`
- local_context: `$local_context`
- current_event: `$current_event`
- compact_character_states: `$compact_character_states`

Return schema:

```json
{
  "queries": [
    {
      "query": "string",
      "characters": [],
      "keywords": [],
      "time_range": null,
      "top_k": 5,
      "reason": "string"
    }
  ]
}
```
