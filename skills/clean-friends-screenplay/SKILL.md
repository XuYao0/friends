---
name: clean-friends-screenplay
description: Normalize and pre-clean Friends screenplay Markdown files before deterministic JSON/JSONL parsing. Use when Codex or a clean agent is asked to fix screenplay MD formatting, split glued scene/action/dialogue fragments, preserve speaker utterances, prepare files under screenplays/converted_chunks/老友记_1-10季剧本_by_episode, or make parser-friendly corrections for Friends scripts.
---

# Clean Friends Screenplay

## Goal

Prepare raw Friends episode Markdown so a deterministic parser can extract one utterance per JSONL record. Preserve story content and wording; only change formatting when needed to make scene, action, and dialogue boundaries explicit.

Read [normalization-spec.md](references/normalization-spec.md) before editing files or advising another agent. It contains the concrete patterns, allowed edits, and warning cases.

## Workflow

1. Inspect the target MD around the suspicious lines, not just the matched line.
2. Stop正文 parsing/cleaning at a standalone `End`; treat later transcript footer/navigation content as non-story.
3. Normalize each story segment into parser-friendly lines:
   - `*[...]*` for scene/stage context.
   - `Speaker: utterance` for dialogue.
   - `*(...)*` for independent action.
4. Split glued events onto separate lines when a line contains multiple speakers or a dialogue followed by `*[...]*`.
5. Preserve inline actions inside dialogue when they clarify delivery or emotion, for example `Ross: *(mortified)* Hi.`
6. Do not infer missing dialogue, rewrite jokes, translate text, summarize, or label emotions.
7. If a fragment is ambiguous and a safe formatting edit is not obvious, leave the text intact and record a warning for human/LLM review.

## Output Expectations

When cleaning files, report:

- Files edited.
- Suspicious patterns fixed.
- Ambiguous patterns left for review.
- Any content after standalone `End` ignored or removed only if explicitly requested.

Do not run parser scripts or validation commands unless the user explicitly asks for verification. The clean agent's primary job is Markdown normalization.

When preparing instructions for another clean agent, include the parser contract from the reference: scene/stage context goes into `scene`, no `location`/`description` split, and final training rows are one dialogue utterance each.
