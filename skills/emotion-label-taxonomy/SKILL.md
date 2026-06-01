---
name: emotion-label-taxonomy
description: Use when designing, revising, or critiquing an emotion label set for dialogue emotion recognition, especially deciding whether labels should be retained as primary emotions, merged into broader labels, represented as mixed emotions, or moved to qualifiers.
---

# Emotion Label Taxonomy

Use this skill when the task is to decide which emotion labels belong in a dialogue emotion recognition schema.

## Core Principle

Do not treat a theoretical emotion list as authoritative by itself. A label should be kept as a primary label only if it has a useful, non-redundant role in the target dataset and annotation task.

For long-form dialogue and screenplay emotion recognition, prioritize **speaker's real emotion** over surface wording, listener perception, or plot function.

## Retention Test

For each candidate label, evaluate it with these questions:

1. **Intensity test**: Is this merely a stronger, weaker, calmer, or more aroused form of another label?
   - If yes, prefer a broader primary label plus `intensity` or `qualifier`.
   - Example: low-arousal happiness may be `contentment`, but do not merge it automatically if it has a distinct appraisal structure.

2. **Mixture test**: Is this mostly a stable combination of other emotions?
   - If yes, consider representing it with `primary_emotion` plus `secondary_emotions` or `emotion_components`.
   - Do not blindly enforce Plutchik dyad formulas; treat them as heuristics, not authority.

3. **Appraisal test**: Does this label have an independent appraisal structure?
   - Keep labels that depend on a distinct trigger/evaluation pattern, not just different wording.
   - Examples: `guilt` involves responsibility for wrongdoing; `relief` involves a feared bad outcome being avoided.

4. **Substitution test**: Are there realistic scenes where this label is clearly the best single label, and all alternatives feel materially worse?
   - If yes, this supports retaining it as a primary label.
   - If no, move it to qualifier, secondary emotion, or drop it.

5. **Frequency and reliability test**: Will the label occur often enough and be distinguishable enough for annotators or LLM labelers?
   - Rare but theoretically neat labels should not become primary labels unless they matter to the research question.

## Recommended Output Categories

Use these categories when advising on a label:

- `keep_primary`: Retain as a primary emotion label.
- `merge_into`: Use another primary label, with optional intensity or subtype.
- `secondary_or_component`: Use only as a secondary emotion or mixture.
- `qualifier`: Use as a modifier/state rather than a primary label.
- `drop`: Do not include unless new evidence shows dataset need.

## Useful Distinctions

`happiness` vs `contentment`:
- `happiness` is general positive affect, pleasure, gladness, or joy.
- `contentment` has a distinct structure: need satisfied, stable comfort, no immediate unmet demand. It may deserve primary status when the dataset contains many relaxed satisfaction scenes.

`happiness` vs `relief`:
- `happiness` is positive evaluation or pleasure.
- `relief` requires a prior threat, worry, or anticipated bad outcome being removed. It is not just happiness with lower intensity.

`sadness` vs `distress`:
- `sadness` centers on loss, disappointment, rejection, grief, or relational hurt.
- `distress` should be narrow if used: acute suffering or pain-related emotional upset. Pure physical pain is a sensation, not necessarily an emotion.

`fear` vs `distress`:
- `fear` centers on threat prediction, danger, escape, defense, or possible harm.
- Do not use `distress` for ordinary pressure, being trapped, or panic if `fear` explains the state better.

`shame` vs `guilt`:
- `guilt`: "I did something bad"; behavior-focused responsibility.
- `shame`: "I am bad/exposed/unworthy"; self-image damage, often close to self-disgust plus fear of social devaluation.

`shame` vs `embarrassment`:
- `shame` is deeper self-evaluative pain.
- `embarrassment` is milder social exposure, awkwardness, or public misstep, common in sitcom dialogue.

`contempt` vs `anger` vs `disgust`:
- `anger`: you harmed, blocked, or wronged me; I want correction or confrontation.
- `disgust`: this is repulsive or unacceptable; I want distance.
- `contempt`: I look down on you; hostility plus devaluation. It may be approximated by anger/disgust, but can be primary if devaluation matters.

`interest` vs `surprise`:
- `surprise` is a short reaction to expectation violation.
- `interest` is sustained attention, curiosity, engagement, or exploration motive.

## Working Procedure

When asked to revise a label set:

1. List the candidate labels.
2. Normalize typos and naming, e.g. `contemp` -> `contempt`.
3. For each label, apply the retention test.
4. Identify likely confusions and write boundary rules.
5. Recommend a primary label set plus optional fields such as:

```text
primary_emotion
secondary_emotions
intensity
qualifiers
physical_pain
reason
```

6. Keep theory references in a supporting role. Use Ekman, Izard, Plutchik, or other theories as sources of hypotheses, not as final authority over annotation design.
