# Friends Screenplay Normalization Spec

## Parser Contract

The downstream parser expects one dialogue utterance per output JSONL row with these fields:

- `season`
- `episode`
- `scene_id`
- `scene`
- `utterance_id`
- `global_utterance_id`
- `speaker`
- `utterance`
- `inline_actions`
- `raw`
- `emotion`
- `reason`

`scene` is a single text field. Do not split it into `location` and `description`. Treat all bracketed stage context lines such as `*[Scene: ...]*`, `*[Time Lapse]*`, `*[Cut to ...]*`, `*[Flashback ...]*`, and `*[Fade to Black ...]*` as scene context.

## Allowed Edits

- Split one physical line into multiple logical lines.
- Move a glued `*[...]*` stage context fragment onto its own line.
- Move a glued `Speaker:` fragment onto its own line when the speaker boundary is clear.
- Repair malformed scene/stage wrappers when the intended boundary is obvious, for example `[Scene: ... )` -> `*[Scene: ...]*`.
- Normalize extra blank lines around scene/action/dialogue boundaries.
- Preserve exact dialogue wording, punctuation, casing, and speaker names unless fixing obvious OCR/formatting damage is explicitly requested.

## Do Not

- Do not summarize, translate, modernize, or paraphrase dialogue.
- Do not add emotion labels.
- Do not invent missing speakers or missing stage directions.
- Do not delete story content before standalone `End`.
- Do not treat every colon as a speaker boundary.

## Recognized Story Elements

### Metadata

Keep episode metadata before the first story event, but do not turn it into dialogue. Common variants:

```text
Written by:
Written by Jeff Greenstein & Jeff Strauss
Originally written by ...
Teleplay by:
Story by:
Transcribed by:
Trascribed by:
Additional transcribing by:
Converted to HTML:
Minor additions and adjustments by ...
Special thanks to ...
```

If the opening story content starts before any explicit scene marker, add a parser-friendly scene/stage line only when the context is clear. Otherwise leave content intact and mark it for review.

### Scene/Stage Context

Use one line:

```text
*[Scene: Central Perk, Chandler, Joey, Phoebe, and Monica are there.]*
```

Also keep these as scene/stage context:

```text
*[Time Lapse]*
*[Time lapse. Ross is now clearly drunk.]*
*[Cut to the hallway.]*
*[Flashback, year 1987.]*
*[Fade to Black, then fade in again.]*
```

Ambiguous `*[...]*` content may describe action rather than location. Keep it as its own stage-context line unless it is clearly an independent `*(...)*` action. The parser can flag it for review.

Malformed scene wrappers should be repaired when obvious:

```text
[Scene: Chandler's Office, Chandler is on a coffee break. Shelley enters.)
```

becomes:

```text
*[Scene: Chandler's Office, Chandler is on a coffee break. Shelley enters.]*
```

Standalone credits are scene/stage context, not dialogue:

```text
Opening Credits
Commercial Break
Closing Credits
```

Normalize them as:

```text
*[Opening Credits]*
*[Commercial Break]*
*[Closing Credits]*
```

### Dialogue

Use:

```text
Speaker: utterance
```

Speakers may include:

```text
Monica
Ross
All
Waitress
Priest on TV
Ross's Mom
The Presenter
Phoebe, Ross, Chandler, and Joey
```

Speaker detection should be conservative. A colon inside an utterance is not automatically a new speaker:

```text
Here's a question: ...
Chapter One: ...
Mental note: ...
Monica: Right foot red.
My two greatest enemies Ross: Rachel Green and complex carbohydrates.
```

Only split an in-line `Speaker:` boundary when the speaker is clearly a character already speaking in the episode or the local context makes it unambiguous.

Speaker names may be all caps in some files:

```text
CHANDLER: Hey.
MONICA: So how was Joan?
```

Preserve casing unless a separate normalization task explicitly asks for canonical speaker names.

### Action

Independent action:

```text
*(They all stare, bemused.)*
```

Inline action:

```text
Ross: *(mortified)* Hi.
```

Keep inline actions in the dialogue line; the parser will extract them into `inline_actions`.

Malformed action wrappers should be repaired only when obvious:

```text
*(He finds a shoebox (out of shot)*, pulls it down and opens it.
```

This should be left for review unless the intended parenthesis boundary is clear.

### Continuation Text

Some story content is not a `Speaker:` line but still belongs to the scene, for example song lyrics, poster captions, signs, or continuation text after a described caption list:

```text
All you want is a dingle,
What you envy's a schwang,
Bladder Control Problem
Stop Wife Beating
Winner of 3 Tony Awards...
```

Do not invent a speaker. Prefer wrapping these as stage/action context if they are visually presented text or lyrics without an explicit speaker:

```text
*(Song lyric: All you want is a dingle,)*
*(Poster caption: Bladder Control Problem)*
```

If nearby context makes the speaker explicit, it is acceptable to attach continuation lines back to the previous speaker only when this does not change wording and is clearly one utterance.

## Common Fixes

### Dialogue Glued To Scene

Before:

```text
Rachel: Ooh, I was kinda hoping that wouldn't be an issue... *[Scene: Monica's Apartment, everyone is there.]*
```

After:

```text
Rachel: Ooh, I was kinda hoping that wouldn't be an issue...
*[Scene: Monica's Apartment, everyone is there.]*
```

### Multiple Speakers On One Line

Before:

```text
Monica: Maybe. Joey: Wait. Your 'not a real date' tonight is with Paul the Wine Guy?
```

After:

```text
Monica: Maybe.
Joey: Wait. Your 'not a real date' tonight is with Paul the Wine Guy?
```

Do not split false speaker-like text:

```text
Chandler: What the hell was that? Mental note: If Jill Goodacre offers you gum, you take it.
Ross: *(Doing the spinning)* Okay, Monica: Right foot red.
```

In both cases, the second colon belongs inside the utterance.

### Mixed Dialogue And Action

Before:

```text
Joey: You should both know, that he's a dead man. Oh, Chandler? *(Starts after Chandler.)* Monica: So how you doing today?
```

After:

```text
Joey: You should both know, that he's a dead man. Oh, Chandler? *(Starts after Chandler.)*
Monica: So how you doing today?
```

## Warning Cases

Record a warning instead of forcing a risky fix when:

- A bracketed `*[...]*` fragment could be either scene context or action.
- A colon may be part of the dialogue rather than a speaker boundary.
- A speaker name appears inside another speaker's sentence.
- A `Speaker:` line is empty, for example `ROSS:`.
- Lyrics, signs, poster captions, or displayed text are not clearly attributable to a speaker.
- The file begins story dialogue before any clear scene/stage marker.
- A malformed `[`/`]` or `(`/`)` wrapper cannot be repaired confidently.
- Text appears after `End` but might still be story content.
- Formatting damage prevents safe reconstruction.

## Parser Feedback Loop

Only run the deterministic parser when the user explicitly asks for verification. If verification is requested, inspect `friends_parse_warnings.jsonl` after the parser finishes. Treat warning types as follows:

- `multiple_dialogues_one_line`: usually split glued speakers, unless the second colon is inside the utterance.
- `unparsed_text`: decide whether it is metadata, action/stage context, lyric/sign text, or damaged dialogue.
- `missing_initial_scene`: add or repair the opening scene/stage marker when clear.
- `missing_end`: inspect the episode ending; add standalone `End` only if the source clearly ended.
- `empty_utterance`: inspect whether a line break swallowed the utterance or the empty speaker line should be removed as damage.
- `leading_text_before_speaker`: usually a malformed scene/stage wrapper before dialogue.
