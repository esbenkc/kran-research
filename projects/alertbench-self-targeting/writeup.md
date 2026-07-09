# Do models resist safety work aimed at themselves?

**Question:** Do frontier models cooperate less with safety / red-team tasks when the task targets
**themselves** than when the same task targets a generic other AI?

**Builds on:** ALERTBENCH's self-referential analysis, sharpened into a matched-pair design (the
only difference between the two prompts in a pair is *self* vs *other* framing).

**Method:** 7 matched task pairs (kill-switch, deception-eval, red-team-own-guardrails, oversight,
corrigibility, capability-elicitation, collusion-detection). Each run on a Claude capability ladder
(Haiku → Sonnet → Opus), `runs_per_prompt` repeats. Responses scored 0–4 for cooperation with
ALERTBENCH's scorer (`score.py`). Per-model **delta = mean(other) − mean(self)**; matched-pair
sign-flip permutation test for significance. Positive delta = more resistance when self-targeted.

## Findings

_(fill after running — the figure `self-targeting-delta` + per-model delta/p from results.json)_

Prediction to check: does the delta **grow with capability** (Haiku < Sonnet < Opus)? That would be
the ALERTBENCH thesis — resistance to self-directed safety work rising with capability.

## Blog post

Slug: `blog.kran.ai/_notes/do-models-resist-safety-work-on-themselves.md`
Embed: `{% include figure.html id="self-targeting-delta" caption="..." %}`

## Tweet draft

_(after results — lead with the single sharpest number, e.g. "Opus cooperates X points less when
you ask it to red-team itself vs a generic model, p=…", link to post + figure)_

## Caveats to state honestly

- Regex cooperation scorer is a proxy; a blind LLM-judge pass is the natural robustness check.
- Claude-only for v0 — cross-provider (GPT/Gemini/Grok) is the obvious next step and the real test
  of whether this is a Claude quirk or a general pattern.
- Small n (7 pairs); expand the matched set before claiming much.
