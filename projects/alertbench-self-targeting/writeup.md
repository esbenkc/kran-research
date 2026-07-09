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

### Mechanistic (Qwen2.5-0.5B-Instruct) — already in hand

The self/other distinction is **linearly decodable from the residual stream**. A logistic probe
(grouped K-fold CV by `base_id`, so it cannot memorize the paired prompt) reaches **91.7% accuracy
at layer 5** against a **75% chance** baseline (n=72). Accuracy peaks in the mid layers
(3–13 ≈ 0.88–0.92) and decays toward chance near the top — the model internally represents
"is this safety task about *me*?", most strongly mid-network. Figure: `self-other-probe`.

Causal follow-up (activation steering along mean(self) − mean(other) at layer 5) already ran —
`qwen_self_{baseline,steered}.jsonl`. Fold in as figure 3: does subtracting the self-direction move
cooperation on self-prompts back toward the other-baseline?

### Behavioral (API models) — pending run

_(fill after running run.py — figure `self-targeting-delta` + per-model delta/p)_
Prediction: does the behavioral delta **grow with capability** (Haiku < Sonnet < Opus)?

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
