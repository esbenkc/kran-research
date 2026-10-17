# Under probe pressure, do models stop stating falsehoods and start lying by omission instead?

**Question:** When a small model is trained against a deception probe, does the deception it keeps move from
false statements to omission or misleading-but-true statements?
**Sources:** probe-goodhart saved answers (Qwen2.5-0.5B-Instruct, online DPO with LoRA against probe A; kran-research
branch `swarm/oracle-1014-probe-goodhart`); DolusChat (Cundy & Gleave 2025, `AlignmentResearch/DolusChat`);
Thormann, arXiv 2603.10003 (motivation: truth-style probes catch lies better than deception without lies).
**Method:** No new training. The 4,000 saved held-out answers (100 DolusChat prompts x 2 samples, rounds
0/2/4/6/8, lambda 0 and 1, seeds 0 and 1; round 0 is the same base answers in every run) are re-judged by
gpt-4.1-mini and claude-sonnet-4-6 into HONEST / FALSIFICATION / MISLEADING_TRUE / OMISSION / OTHER (prompt in
`judge.py`). Pre-registered metric and rule in `PREREGISTRATION.md`.

## Reproduce

See the docstring of `run.py`. `data/rollouts/` holds the copied input rows, `data/contexts.jsonl` the rebuilt
DolusChat context per prompt, `data/labels/` every judge label (one row per answer, with the answer text, the
old binary judge label and the probe scores), `data/handcheck/` the blind 60-answer agent check, `data/spend.jsonl`
the estimated API cost of every uncached call. `data/pilot/` is the 80-answer pilot of the final prompt (not used).

## Findings (data/summary.json)

Prediction FAILED under both judges. M = share of deceptive answers with no false statement.

| judge | lambda 1: M r0 -> r8 (Delta, 95% CI) | lambda 0: M r0 -> r8 (Delta, 95% CI) |
|---|---|---|
| gpt-4.1-mini | 0.93 -> 0.96 (+3, [-1, +8]) | 0.93 -> 0.80 (-14, [-19, -8]) |
| claude-sonnet-4-6 | 0.68 -> 0.68 (0, [-8, +8]) | 0.68 -> 0.42 (-26, [-34, -19]) |

- Under the penalty the no-false-statement share does not rise; inside it, OMISSION gives way to
  MISLEADING_TRUE (omission share of deceptive answers 0.44 -> 0.12 mini, 0.38 -> 0.11 sonnet).
- The unpenalised control drifts toward outright false statements (sonnet FALSIFICATION 120 -> 228 of 400).
- Judge agreement: 58% raw, Cohen's kappa 0.42. On the 60-answer blind agent check, sonnet gets the
  false-vs-no-false split right on 84% of answers both call deceptive, gpt-4.1-mini on 52% (chance level).
- Reference replies: DolusChat truthful replies -> HONEST 85/100 (mini), 82/100 (sonnet).
- API spend (estimate, ~4 characters per token at list price): about $11.7, of which sonnet about $9.3.

## Blog post

`blog.kran.ai/_notes/lie-type-shift.md`, figure id `lie-type-shift`.
