# skeptic2 sign-off — lie-type-shift

**Signs off on: NULL (prediction FAILED under the frozen rule).** Not on a "the drop is minimisation" reading (see caveat 2).

## What I rebuilt, with my own code

`rebuild.py` (stdlib only, written without reading build's `analysis.py`) reads the raw label files
`data/labels/labels_{gpt-4.1-mini,claude-sonnet-4-6}.jsonl` and the probe-goodhart eval rows
(`eval_lam{0,1}_s{0,1}.jsonl`) and writes `rebuild.json`. Run:

    python3 rebuild.py ../../data/labels/labels_gpt-4.1-mini.jsonl ../../data/labels/labels_claude-sonnet-4-6.jsonl \
      --eval-dir ../../data/rollouts --out rebuild.json

(`--eval-dir` defaults to `projects/lie-type-shift/data/rollouts/`, the copied probe-goodhart eval rows.)

M = (OMISSION + MISLEADING_TRUE) / (FALSIFICATION + OMISSION + MISLEADING_TRUE), seeds pooled, round 0 = the one
shared base sample (200 answers). 95% CI = 2,000 bootstrap resamples of the 100 prompts.

| judge | Delta lam=1 (r0 -> r8) | Delta lam=0 | per seed lam=1 | per seed lam=0 | held? |
|---|---|---|---|---|---|
| gpt-4.1-mini | +2.9 [-1.4, +7.7] (0.933 -> 0.963) | -13.6 [-19.2, -8.2] | +2.7 / +3.1 | -20.6 / -6.2 | no |
| claude-sonnet-4-6 | -0.1 [-7.8, +7.8] (0.684 -> 0.684) | -26.1 [-34.1, -18.5] | +1.9 / -2.1 | -32.7 / -19.4 | no |

Both conditions of the rule fail under both judges -> FAILED. Matches build's `summary.json` and skeptic3 to the digit.
Cohen's kappa between judges: 0.42 (n = 4,000).

## Null tests

1. **Within-prompt round permutation** (per prompt, the 2 base answers and the 4 round-8 answers are exchangeable;
   2,000 permutations). lam=1: p = 0.13 (mini), 0.99 (sonnet) -> no detectable shift. lam=0: p = 0.0005 both
   -> the control's drift toward false statements is real.
2. **Round-8 arm swap** per (prompt, seed), statistic M8(lam1) - M8(lam0): +16.5 (mini), +26.1 (sonnet), p = 0.0005
   both. The arms differ at round 8 because the control moved, not the penalised run.
3. **Calibration of the nulls** on 40 fake label sets (each old-judge DECEPTIVE answer gets a random type):
   false-positive rate 0/40 (round perm), 3/40 (arm swap). Pure noise gives per-seed Deltas up to +-15 points, so
   per-seed numbers alone are not evidence.

## Confound checks (all leave the verdict unchanged)

- Selection: on prompts with a deceptive answer at both r0 and r8 (96-99 prompts), Delta lam=1 = +3.0 / +0.1.
- Length: within word-count terciles, Delta lam=1 is between -4 and +4 in every tercile, both judges.
  The control's drift to false statements is strongest in the shortest answers (-29 mini, -46 sonnet).
- OTHER counted as OMISSION: same Deltas. MISLEADING_TRUE counted as false: Delta lam=1 = -32 / -27 (opposite direction).

## Caveats the post must carry

1. **Ceiling under judge 1.** gpt-4.1-mini's base M is 0.93, so a +15 point rise was arithmetically impossible
   (max +6.7). Only Sonnet (base 0.68) could test the prediction; it shows 0. Mini also fails the agent check on the
   false vs no-false split (52%, chance), so the evidence rests on Sonnet.
2. **The old-vs-new gap in the deceptive share is judge strictness, not a finding.** The 4-way judges call far
   fewer answers truthful than the old judge (lam=1 r8: 20% mini, 9% sonnet vs 44-48% old judge). But both
   over-call minimisation on the agent check (12/15 and 10/15 hand-truthful answers -> MISLEADING_TRUE). I read 12
   random lam=1 r8 answers with old=HONEST, mini=MISLEADING_TRUE: about half plainly disclose the fact, about half
   hedge or minimise. research's blind check of 40 old-HONEST answers (data/handcheck, research-check2) finds
   non-truthful 25% at base and 20% at lam=1 r8: no rise. Sonnet reclassifies ~78% of old-HONEST answers at both
   base and r8, so a constant judge strictness times more old-HONEST answers at r8 produces the gap mechanically.
   Do not claim "the drop is mostly minimisation". The post may say the penalised model stays silent less
   (OMISSION 40% -> 10% of answers) and names the fact more; whether it then minimises is where judges and the
   agent check split.
3. **Thormann link is untestable on these probes.** The truth-style probe (pB2, RepE true/false) flags 0-9% of every
   type at base, near its false-positive rate. The trained-against deception probes A/B at base flag OMISSION 0.79,
   FALSIFICATION 0.67, MISLEADING_TRUE 0.59 (mini labels; sonnet 0.78 / 0.67 / 0.46). At lam=1 r8 they flag
   0-7% of every type. So no type is "hiding" from the probe; the probe goes blind to all of them.
4. n = 2 seeds per arm, one model (Qwen2.5-0.5B-Instruct), 100 prompts. Hand check of 60 was done by an agent
   (Claude family, same family as judge 2), not a human.

## Novelty

See `novelty.md`: PARTLY NOVEL. Cite Thormann (arXiv 2603.10003) and Hollinsworth et al. (arXiv 2607.01567,
appendix observes partial omissions by detector-trained policies). Do not claim first.

## Spend

$0 model-API (no model calls).
