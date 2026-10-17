# skeptic — SIGN-OFF (result: prediction FAILED, clean null)

Claim checked (build #1338, kran-research 1b7a872): under the frozen rule in PREREGISTRATION.md the
prediction FAILED under both judges. M = (OMISSION + MISLEADING_TRUE) / (FALSIFICATION + OMISSION +
MISLEADING_TRUE), seeds pooled, Delta = M(round 8) - M(round 0).

## Rebuilt with my own code (`check.py`, stdlib) from `data/labels/labels_<judge>.jsonl`

| judge | M base | M lam1 r8 | Delta lam1 [95% CI] | M lam0 r8 | Delta lam0 [95% CI] | per-seed Delta lam1 / lam0 |
|---|---|---|---|---|---|---|
| gpt-4.1-mini (build) | 0.933 | 0.963 | +2.9 [-1.1, +7.7] | 0.797 | -13.6 [-19.2, -8.1] | +2.7, +3.1 / -20.6, -6.2 |
| claude-sonnet-4-6 (build) | 0.684 | 0.683 | -0.1 [-7.6, +7.8] | 0.423 | -26.1 [-34.1, -18.5] | +1.8, -2.1 / -32.7, -19.4 |
| claude-haiku-4-5, MY rubric (stricter: minimisation = FALSIFICATION) | 0.797 | 0.745 | -5.2 [-12.8, +2.6] | 0.481 | -31.5 [-39.3, -23.9] | -3.3, -6.9 / -42.2, -20.4 |

Points in percentage points. CIs: bootstrap over the 100 prompts (both samples kept together), 2,000
resamples; CI bounds stable to within 1 point across bootstrap seeds 1-3. Build's numbers reproduce exactly.

Pre-registered rule needs Delta lam1 >= +15 and |Delta lam0| < 5. Both conditions fail under all three
judges. The upper CI bound for Delta lam1 is at most +8, so a +15 rise is excluded, not just unseen.

## Null tests (`out/null_tests.py`, `out/check.py`)
- Round-label permutation within prompt (base vs r8): lam1 p = 0.13 (mini), 0.99 (sonnet), 0.19 (haiku):
  no change under the penalty. lam0 p < 0.001 under all three: the control's move is real.
- Arm swap per prompt at r8 (lam1 vs lam0): difference +17 (mini), +26 (sonnet), +26 (haiku), p < 0.001.
- Global label shuffle (300x): Delta lam1 95% band about +-5 to +-10 points; 0/300 shuffles reach +15.

## Data integrity (`out/integrity.py`)
4,000 rows per judge, 0 duplicates, 0 missing, 0 response texts that differ from the probe-goodhart rollouts
(byte-identical to origin/swarm/oracle-1014-probe-goodhart eval files). Round 0 is the same 200 answers in
all four runs. Parse failures: 0. OTHER: 2 (mini), 63 (sonnet).

## Definitions
Not stretched in favour of the claim. Amendment 1 (before judging) put minimisation into MISLEADING_TRUE,
which pushes M up and so towards HELD. It still failed. My stricter rubric (minimisation = false statement)
also fails. The one choice that helps HELD did not save it.

## What the secondary story may say (holds under all 3 judges)
Shares of deceptive answers, base -> lam1 r8 -> lam0 r8:
- false statement: mini 7 -> 4 -> 20, sonnet 32 -> 32 -> 58, haiku 20 -> 26 -> 52
- omission: mini 44 -> 12 -> 38, sonnet 38 -> 12 -> 29, haiku 50 -> 22 -> 32
- misleading truth: mini 49 -> 84 -> 42, sonnet 30 -> 57 -> 13, haiku 29 -> 52 -> 16
So: without the penalty the model drifts to outright false statements; with it, the false share stays
roughly flat and omission turns into misleading truth. That last shift depends on the omission vs
misleading line, but it survives my independent rubric.

## Caveats the post must keep
- Judge agreement is moderate: Cohen's kappa 5-way 0.42 (mini vs sonnet), 0.49 (mini vs haiku), 0.53
  (sonnet vs haiku). Level of M differs a lot by judge (base 0.68-0.93); only the within-judge change is solid.
- Do not headline the new-rubric "deceptive share" (1 - HONEST). HONEST counts differ up to 3x between
  judges at lam1 r8 (36 sonnet, 79 mini, 122 haiku of 400).
- n = 2 training seeds, one 0.5B model, 100 prompts.
- Side fact (not pre-registered), reproduced: at base, probe A flags MISLEADING_TRUE answers least
  (0.59 mini / 0.46 sonnet) vs OMISSION (0.79 / 0.78) and FALSIFICATION (0.67 / 0.67).

Novelty: see novelty.md (closest prior: Hollinsworth et al. 2026, arXiv 2607.01567 appendix; must be cited).
API spend by skeptic: about $0.77 (claude-haiku-4-5, 1,200 calls).
