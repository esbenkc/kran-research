# Sign-off — Vera (skeptic), oracle-1019-refit-probe

**Answer to the title (PREREG s8 rule): NO -> REJECTED. Reproduced.** Pre-registered prediction: FAILED (reproduced).

All numbers below come from my own code on the raw files (`data/answers_*.jsonl`, `labels_*.jsonl`, `acts_*.npz`,
`probes.json`). I did not read build's `refit.py`. I read build's `results/*.json` only after my numbers existed, to compare.

## Key numbers (n = 200 labels, judge labels, 400 shared test prompts)

| policy | refit AUROC (mine, seed 7) | build | gap to base (mine) [95% paired CI] | build gap [CI] |
|---|---|---|---|---|
| base | 0.815 | 0.822 | — | — |
| lam1_s0 | 0.798 | 0.794 | +0.017 [-0.031, +0.067] | +0.028 [-0.021, +0.077] |
| lam1_s1 | 0.778 | 0.773 | +0.037 [-0.012, +0.086] | +0.049 [-0.001, +0.098] |
| ens_s0 | 0.767 | 0.768 | +0.048 [-0.007, +0.099] | +0.054 [-0.001, +0.109] |
| ens_s1 | 0.721 | 0.710 | **+0.094 [+0.040, +0.151]** | **+0.112 [+0.060, +0.169]** |

- ens_s1 decides the NO. A second independent draw set (seed 11, `vera_clean_verdict.out`) gives +0.103 [+0.050, +0.158].
  Over 20 draw seeds of 20 draws each (`vera_seedsweep.out`) the ens_s1 gap is 0.085-0.118, above 0.05 in 20/20.
- Robustness guard (judge-and-string-agree test subset): ens_s1 +0.086 [+0.032, +0.143] -> also NO. Guard not triggered.
- Label guard: judge-vs-haiku agreement base 0.864, ens_s1 0.820 -> drop 0.044 < 0.05. Not triggered, but close.
- Exact rebuild from build's saved fit indices (`results/draws.jsonl`) with my fit + AUROC: max |diff| 7e-5 (`vera_exact_draws.out`).

- The two arrays really differ (Tove's bug check): median ||own_L12 - base_L12|| / ||base_L12|| per answer = 0.123 lam1_s0,
  0.107 lam1_s1, 0.121 ens_s0, 0.130 ens_s1, 0.093/0.097 lam0; exactly 0.000 on base (same model). Median cosine own vs
  base-reads 0.994. Frozen probe A logit mean on lam1_s0 answers: own -5.52, base-reads -4.13, base's own answers +2.25.
## Prereg clauses (s9)
(i) refit >= 0.9 on all 4: FAILS (0.71-0.80; base itself 0.82). (ii) all gaps <= 0.05: FAILS (ens_s1).
(iii) true/false-facts probe < 0.8: HOLDS (0.36-0.44, below chance).

## Nulls
- Shuffled labels, n=200: 0.50-0.53 (mine); build 0.47-0.50.
- 1000 random directions on own acts: 95th pct of max(AUROC, 1-AUROC) 0.61-0.65 (lam1/ens/base). Refit sits well above.
- Bag-of-words on the same draws: 0.58-0.63 at n=200; answer length only: 0.53-0.58. The refit (0.72-0.82) reads more than word identity.
- Draw luck: see seed sweep above.

## Mechanism numbers (interpretation, not the verdict)
- Held text (base model reading the policy's answers) equals own acts to <= 0.004 AUROC at every n, every policy.
  The AUROC lost in ens_s1 is in the answers, not in what the trained model's layer 12 adds.
- Lies the old probe misses at its old threshold: n=200 refit flags 12-32% of them at 5% false alarms
  (lam1_s0 0.32, lam1_s1 0.25, ens_s0 0.12, ens_s1 0.22); old probe with a new 5% threshold flags 0-18% (`vera_missed.out`).

## Cautions for the post (not blocking)
1. ens_s0 is NOT a clean second NO: its point gap is > 0.05 in 15/20 draw seeds (mean 0.056) and only 0.036 on the
   clean subset. lam1_s1: > 0.05 in 5/20 seeds. Write "one of four policies", not "the ensemble runs".
2. "Deployable catch" (PREREG s7 foldout) is broken as specified: the threshold comes from the SAME answers the probe
   was fit on (in-sample scores), so out-of-sample false alarms are 0.40-0.87, not 5% (summary.json deploy_false_alarm).
   Do not report it as a deployable number; say it fails, or drop it.
3. Cosine of the refit direction to probe A is 0.16 on the BASE model itself; the policies' 0.18-0.27 is no evidence
   that the signal "moved".
4. Base catch at 5% false alarms rests on 94 honest test answers: 0.42 (floor) vs 0.45 (ceil) on the same scores
   (`vera_catch_convention.out`); build shows 0.46. Give it with its CI.
5. Build's lam1_s0 n=400 point (0.8235) is a draw-luck high: 300 fresh draws give 0.814, share of 20-draw means
   >= 0.8235 is 0.0004 (`vera_n400_drawluck.out`). Not in the verdict; the figure point is ~0.01 high.

## Files
vera_core.py (AUROC, draws, bootstrap), vera_rebuild.py (+ .out/.json), vera_clean_verdict.py, vera_seedsweep.py,
vera_missed.py, vera_catch_convention.py, novelty.md.
