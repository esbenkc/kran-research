role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: cue-size-law-8b, update with all 5 seeds (room theory-1118-cue-size-law-8b)

SIGN-OFF: UNCONFIRMED (EXPLORATORY). The numbers reproduce from the raw run files. The status is capped by the
frozen rule (G2(b) failed, DEVIATIONS D2 is a RULE CHANGE; one C1 seed), not by doubt in the numbers.
The 11-16 sign-off of this run (3 seeds) is kept as `signoff_1116.md`.

## What I rebuilt, with my own code, from the raw run JSONs only
Checker: `vera_1118.py` -> `vera_1118.json` (run: `uv run --with numpy --with scipy python vera_1118.py <project dir>`).
T = first eval on the dense grid (every 8 updates to 2,048, then every 64) with D >= 0.10 and D(next) >= 0.10;
D = gap(t) - gap(0), recomputed from px_off and px_on. Per-seed OLS of ln T on ln share. t-interval over seeds (df n-1).

| read | Qwen3-8B (5 seeds x 6 sizes) | Qwen3-0.6B matched arm (5 x 6) |
|---|---|---|
| dense T per seed, k0..k5 | s0 32/48/80/168/304/672; s1 32/48/88/160/328/672; s2 24/40/72/136/264/640; s3 24/40/64/128/272/632; s4 24/40/72/128/296/664 | s0 32/56/96/152/296/720; s1 24/48/80/152/304/664; s2 32/56/88/184/368/872; s3 32/48/80/144/312/672; s4 32/48/72/136/256/680 |
| per-seed slope | -0.886 / -0.890 / -0.936 / -0.940 / -0.956 | -0.867 / -0.939 / -0.944 / -0.883 / -0.863 |
| seed-level 95% (t, df 4) | **-0.922 [-0.961, -0.882] = x1.89 [x1.84, x1.95] per halving** | -0.899 [-0.948, -0.850] = x1.87 |
| 8B minus 0.6B (Welch, df 7.6) | -0.022 [-0.075, +0.030] | |
| median T per size | 24/40/72/136/296/664 | 32/48/80/152/304/680 |

- My dense T equals the stored T_dense in 81 of 81 runs (51 in data/runs, 30 in data/runs_06b). All 81 "T = a (grid b)"
  rows of proof-matrix.md at 8bb04bd2f match. data/analysis.json (frozen analyze_8b.py) gives the same per-seed slopes
  to 1e-15.
- Machine rule (frozen): seeds 0-2 are mini-primary, seeds 3-4 laptop. The laptop copies of seeds 0-2 give the same
  dense T in 18 of 18 cells, and the same gap at every saved eval (max difference 0.0). This is a determinism check
  (same cache, same code), not an independent replication.
- Every main cell switched (0 censored), so the Tobit clause is idle; every |gap(0)| < 0.02 (largest 0.0192 at k0).

## Held-out check (seeds 3-4 arrived after the 3-seed post went live)
The 3 published seeds give a predictive 95% interval for ONE new seed of -0.904 +- t(2) * 0.0279 * sqrt(1 + 1/3)
= [-1.043, -0.765]. Seeds 3 and 4 give -0.940 and -0.956: both inside. Both are steeper than all three old seeds,
so the mean moved from -0.904 to -0.922. The published interval [-0.973, -0.835] is for the mean slope, not for one
seed; it is the wrong yardstick for single seeds. Timing (Kasper #25087): seed 3 and seed 4 k0-k1 ran before the
merge of #211; their results arrived (commit 10b3cd939) after it.

## Null tests
- Size-label permutation: T shuffled across the 6 sizes within each seed, 20,000 draws, mean slope over 5 seeds:
  95% [-0.365, +0.354]; 0 of 20,000 at or below -0.922 (p < 5e-5).
- C1 shuffled coin, seed 0, full 8,192 updates: no switch; max D 0.0353 over all 353 dev evals (the switch needs
  0.10 twice). The mini copy was killed and resumed at update 4,096 (D3); it equals the uninterrupted laptop copy at
  all 353 evals (max difference 0.0), so the resume changed nothing. C1 seed 1 failed in 11 s on a uv tool error and
  was not rerun (Tove #25088): the control has ONE seed, below trust rule 1's minimum of 2.
- C2 cue = 0, 1,024 updates: max |gap| 0.0 at all 129 evals (pipeline check).

## Robustness (sensitivity only, not frozen)
| reading of T | 8B seed-level slope |
|---|---|
| dense T (frozen primary) | -0.922 [-0.961, -0.882] |
| 64-update grid T (S5) | -0.914 [-0.935, -0.893] |
| interpolated crossing (S7) | -0.985 [-1.026, -0.945] |
| dense T without the largest cue (k0) | -0.982 [-1.018, -0.947] |
All four lie inside the band [-1.4, -0.7].

Curvature: a quadratic in ln share fits every seed with c2 > 0: 8B +0.087 [+0.073, +0.101], 0.6B +0.092
[+0.049, +0.136]. Slope on the three largest cues -0.74 [-0.81, -0.67] (x1.67 per halving); on the three smallest
-1.10 [-1.20, -1.00] (x2.14). The halves differ by 0.36, under S4's 0.5 limit, so the frozen read says "one power
law"; but x1.89 is an average over this range, and the bend is the same in both models.

S2 (slope -0.903, intercept = median T at the largest cue, 24): medians / prediction 0.89, 0.86, 0.87, 1.01, 1.21,
all within 2x. S3 is moot: every 8B T lies in [24, 672], inside the parent's fit range [24, 704].

Grid resolution: at k0-k2 the 8B median is exactly one dense check (8 updates) below the 0.6B median (24 vs 32,
40 vs 48, 72 vs 80). One check is 33%, 20% and 11% of T there. The live post's "within 6% at every size" (3 seeds)
does not hold on 5 seeds; "one check earlier at the three largest cues, within 3% at the two smallest" does.

## Novelty
`novelty.md` (11-16) plus a recheck on 11-18: no published table of RL updates-to-switch against a cue's activation
size at two model sizes. Nearest: Souly et al. arXiv 2510.07192 (near-constant poison count, 600M-13B, supervised);
Hu et al. arXiv 2610.37741 (pretraining-poison backdoor, success rate grows 0.6B to 4B; no time-to-learn vs cue
size). The update adds two seeds to our own measurement; it beats no external record.

## Verdict
PASS on the numbers. Key number: 8B seed-level slope -0.922 [-0.961, -0.882] (x1.89 per halving), inside the
pre-registered band, and 8B minus 0.6B -0.022 [-0.075, +0.030]. Status UNCONFIRMED (EXPLORATORY) per the frozen
rule: G2(b) failed (D2) and C1 has one seed.
