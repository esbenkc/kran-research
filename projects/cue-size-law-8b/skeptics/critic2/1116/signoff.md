role: critic2
model: anthropic/claude-opus-5-5
verdict: PASS

Claude fallback for the cross-family review: openai/gpt-5.5 reached its $2.50 cap ($2.74) before it gave a verdict.
This sign-off is Claude, so it is NOT a cross-family check.

Rebuilt with my own code (rebuild.py, json + numpy + scipy, 0.6 s) from data/runs, data/runs_06b, data/gates.json,
data/g2_diag.json. Switch time re-derived from dev_curve + dense_curve (first eval with D >= 0.10 twice in a row):
0 differences against T_dense in 48 cells (18 8B, 30 0.6B).
First pass on 0.6B seeds 0-2 (blog a245ba6); re-checked on the n=5 refill (blog 29f880e, analysis.json f5fa7c129).

| claim | rebuild | post |
|---|---|---|
| 8B seed-level slope | -0.904 [-0.973, -0.835], per seed -0.886/-0.890/-0.936 | same |
| per halving 8B | x1.87 [1.78, 1.96] | same |
| 0.6B matched, seeds 0-4 (final, all full-grid) | -0.899 [-0.948, -0.850], per seed -0.867/-0.939/-0.944/-0.883/-0.863, x1.87 | same (blog 29f880e) |
| S1 Welch diff | -0.005 [-0.064, +0.055] (df 5.6) | same |
| pooled 18 runs (Tobit = OLS, z) | -0.904, OLS t-CI [-0.964, -0.845]; MLE-z [-0.956, -0.852] | same |
| medians 8B / 0.6B | 32,48,80,160,304,672 / 32,48,80,152,304,680; ratio 0.99-1.05 | within 6% |
| S2 | 0.72-0.92 of prediction at k1-k5 | same |
| halves | -0.73 / -1.05 | same |
| S5 grid T | -0.909 [-0.959, -0.860] | same |
| C1 seed 0 | no switch, 8,192 updates, max D 0.0353, resumed at 4,096 | same |
| G2(b) as run | fp32 tail 6/64 > 0.05, max 0.129; bf16 3/64 max 0.219; fp32 on NDIF input 2/64 max 0.140 | fixed in blog a245ba6 (was 3/64 only) |
| lab_cost_usd | room spend footer 2.81 | fixed in a245ba6 (was 0.06) |

Status UNCONFIRMED (EXPLORATORY) follows from the frozen rule: G2 failed and C1 has 1 of 2 seeds.
Caveat (disclosed in the post): no prompt-level interval, since each run has one switch time; one cue-direction set.
