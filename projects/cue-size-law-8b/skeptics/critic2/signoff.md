role: critic2
model: anthropic/claude-opus-5-5
verdict: PASS

Claude fallback for the cross-family review, theory-1118 (5-seed refill). openai/gpt-5.5 gave one REJECT (#25224: its own
rebuild matched every number, but Figure 1's answer line did not say what failed), then reached its $2.46 cap ($2.52) on
the rerun before a second verdict. This sign-off is Claude, so it is NOT a cross-family check.

Rebuilt with my own code (rebuild.py, json + numpy + scipy, 0.7 s) from data/runs, data/runs_06b, data/gates.json,
data/g2_diag.json at kran-research origin/swarm/theory-1118-cue-size-law-8b 487bdc499. Switch time re-derived from
dev_curve + dense_curve with D = gap - gap0 (first check with D >= 0.10 twice in a row): 78/78 cells equal T_dense.
Post read on deploy preview 221 at blog 7e348d0 (modified_time 2026-11-18T21:42:36Z) and its figure1.html.

| claim | rebuild | post |
|---|---|---|
| 8B per-seed slopes | -0.886 / -0.890 / -0.936 / -0.940 / -0.956 | same |
| 8B seed-level (t, df 4) | -0.9215 [-0.9606, -0.8825] = x1.894 [1.844, 1.946] | -0.92 [-0.96, -0.88], x1.89 [1.84, 1.95] |
| 0.6B rerun seed-level | -0.8992 [-0.9483, -0.8502] = x1.865 [1.803, 1.930] | -0.90 [-0.95, -0.85], x1.87; Fig 1 x1.80 to x1.93 |
| S1 Welch diff | -0.0223 [-0.0748, +0.0303], df 7.6 | -0.02 (-0.07 to 0.03) |
| 8B interval inside [-1.4, -0.7] | yes | yes |
| pooled OLS, 30 runs | -0.9215, t-interval [-0.969, -0.874] | main: -0.97 to -0.87; foldout [-0.965, -0.878] = the MLE z-interval (unlabelled) |
| medians 8B / 0.6B | 24,40,72,136,296,664 / 32,48,80,152,304,680 | same; "2% to 25% fewer" ok |
| T table, 60 cells | equal | equal |
| prediction range from seeds 0-2 | [-1.043, -0.766]; seeds 3, 4 = -0.940, -0.956 | -1.04 to -0.77; -0.94, -0.96 |
| halves (3 largest / 3 smallest) | -0.737 / -1.098 | -0.74 / -1.10 |
| quadratic curvature t, 8B | 5.46 | 5.5 |
| C1 coin flip, mini + laptop copy | no switch, 8,192 updates, max D 0.0353 | 3.5 points, no switch |
| C2 zero cue | no switch, 1,024 updates, max D 0.0000 | gap exactly 0 |
| mini vs laptop | 19/19 cells equal curves and T (1,539 points) | 18 main copies equal |
| G2(b) | 6/64 over 0.05 or top-1 differs (1 top-1 miss, inside the 6), max 0.1287 | 6 of 64, max 0.13 |
| G2 diagnosis | fp32 NDIF input 2/64 max 0.140; bf16 our cache 3/64 max 0.219 | same |
| Fig 1 answer line | names the failed check: 'missed the full model's first-word probabilities on 6 of 64 questions' | fixed (REJECT #25224 resolved) |

Status UNCONFIRMED (EXPLORATORY) follows from the frozen rule: G2 failed and C1 has 1 of 2 seeds.
Non-blocking: label the foldout's pooled interval [-0.965, -0.878] as the censored-fit (MLE, z) interval, since the main
text quotes the OLS t-interval -0.97 to -0.87 for the same fit.
