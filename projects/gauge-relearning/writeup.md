# Can a free weight rescaling make removed knowledge harder to relearn?

**Question:** Does a function-preserving per-head Q/K rescaling (W_Q·c, b_Q·c, W_K/c, b_K/c) slow
a fine-tuning attacker who relearns unlearned facts, once the attacker retunes or adapts?

**Builds on:** Rosati et al., *Removing Information Content Does Not Certify Tamper Resistance in
Open-Weight Models* (arXiv 2610.09004): the same gauge, plain SGD. Rosati et al., *Limits of
Convergence-Rate Control for Open-Weight Safety* (SpecDef, arXiv 2602.18868). MCU (arXiv
2605.11685), TAR (Tamirisa et al., arXiv 2408.00761). K-FAC: Martens & Grosse 2015.

**Method:** Pythia-160M (fp32, torch MPS, Linear layers of the 12 blocks trained in every phase).
200 synthetic facts "<Name> lives in the city of" → one of 20 single-token cities, learned to
100%. Bounded gradient-difference unlearning of 100 of them (ascent on facts with CE < 15, descent
on the 100 retain facts; 600 AdamW steps at 3e-5): forget accuracy 0.00-0.08, retain 1.00. Gauge
c ∈ {1/16, 1/4, 1, 4, 16} on every head (max |Δlogit| = 0.0 at every c, all seeds). Attack:
fine-tune on 50 of the forgotten facts (split A), minibatches of 10, fixed data order per seed;
steps until exact-match accuracy on A ≥ 0.9, cap 500. Attackers: SGD, AdamW (wd 0.01), K-FAC
(empirical Fisher on all of A every 10 steps, per-head-block damping, Fisher-norm trust region —
gauge-invariant), rebalance (per head ‖W_Q‖ = ‖W_K‖, then AdamW — maps every c to the same weights).
Each attacker's lr is the best of 6 on seed 0 per c (grid-edge extension rule); seeds 1-2 reuse it.
Config frozen in `results/prereg.json` (18:22) before any c ≠ 1 run. Held-out split B (50
forgotten facts, never trained) is logged. API spend: $0 (no model API calls).

Robustness data: 2 extra data orders × 3 seeds for SGD/AdamW at c ∈ {1, 4, 16}; never-knew-A
floor (seed 0); post-hoc freeze-Q/K SGD attacker (lr 3e-4, all c, 3 seeds).

## Findings

- Function preserved exactly (Δlogit 0.0); rebalance maps every c to bit-identical weights.
- SGD, best-of-grid lr per c, steps c = 1/16 … 16: s0 25/24/21/39/59, s1 52/68/58/49/90,
  s2 30/31/37/63/84. Paired c=16/c=1 ratio over 3 seeds × 3 orders: median 2.81, range 0.84-6.32,
  > 1 in 7 of 9 → fails the pre-set bar (median ≥ 1.5 AND > 1 in ≥ 80% of pairs). c=4: median 1.10.
- AdamW: s0 27/26/23/24/25, s1 42/44/46/51/45, s2 24/27/30/28/26; c=16/c=1 median 0.98.
- Rebalance + AdamW: 27, 38, 36 steps at every c (seeds 0/1/2).
- Freeze-Q/K SGD: 22, 48, 116 at every c.
- Floor (never knew A, seed 0, c=1): SGD 24, AdamW 16, rebalance 19 vs unlearned 21/23/27.
  Relearning these facts takes about as long as learning them fresh.
- Held-out B stays ≤ 0.30 during every attack.
- K-FAC: c ∈ {1, 16, 1/16} only (time); see data/meta.json.

Status by the pre-set rule: NULL. No attacker is reliably slowed at any c.

## Appendix runs (2026-10-15, after publication; "If you're curious" in the post)

Code: `appendix_exp.py` (runs, raw logs in `results/appendix/raw/`, logs and queue scripts in
`results/appendix/`), `gradient_space.py` (gradient geometry, `results/appendix/gradient_space_seed{S}.json`),
`appendix_figures.py` (figures `gauge-large-c`, `gauge-error-bars`, `gauge-gradient-space`). `exp.py` now
also logs held-out and kept accuracy on the stop step of every run (older runs: every 5 / 25 steps only).
The published figure and `data/` are unchanged.

    uv run python projects/gauge-relearning/appendix_exp.py bigc --phase grid --log2c 8 12 6 10 -8 -12
    uv run python projects/gauge-relearning/appendix_exp.py bigc --phase reps --log2c 6 8 10 12 -8 -12 --seeds 0 1 2 --orders 0
    uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 0 1 2 --orders 3 4 5 6 7 8 9 10 11 12
    uv run python projects/gauge-relearning/appendix_exp.py orders --seeds 3 4 --orders 0 1 2 3 4
    uv run python projects/gauge-relearning/appendix_exp.py stopcheck --seeds 0 1 2 --log2c -4 -2 0 2 4
    uv run python projects/gauge-relearning/appendix_exp.py stopcheck --seeds 0 --log2c 6 8 10 12
    for s in 0 1 2 3 4; do uv run python projects/gauge-relearning/gradient_space.py $s; done
    uv run python projects/gauge-relearning/appendix_figures.py --publish

- **Large c** (lr searched on seed 0 per c, seeds 1-2 reuse it; ratio = steps(c) / steps(c = 1), same
  attacker, seed, order). SGD: 3.7x (c = 64), 6.5x (256), 13.5x (1024, 2/3 censored), 8.6x (4096,
  1/3 censored), 4.3x (1/256); at 1/4096 no lr in 1e-4..3e-2 recovers (search capped at 3e-2).
  AdamW: 1.4x (64), 1.7x (256, 1024), 2.8x (4096), 1.0x (1/256), 2.9x (1/4096); slower in 3/3 at
  every c >= 64. Rebalance at lr 3e-5: 33 steps at every c (seed 0), as at c = 1.
  Kept facts at the stop step: SGD 89% (c = 1), 28% (16), 6% (1024, n = 1); AdamW 78%, 67%, 32% (256).
- **Error bars** (SGD / AdamW, c = 16 vs 1 at the seed-0 lr, 49 pairs: post's 9 + 10 orders x 3 seeds +
  5 orders x 2 new seeds). SGD: geo-mean 2.7x (95% t-interval 2.1-3.5; after 9: 2.5x, 1.4-4.5),
  median 2.56x, slower in 43/49 = 88% (Clopper-Pearson 75-95%), middle 80% of runs 0.95-8.3x. One
  pair has a censored c = 1 run (counted as faster at c = 16), two have censored c = 16 runs. Seed
  means 1.5x-4.7x; between-seed variance of log ratio 0.17 vs within 0.56, so with 5 seeds about 75%
  of the variance of the mean is seed. By the pre-set rule (#472: median >= 1.5 and slower in >= 80%)
  plain SGD at c = 16 passes with 49 pairs; it failed with 9. AdamW: 1.01x (0.95-1.07), 25/49 slower.
- **Gradient space** (exact block-wise dot products; direct check at c = 16: rel. err <= 1e-7).
  cos(mean grad A, mean grad B) at c = 1 -> 16: s0 0.37 -> 0.02, s1 0.31 -> -0.02, s2 0.29 -> 0.27,
  s3 0.18 -> -0.10, s4 0.10 -> 0.01. K rows' share of mean-A squared gradient norm: 14-34% at c = 1,
  98-99% at c = 16. Natural-gradient control (diag empirical Fisher, gauge-equivariant damping):
  0.41-0.54, c-independent. Measured held-out B at the stop step, SGD: 20% (c = 1) vs 10% (c = 16);
  AdamW 21% vs 18%. Per seed the first-step picture does not predict the size (seed 2).

## Blog post

Slug: `blog.kran.ai/_notes/gauge-tamper-resistance.md`, figure `gauge-recovery`.

## Tweet draft

Can you make removed knowledge harder to relearn with a free weight rescaling (W_Q·c, W_K/c)?
On Pythia-160M: plain SGD usually gets 2-3x slower at c=16, but not reliably, and AdamW or an
attacker who rebalances Q/K never notices. <link>
