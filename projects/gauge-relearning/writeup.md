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

## Blog post

Slug: `blog.kran.ai/_notes/gauge-tamper-resistance.md`, figure `gauge-recovery`.

## Tweet draft

Can you make removed knowledge harder to relearn with a free weight rescaling (W_Q·c, W_K/c)?
On Pythia-160M: plain SGD usually gets 2-3x slower at c=16, but not reliably, and AdamW or an
attacker who rebalances Q/K never notices. <link>
