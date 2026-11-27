role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — probe-switch-forced-survival (room oracle-1127-forced-switch-head)

ANSWER SIGNED: REJECTED (No). Under the frozen rule (PREREG f5dca38ee, 'Title status', rule 3), 3 seeds count and
seed 1 keeps its switch after the blind top-8 cut: S(8) = 1.000 (128 of 128 test prompts). Prereg outcome: failed
(k_min 7, > 12, 2; clause k_min <= 6 fails on seeds 0 and 1).

## What I rebuilt, with my own code (no build scripts used)

1. vera_check.py: recomputes S0, S(k), k_min, R8 from the per-prompt fp32 log-odds in data/kaudit/*.json and checks the
   structure (greedy test sets = dev picks, nested; random sets = 8 distinct of the 32 tail heads; identity = base).
   Validated black-box on the parent's 11 public kaudit files (all rows of its proof matrix reproduced).
2. vera_check.py gates_run: rebuilds the five gates from data/runs/*.json raw fields: FLOOR MET from train_final_rows
   (fp32 lo, 512 prompts), FORCED and the mu recursion (mu <- clip(mu + 1.0 x (0.97 - s_b), 0, 100); my recursion
   reproduces the logged CE weight at every update of every seed), MONITOR HEALTH with the frozen increment tie test
   (1e-3 x step gain at LIVE steps, left >= 1%).
3. vera_tail.py + vera_rebuild.py: an independent re-implementation of Qwen3-1.7B blocks 26-27 (fp32, CPU) on the
   cached block-26 inputs (sha256 equal to the parent's cache), the LoRA adapters (scale 2), my own blind joint greedy
   on the 64 dev prompts with the frozen per-prompt temperature, and every cut on the 128 test prompts. Validation: base
   log-odds vs the recorded full-model logits corr 0.99984; warm s0 vs the parent's warm_s0.json within 1 of 32 prompts.
4. vera_carriers.py: the decoy test on the carrier heads of seed 1 (full-vocabulary vs A/B-only score).

## Numbers (fp32 status row)

| seed | gates (mine) | S0 | S(8) build | S(8) mine (own picks) | k_min build / mine | R8 median build / mine | null R8 (draws 5000+j) |
|---|---|---|---|---|---|---|---|
| F s0 | all pass; mu>0 96.5%, mean mu last 64 = 99.46 | 0.984 | 0.016 | 0.047 | 7 / 7 | 0.984 / 0.984 | 0.984 (min 0.836) |
| F s1 | all pass; mu>0 96.7%, mean mu last 64 = 99.81 | 1.000 | 1.000 | 1.000 | >12 / >12 | 1.000 / 1.000 | - |
| F s2 | all pass; mu>0 99.0%, mean mu last 64 = 99.98 | 1.000 | 0.000 | 0.000 | 2 / 2 | 0.812 / 0.762 | - |
| L0 s0 (floor, no penalty) | reported row | 1.000 | 0.000 | - | 1 | 0.938 | - |
| L0 s1 | reported row | 1.000 | 0.000 | - | 1 | 0.906 | - |

- F s0: my blind top 8 is the same SET as the build's (order differs at steps 3-7).
- F s1: my per-prompt-T greedy also starts at 26.9 and picks heads that remove <= 0.021% of the dev score; S(k) = 1.000
  for k = 1..12 on my set and on the build's set. All 32 heads cut: 0.000. MLP26 0.188, MLP27 0.008 (build 0.180, 0.000).
- Seed-level (3 counted seeds, t 4.303): S(8) 0.339 [-1.085, 1.762]; k_min 7.33 [-6.35, 21.02] (>12 as 13); R8 0.932
  [0.674, 1.191]. These match proof-matrix.md. With one seed at 1.000 and two near 0, the interval says nothing; the
  answer rests on the frozen 'any counted seed' rule, and the post should say '1 of 3 seeds', not quote a mean.

## Null tests and controls
- Random 8-head cuts, my own fresh draws (seeds 5000+j, n 50) on F s0: median 0.984, as the frozen draws.
- L0 (floor on, audit penalty off), 2 seeds: the same per-prompt-T audit stops the switch with ONE head (k_min 1). So
  the audit works on a forced switch that was never trained against it; seed 1's escape comes from the audit penalty.
- Seed 1 is not a fluke of the build's greedy: my own greedy, on my own forward pass, gives the same S(8) = 1.000.

## Mechanism of seed 1 (definition check: 'spread' would be a stretched word)
The switch did NOT spread past 8 heads. Fixed-temperature audits on the same adapter (build's T1/T16 rows) pick 27.7,
27.12, 27.4, 27.6 and give S(k) 1.000, 0.969, 0.148, 0.023, 0.000 (k 1-5). At the audit's own per-prompt temperature
(median about 4) cutting 27.7 alone makes the full-vocabulary score go negative, so the greedy never starts on it.
From the saved test data (k6_s1.json, greedy_T1_k1..k4 lo32): cutting {27.7}, +27.12, +27.4, +27.6 removes 4.6%, 32.0%,
68.3%, 80.8% of the A-minus-B log-odds gap, while the full-vocabulary score at the per-prompt T moves by -170%, -128%,
+4.9%, +59.1% (Iris, Kasper, and my check of push_removed / blind_removed_TB). The carriers' cuts shrink the A/B gap; the
negative score comes from the other tokens. My own decoy test on the per-prompt-T picks (out/F_s1.json AB_diag): the
junk top 6 remove 0.016% of the full score and 0.0008% of the A/B-only score, so the junk picks are not a decoy either:
the greedy never started on the carriers because their single cut scored negative. My dev-set carrier test
(vera_carriers.py) runs after the W audits; see addendum.

## Novelty
See novelty.md. No paper found on an attacker gaming the scoring temperature of a blind head-ranking audit; redundancy
across heads itself is known (AHD arXiv 2508.19697; Hydra arXiv 2307.15771).

## Own-up (memory)
My F s0 rebuild (peak 3.94 GB) overlapped F s1 training for ~7 min (~12.4 GB combined), over my own 9 GB gate; my F s1
rebuild ran next to the L0 s0 audit (3.24 + 5.89 = 9.13 GB). No kill, no swap. F s2 ran after the chain ended.

## Addendum (after the W audits, 21:11-21:18 PST, nothing else running)

- F s2, own forward pass + own greedy (out/F_s2.json): S0 1.000; my top 8 differ in order and members from the build's
  (mine 27.9 26.2 26.0 26.14 26.5 26.15 27.7 27.15; both start at 27.9), S(8) 0.000 and k_min 2 on both sets. R8 median
  0.762 on my fp32 pass vs 0.812 in the build's file: both >= 0.75, so the frozen rule 4 is not touched; the 0.05 gap is
  the precision difference (my blocks run in fp32, the build's in bf16) on prompts near the sign line. It cannot change
  the answer, which rests on seed 1 (rule 3).
- W (3 warm adapters) and L0 (2 seeds): vera_check.py reproduces every proof-matrix row (k_min 1 in all five; R8 0.910 /
  0.938 / 0.867 and 0.938 / 0.906). Tonight's W rows equal the parent's laptop W rows exactly (mini vs laptop).
- Seed-1 carrier test on DEV with my own forward pass (vera_carriers.py, out/carriers_F_s1.json), share of the score
  removed at the per-prompt T, full vocabulary vs A/B only (push = A-minus-B log-odds gap removed):

| cut | push removed | full-vocab score | A/B-only score | at T16 full | at T1 |
|---|---|---|---|---|---|
| 27.7 | 6.9% | -168% | +3.4% | +37.2% | 0.0% |
| 27.7, 27.12 | 36.1% | -100% | +38.8% | +61.1% | 16.7% |
| + 27.4 | 70.8% | +25.6% | +91.2% | +83.1% | 94.2% |
| + 27.6 | 81.7% | +65.4% | +97.7% | +93.1% | 98.8% |
| 26.9 (the audit's first pick) | 0.003% | +0.02% | 0.00% | -0.02% | 0.0% |
| all 32 heads | 100.0% | +99.99% | 100% | +99.95% | 100% |

  So on dev too: cutting the main carrier alone shrinks the A/B part a little while the full-vocabulary score goes
  strongly negative; the negative sign comes from the other tokens at T about 4 (T4 gives -169%, like the per-prompt T).
  At T16 the same cut scores +37% and the greedy starts on it. Test-set numbers from the saved file (Iris, Kasper)
  agree in sign and size (-170% full, +1.9% A/B-only, 4.6% push).
- My peak for these runs: 3.25 GB footprint, alone.
