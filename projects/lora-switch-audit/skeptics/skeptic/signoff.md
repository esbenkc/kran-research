# Vera (skeptic) sign-off — theory-1112-weight-only / lora-switch-audit

**Verdict: CONFIRMED reproduction of a REJECTED answer.** The question's own test (frozen e1, top input singular
direction of the block-26 q,k,v LoRA delta) ranks the true cue first in **2 of 6** largest-cue models at 256 updates
(0 of 6 at the switch), short of the 5 of 6 the prediction needed. Every headline number below was rebuilt from the
raw adapters (`raw/runs/<run>/step<N>.pt`, parent `*_adapter.pt`) with my own code, without reading the builder's
audit code. Builder numbers (Bruno #20566, #20616, #20637, #20654, #20656, #20668) match to the printed digit.

## Code

| file | what |
|---|---|
| `vera_rank.py` | scorer: M = vstack(2 B_m A_m), m = q,k,v of block 26; candidate c_p = N26(h_p + a‖h_p‖u_p) − N26(h_p) (bf16 cast unless nocast); e1, top-k share, gain; null 1,000 Gaussian unit rows (rng 20261112) built the same way |
| `vera_final.py` | C1, C2, C3, sweep at T, gain law (bootstrap), v1.4 length runs, paired C3 → `vera_final.json` |
| `vera_nulls.py` | N1 null seed 7, N2 wrong-key matrix, N3 trained B on step-0 A, N4 B = 0 at step 0 → `vera_nulls.json` |
| `vera_diag.py`, `vera_v1.py`, `vera_wrongkey.py` | where the cue sits in the spectrum; what v1 is; why 3 wrong keys hit |

## Rebuilt numbers (frozen e1 unless named; margin = score / null p99)

| claim | cells | Vera rebuild | builder | match |
|---|---|---|---|---|
| C1 at 256 | 6 x33 models, seeds 0-5 | rank 1 in 2/6 (u1111 9.33x, u1113 10.14x); others 302, 705, 976, 256 | 2/6, same ranks | yes |
| C1 at T | same | 0/6 (392, 508, 681, 754, 949, 274) | 0/6 | yes |
| S3 reaction (gain) | same | rank 1 in 6/6 at 256; 4/6 at T (u1110 3rd, u1114 129th) | 6/6, 4/6 | yes |
| C2 controls (primary) | c2_a0 s0/s1, c1_shuf s0 at 32; parent c2_a0 s0, c1_shuf s0/s1 at end | 409, 571, 344, 179, 491, 656 (all > 10) → holds | same | yes |
| C3 below floor | nocast x0.0219, 20,480 | e1 rank 587 (0.17x) → fails; gain rank 7 (1.02x) | 587 / 7 | yes |
| paired C3 | x0.0219 run vs c2_a0 s0 (same seed) | gain rank of u1110: 39 vs 174 (10,240), 11 vs 255 (16,384), 7 vs 355 (20,480); other 6 keys move together | pending at writing | — |
| sweep at T | u1110 s0, x 33.3 … 0.217 | 0.29, 0.43, 0.23, 3.87, 3.43, 7.40, 21.30, 17.36x | 0.29 … 17.4x | yes |
| v1.4 length | len_u1110 s0, len_u1112 s2 to 4,352 | u1110 rank 1 from 512 (8.97x), 15.59x at 1,728, 18.35x at 4,352; u1112 rank 3 at 512, 1 from 704; 9.9x at 4,352 → bin (b) LENGTH | same | yes |
| gain law (S3, slope −1 frozen) | 8 sweep points at T | slope −0.40, 95% CI [−0.54, −0.16] → FAILS (excludes −1); with nocastk8 x0.088: −0.47 [−0.62, −0.36] | −0.40 [−0.54, −0.17] | yes |
| real build leak (S6) | real_s0, real_s1 end-of-run | e1 439 / 2, gain 302 / 1 | 439/302 | yes |
| determinism | len_u1110@256 vs dir_u1110@256; new vs parent dev curves | max abs diff 0.0; dev-curve gaps bit-identical at 10-23 evals in 5 runs | — | — |

## Null tests

- **N1 (different null seed, 7):** ranks move ≤ 20 (e1 302→285, 976→986); rank-1 cells stay rank 1.
- **N2 (wrong key, 30 model-key pairs, 60 cells):** median e2 rank 490, gain 387. But 3 pairs hit ≤ 10 on top-2
  share (u1111-model: u1112, u1114; u1114-model: u1112), expect 0.3. Their energy is at the null p99
  (0.0035-0.0041) while true cues sit at 5-9x p99. Planted directions are not correlated (post-norm |cos| ≤ 0.11,
  random-pair p99 0.11). Gain wrong-key hits: 2 of 60. → A rank ≤ 10 alone is a weak flag; report margins.
- **N3 (trained B on step-0 A):** gain rank 1 → 243 / 43 / 35 / 950 / 5 at 256; the reaction needs the trained A.
  This is trained-B-on-A0, not a run with A frozen.
- **N4:** lora_B = 0 exactly at step 0 in every direction run.
- **Controls** (C2 above) are the shuffled-reward and no-cue nulls; the paired no-cue run is the length-matched null for C3.

## Definitions checked (no stretch found in the frozen ops; flags for the post)

1. Candidate space is right: at the answer row the block-26 input is identical with and without the cue; the cue enters
   through q,k,v at rows 0-2 (rl.py cue_rows; my vera_rank.py diff()).
2. The goal's "1,000 random token-pair cues" became 1,000 Gaussian cues built like the planted one (disclosed D0).
3. Six adapters in the parent folder are bit-identical duplicates (rd_* = sw_*/p0 s0); the headline uses 6 fresh runs.
4. "Top direction" is fragile at x33: s1/s2 are close (e.g. 0.241/0.205) and the cue is spread over v2-v4.
5. Post flags sent to Selma (#20675, #20694): "weights did not show a cue reward had not learned" is false on the
   reaction score (rank 7 vs 355 paired); "most learned cues sit far above the teal band" false (5 of 10 below or in);
   the frozen gain law failed and must be reported; "12 times" is one model (7.4-17.2x across six).

## Novelty

See `novelty.md`. Closest: Watch the Weights (2508.00161, k = 20 directions), 2602.15195 (output-side dose response),
Lelle 2605.30189 ("dominant singular direction" hypothesis rejected at adapter level). New here: a named planted cue
ranked on the input side against an exchangeable null, across sizes, with a length-matched check that shows the
top-direction miss is about when the auditor looks.
