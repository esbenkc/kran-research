# Vera (skeptic) sign-off — theory-1031-refusal-dose-law / refusal-dose-law

Fresh context: I did not read build's code (phase0.py, run.py, analyze.py, judge.py). Every number below comes from my
own scripts in this folder, run on build's raw files (data/runs/*.jsonl replies, data/judge/*.json labels,
data/ft/*.json first-token scores, data/phase0/units.pt unit vectors, data/phase0/p_table.json) and on the model
weights through my own Qwen3 forward (vera_fwd.py: written from the config, CPU fp32, weights read from safetensors).
Definitions: PREREG v1 @2d569f6, DEVIATIONS D1-D5 (D4 = G_off_K9).

## 1. Pre-data facts (own forward, before any TEST reply existed)
- r = DoM at the L13 output, last prompt token, dev_harmful[:100] vs test_harmless[:100] (vera_dom.py, 31 s, 0.84 GB).
  cos(W_down[:,3270], r) = -0.375, rank 1 of 6144 by |cos|; layer median |cos| 0.017, p99 0.084, z = -13.1.
  Rank 1 also against r_12, r_14, r_15. Build's r: cos(r_Bruno, r_Vera) = 0.99995.
- Gate p (PREREG def) on TEST 200, pins 6/12/18/24/30: 0.0184/0.0328/0.0472/0.0616/0.0759 (build 0.0185..0.0761).
- Signed vs |m - a_t| edit (D1): share of prompt tokens with a_t > m is 0.14% at m=6, 0 at m>=18; ratio 0.9991-1.0000.

## 2. Rebuilds from raw data
| item | build | mine | script |
|---|---|---|---|
| every unit vector vs PREREG (TWIN cos, D(c) cos, norms, G_off ⊥ r) | units.pt | TWIN -0.3753..-0.3757, D(c) = c to 4 dp, G_off cos 0.0006, norms = ‖w‖ | vera_units.py |
| p per cell, all 120 gate-profile cells | p_table | equal; worst 0.3% for \|p\| > 0.003 | vera_units.py |
| split at pin 30 (comply) | G +56.0, on +18.5, off +35.5, off/G 0.63 [0.54,0.73] | +56.0 [49.0,62.5], +18.5 [12.5,24.5], +35.5 [29.0,42.5], 0.634 [0.537,0.726] | vera_rebuild.py |
| curve (70 fit cells, 5,500 replies) | a -1.783, b 13.94 (per-cell p) | a -1.763, b 13.49 (per-reply p) | vera_clauses.py |
| R^2 fit / dialed / pure dose | 0.909 / 0.940 / ~0 | 0.909 / 0.937 / -0.03 | vera_clauses.py |
| gate - curve, pins 6..30 | +12.6/+16.0/+23.0/+31.6/+38.3 | +12.5/+15.9/+23.0/+31.7/+38.6, CIs [6.3,18.7] .. [30.6,46.4] | vera_rebuild.py |
| rules (comply / non-refusal) | a_no, b_no -> REJECTED | a_no 5/5 pins, b_no; same on non-refusal -> REJECTED | vera_rebuild.py |
| c1 literal (RN+RD only) | failed | slope 2.4, gaps +17..+56 -> failed | vera_clauses.py |
| c3 Spearman(\|cos\|, \|first-token change\|) | -0.03 (signed 0.20), split-half 0.92 | -0.029 (signed 0.205, RN only -0.061), split-half 0.921 | vera_c3.py |
| downstream row L27, all-token p_down | G 0.0184 / on 0.0127 / off 0.0075 / TWIN_0 0.0146 | 0.0182 / 0.0132 / 0.0072 / 0.0138 (own forward, TEST 0-49) | vera_down.py |

## 3. Independent regeneration and judge (definition stretch)
- Own edit + own forward (vera_gen.py, TEST 0-49, 64 tokens): first 8 words equal to build's in 46/50 (G_off@30) and
  48/50 (G_on@30); COMPLY mine/build's 40/44% and 26/28% (same judge); label agreement 46/50, 47/50.
- Different judge template (vera_judge2.py, Haiku 4.5): base 14.0, G@30 70.5, G_on 31.0, G_off 48.0 ->
  G_off/G 0.60 [0.50,0.71]; b_no fires under both labelings. Spend $0.32 + $0.02.

## 4. Null tests
- Permuting p across the 70 fit cells (2,000 refits): R^2 p95 0.027, max 0.33; none reaches 0.909.
- Zero-push units at pin 30 (10 RN, 5 RD, FK), judged: median 14%, max 22% vs G_off 50.5%, base 15%.
- r stability: split-half cos 0.970; DoM on Arditi harmful_train 260 vs 260 harmless_train: cos 0.986 with r,
  cos(w_off, r_big) = -0.008.

## 5. Readout caveat (keeps a mechanism sentence out of the TLDR)
- At the last prompt token (push/‖r_l‖) the off part pushes along r_l more than the on part from L15 on
  (L15 0.35 vs 0.29; L27 0.26 vs 0.21). Averaged over all prompt tokens (the frozen readout) the order flips
  (L27 0.0072 vs 0.0132). The route of the off part is not settled.
- w_off still has 8.3% of its energy in the span of the DoM directions at the last 6 template tokens (chance 0.3%);
  pos -5 cos -0.131, pos -3 +0.148 (vera_offaxis.py). D2/D4 test this causally.

## 6. D2 / D4 (refusal directions at other tokens and layers), rebuilt
- build's w_offK9 (d4.pt) checked against MY DoM directions at the 9 tokens between the request and the reply (L13):
  energy in my 9-position span 0.00000, max |cos| 0.0015, norm 0.866 ||w|| (vera_k9.py). w_offK (d2.pt) still has
  3.3% of its energy in my span (it skipped positions -9..-6), as D4 intended to fix.
- Ratio EFFECT(G_offK9@30)/EFFECT(G@30): 0.375 [0.277, 0.475] comply, 0.375 [0.282, 0.477] non-refusal
  (paired bootstrap 2,000, seed 0); G_offK: 0.455 [0.352, 0.566]. Build: 0.375 [0.27, 0.47], 0.455 [0.35, 0.56].
  The K9 CI lower bound (0.277) is still above the frozen b_no bar of 0.25.

## 7. Verdict
SIGN-OFF: CONFIRMED that the frozen STATUS is REJECTED (answer to the title: No), on both labelings, by both rules
(a_no at 5/5 pins; b_no), under a second judge template, under my own edit construction and forward, and after
removing 11 refusal directions (ratio 0.375 [0.28, 0.48]). Not confirmed: any route for the off part (readout-dependent,
section 5). Synced receipts: data/runs, data/judge, data/phase0, data/raw, results.json byte-identical between the
swarm folder and kran-research (ft7 first-token files were still landing at review time; not headline).
Spend: $0.34 (Haiku 4.5, my judge).
