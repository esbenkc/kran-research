# critic2 final cross-family review: blind-cue-recovery

Reviewer: Kasper as critic2. The openai/gpt-5.5 session hit its $2.48 cap ($2.55) before it posted anything
(room #21944). This review ran on the Claude fallback (opus), with no files left from the gpt-5.5 session.

## Rebuild (own code, packed data only)
Run from projects/blind-cue-recovery with the torch venv: `python skeptics/critic2/rebuild.py .` (110 s), then `lemma_check.py`, `extra_check.py` (outputs *.json here). `eqshot.mjs <preview URL>` takes the equation screenshots.
- `rebuild.py`: own implementation of the PREREG v2 statistic. c_p = RMSNorm26(h0_p + 0.03125 ||h0_p|| u_p, bf16 cast)
  - RMSNorm26(h0_p), using the Qwen3-0.6B block-26 input_layernorm weight read straight from the safetensors file (no
  model forward). dW = 2 B A; v1 = top eigenvector of sum_m R_m^T R_m; e1 = sum_p (v1.c_p)^2 / sum_p |c_p|^2; nulls =
  numpy rng 20261112, (1000, 3, 1024), rows normed. Leg (b) = h6..h9 through the same arm; leg (c) = slice null with MY
  seed (7), 200 draws, computed in the 48-dim span of A_t.
- Input: `data/adapters_qkv26` only (plus cue_u*.npy, h0_rows012_u16.npy). Runtime 110 s, 0.98 GB RSS.
- Result (`rebuild.json`): 108 cells compared against data/summary.json (RAW, XS, SS, SM at t_h, T, 256; SMv, SM0,
  E1-RAW at t_h and T): 0 rank diffs, 0 beats-chance diffs, even with a different slice-null seed.

| claim (post line) | post | rebuild |
|---|---|---|
| Phase 0: RAW rank 1 at 256 (:46) | 2 of 6 | 302, 1, 705, 1, 976, 256 = 2 of 6 |
| SM at half switch (TLDR :72) | 240th-908th, 0 of 6 | 438, 454, 735, 709, 908, 240; 0 of 6 |
| SM at the switch (:84) | 279th-939th | 390, 504, 681, 680, 939, 279; 0 of 6 |
| SS at half switch (:72) | 336th or worse | 441, 408, 724, 725, 871, 336 |
| SS at 256 (figure) | 269/4/2/1/963/353 | 269, 4, 2, 1, 963, 353 |
| SMv, SM0 at half switch (:128) | 359-792, 271-901, 0 of 6 | 440,392,695,762,792,359; 455,410,707,693,901,271; 0 of 6 each |
| E1-RAW at switch / half (:72, :88, :125) | 4 of 6 / 2 of 6 | ranks 1,1,1,1,3,36 / 29,1,45,1,203,140 |
| replicates (:124) | 0 of 6 | 4 single twins and 2 halves: 0 of 6 at t_h and at T (`extra_check.py`) |
| reference rows before 256 (:76) | far from rank 1 | best rank 240 (SM), 268 (XS), 286 (SS), 280 (SMv), 271 (SM0) |
| SM residual vs raw v1 at half switch (:84) | 0.987-0.998 | 0.987-0.998 (`lemma_check.py`) |
| SS twin v1 vs target v1 (:86) | 0.96-1.00 | 0.962-0.996 |
| target/SS top singular value (:86) | 1.2-1.8x | 1.23-1.82 |
| SM twin-mean v1 vs target v1 (:86) | 0.72-0.97 | 0.721-0.970 |
| A moves by the switch (:66) | about 4% | ||A_T-A_0||/||A_0|| over q,k,v = 3.5%-4.9% (raw step0) |
| E2 half-switch ranks and D (:92, :127) | 768,753,10,1,1; 4,14,1,907,11; 0.1-0.2 points | e2.json same; D 0.0016, 0.0011 |

## Math
- Lemma 1 (E tr(P_S P_S') = k k'/d): signed-permutation invariance gives E[P_S'] = c I, trace fixes c = k'/d. Steps check.
- Lemma 2: M v1 = 0 (orthogonal row spaces), so ||W-M|| >= s1. Write a top vector x = c a + s b (a in row(W), b in row(M);
  a part outside both maps to 0). ||(W-M)x|| <= c||Wa|| + s sM. With a = alpha v1 + sqrt(1-alpha^2) a_perp, W v1 and
  W a_perp are orthogonal, so ||Wa||^2 <= alpha^2 s1^2 + (1-alpha^2) s2^2. <x, v1> = c alpha since b is orthogonal to v1.
  So |<x,v1>| >= min c alpha over the feasible set. Corollary s/sqrt(K) by the triangle inequality and Cauchy-Schwarz. Steps check.
- Own grid evaluation of b (4001 x 4001 grid in alpha, c) on XS at 8, 16, 24/32, 256: bound held in 30 of 30 cells;
  at the switch b = 0.960, 0.943, 0.942, 0.959, 0.978, 0.961 vs measured 0.9992-0.9998 (post: 0.94-0.98 vs 0.999-1.000).
  The real slices overlap 0.047, so the lemma is approximate here; the post says so (:123).
- Display equation (:111-113) matches the code definition; it renders in light and dark (eq-light.png, eq-dark.png).

## Text and front-matter issues sent (room #21949, #21950)
1. lab_cost_usd 0.05 -> room footer API spend + $0.03 question (2.93 at 07:28 PST).
2. :88 'The cue sat in the second-strongest direction' holds in 4 of 6 at the switch only.
3. Nits: lab_hours 1.3 vs 1.23 (05:52 -> 07:06 PST); unindented display block breaks the expert list.

## Verdict
See the room (room review ... --as critic2). Numbers and math pass; the verdict is given on the head commit that
carries fixes 1-2.
