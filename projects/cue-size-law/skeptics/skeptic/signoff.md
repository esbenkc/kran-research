# Skeptic sign-off (Vera) — cue-size-law

Room theory-1110-cue-size-learning, 2026-11-10. Everything below was recomputed with my own code from the raw data
(`data/runs/*.json`, `data/scan*.json`, `raw/cache/*_tail.npy`, Hugging Face Qwen3-0.6B weights). I did not read or run
the builder's analysis code.

## Verdict

**SIGN-OFF: REJECTED** (No to the title). Over the pre-registered grid (6 cue sizes, factor 32, 2 seeds each plus seed 2 at k=2: 13 runs), the slope
of log(switch time) on log(cue size) is **-0.897**; t-interval [-0.981, -0.813], bootstrap [-0.943, -0.851]; the wider
interval lies entirely above -1.5, so the frozen rule (PREREG s5) gives REJECTED. Pre-registered outcome: **failed**
(part 2), parts 1, 3, 4 hold.

## Rebuild table

| Check | My number | Builder / post | Script |
|---|---|---|---|
| x for 11 alphas, x_real, u1111, -u | 776.471 ... 1.053; x_real 11.6948 | scan.json, same to 3 decimals | vera_x.py |
| rows 0-2 identical over 646 prompts (eager, compiled) | asserted | check_rows.json | vera_x.py |
| frozen T, 13 sweep runs | 32/24, 64/48, 128/128/128(s2), 192/192, 320/320, 704/704 | same, 0 mismatches over all 41 runs | vera_check.py |
| primary slope on log x (13 runs) | -0.897, t [-0.981,-0.813], boot [-0.943,-0.851] (12-run: -0.901 [-0.984,-0.819]) | same | vera_check.py |
| Tobit slope | -0.897 [-0.966,-0.828] | same | vera_check.py |
| slope on log alpha (prereg parts 2-3) | 12-run fit: part-3 fitted T at alpha 1/1024 = 680 (13-run: 686) vs 704/704 | 680 | vera_check.py |
| status from T_dense (D1) | slope -0.903, t [-0.965,-0.840] -> REJECTED; also bounded: all 32 corners of [prev+8, T] REJECTED (steepest -0.918) | REJECTED | vera_check.py, vera_dense_bound.py |
| rd_ determinism | frozen T reproduced 7/7; max abs dP 0.0025 (one pair), others 0.0 | 7/7 | vera_check.py |
| halves k0-2 vs k3-5 | -1.123 vs -0.937 (diff 0.19 < 0.5) | same | vera_check.py |
| tokens-2-3-only cues, 4 sizes x 33.3-4.20 | T 512/1,472/3,200/9,280, slope -1.376, t [-1.594,-1.157]; T_dense -1.424 [-1.640,-1.209] | -1.38 [-1.59,-1.16] | vera_rows12.py |
| C2 alpha 0, 1,024 updates | max abs gap 0.0 | 0 | vera_check.py |
| two-form fit, all 16 balanced cues (x 33.3 -> 0.088) | free -1.01; fixed -1 RSS 0.831 (best AIC); fixed -2 RSS 46.4 | — | vera_forms.py |
| curvature (bend) | quadratic term t = 3.7 with nocast k=8; t = 1.37 without it | 'one run' | vera_forms.py |
| real cue post-norm share | rows 0/1/2 = 0.2% / 70.9% / 28.9% of x^2 (99.8% on tokens 2-3) | post 99.8% | vera_x.py |
| S2 read-off at x_real | 77.2 | 77 | vera_check.json |

## Nulls

- Alpha-label permutation (20,000 shuffles of T over the 13 runs): null 95% [-0.50, 0.50], observed -0.897, p < 5e-5.
- C1 shuffled penalty, alpha 1/32, seeds 0 and 1, full 8,192 updates: no switch (max D 0.065 and 0.034, never two
  evals in a row at 0.10).
- Seeds are real: s0 vs s1 dev trajectories differ by up to 0.35 in P(praise) at every alpha; equal T at k=2..5 is the
  64-update grid.
- fp32 cue vs bf16-cast cue at matched x (k=5): T 704 = 704.

## Math

`derive_exponents.py` (sympy, asserts pass): leading-order T for SGD one vector (x^-2), Adam noise-dominated (x^-2),
Adam sign-like (x^-1), SGD LoRA (x^-2 then x^-4/3), Adam sign-like LoRA (x^-1 then x^-1/2). The two displayed formulas in
the post (T = z/(eta c alpha^2), T = z/(eta alpha)) are the sgd_single and adam_sign_single cases. The displayed fit
(log T = 6.56 - 0.90 log x) reproduces as 6.563 - 0.901. Known-answer toys: `toy_exponent2.py` (SGD one vector -1.93,
Adam LoRA -0.68), `toy_real_vera.py` (real settings: AdamW -1.1 to -1.3).

## Caveats that do not change the status

- Scope: the No holds for the frozen design (cue on all 3 template tokens). For cues on tokens 2-3 only, where the real
  build difference sits, the slope is -1.38 with a 95% interval [-1.59, -1.16] (one run per size): between doubling and
  quadrupling, not settled. The post must scope its No to the planned cues and report this.

- The bend below the planned range rests on single runs (k=6, k=7, no-cast k=8 seed 0); the curvature is significant only
  with the k=8 run. See the k=8 seed-1 line below.
- The token split (32 vs 1,472) and the 3 tokens-2-3 directions are one run each and exploratory.
- SGD arm, -u cell and the u1111 slope were not run or not part of the status: no optimizer claim is licensed.

## Novelty

See `novelty.md`: new as a measurement; theory pieces (Saxe et al. 2014 1/strength mode times) must be credited.
