# Vera (skeptic) sign-off — rope-sink-band (room theory-1102-rope-sink-band)

**Verdict on the title: UNCONFIRMED (Not settled), reproduced.** Pooled cut 0.548 sits between the frozen bars
(No < 0.50, Yes >= 0.70 on both sets); 0 of 20 random 3-pair bands reach it. Prereg outcome: FAILED (cut and loss
clauses). Fresh context; I did not read build's code (rsb.py, calibrate.py, heldout.py, analyze.py). Every number
here comes from my own scripts in this folder.

## Scripts and outputs

| file | what it does | output |
|---|---|---|
| vera_qwen.py | my own minimal Qwen3 decoder, CPU fp32, reads safetensors lazily (no transformers model) | — |
| vera_math.py | sympy/numeric checks of every derivation step | stdout: PASS L1-L7 |
| vera_screen.py | pre-data screen on 8 own texts (not pile): first-token split with/without <\|endoftext\|>, raw vs gap band | vera_screen.npz, vera_screen_bos.npz |
| vera_band.py | band rebuild from build's calib_cal/cal2.npz; prompt bootstrap; other weightings; per-layer bands | vera_band.json |
| vera_check.py | held-out status, paired bootstrap CIs, prereg outcome, licence rows from build's arms/*.npz | vera_check_Qwen3-1.7B.json |
| vera_rebuild.py | independent fp32 forward on the same prompts (my own rebuild of the D1 split) | vera_fp32_A_0_16.npz/.log, vera_fp32_law_A_0_8.npz/.log |
| vera_pred_law.json | out-of-sample law test, frozen 07:56 PST (sha 2c6012cb) before its run | scored in signoff below |
| novelty.md | prior-work check | — |

## Checks

1. **Derivation (vera_math.py): PASS.** L1 a pair's part of the logit depends only on the offset (sympy); L2 polar
   form; L3 zeroing a pair commutes with RoPE; L4 sum of 64 pair parts = the first token's lead (1e-9); L5 rotate-half
   pair (i, i+64) turns at base^(-i/64), pair 0 fastest; L6 location edges 20.38 / 39.65 / 45.03 -> first pairs
   21 / 40 / 46; L7 softmax over one key = 1 (token 0 cannot change in any arm). Chance a random 8-set lies inside
   the edges: 0.0328 / 1.66e-4 / 9.9e-6; for a 3-set inside i >= 46: 816/41664 = 0.020.
2. **Pre-data design fact (vera_screen.py, #12605):** prepending <|endoftext|> split the sink between positions 0
   and 1 (0.274 / 0.279) vs 0.521 / 0.003 without; the room dropped the prepend before calibration (PREREG v1.1).
3. **Split (D1):** rebuilt with my own code (permute all 10,000 rows with default_rng(0), keep >= 512 tokens):
   cal and cal2 doc ids identical to build's calib npz; my 2 peek docs (6038, 5900) are in no split.
4. **Band (vera_band.py): reproduced.** G_j identical to band.json (max diff 0.0); B = {61,62,63}, k = 3, share
   0.613; cal2 same B (0.614); 2,000/2,000 prompt-bootstrap resamples give the same B; head-normalised, unweighted
   and sink-heads-only weightings agree; e_B 32.78, energy ranks 8/9/10.
   Qwen3-0.6B cell (vera_check.py --model Qwen3-0.6B, band rebuilt from its calib_cal.npz): band {61,62,63}, share
   0.599; set A cut 0.582 [0.579, 0.585], 0/20 random >= it, dLoss +2.69; set B paused at 6 of 22 arms (disclosed).
5. **Held-out status (vera_check.py): reproduced to 3 dp.** cut(P) A 0.547 [0.545, 0.550], B 0.548 [0.546, 0.551],
   pooled 0.548 [0.546, 0.549]; random median 0.003 / 0.005 / 0.004, max R01 {60,61,63} 0.429 / 0.428 / 0.428;
   0/20 >= cut(P) on A, B, pooled. dLoss P +1.582 / +1.538 / +1.560 vs random median +0.111 / +0.100 / +0.106.
   The tight CI is real: per-prompt cut p10/median/p90 = 0.528/0.546/0.568 on set A, 0/128 prompts reach 0.70.
   Core arms A: GN 0.571 (+1.599), T 0.523 (+1.978), SUF keeps 0.453 (+0.339, position 1 rises 0.005 -> 0.233,
   so position 0+1 mass 0.449 vs 0.482 base), LOW8 0.885 (+4.174).
   Core arms B / pooled (#12826): GN 0.571 / 0.571 (+1.523 / +1.561), T 0.522 / 0.523, SUF keeps 0.472 / 0.463
   (+0.368 / +0.354; token 1 0.0051 -> 0.2132 on B), LOW8 0.884 / 0.885. All match build.
6. **Independent implementation and precision (vera_rebuild.py): reproduced.** Own fp32 CPU forward, set A prompts
   0-15, 22 status arms: cut(P) 0.5413 vs build bf16 0.5415 on the same prompts; R01 0.4243 vs 0.4246; median R
   0.0018 vs 0.0019; dLoss P +1.571 vs +1.579; per-head s max |diff| <= 0.018 in every arm. Peak RSS 2.70 GB.
7. **Null / law test (frozen before the run): PASS.** Line from the 20 frozen random bands (pooled whole-model cut
   vs calibration share of the lead): cut = 0.850*share - 0.016. On 30 unseen 3-pair sets (15 random, 15 designed
   at shares 0.25-0.58), 8 set-A prompts, fp32: 29/30 within 0.05 of the prediction, Pearson 0.995. Structured
   residual: sets holding both pairs 62 and 63 sit +0.03 to +0.07 above the line (the band +0.04).
   The random-band null itself: 0 of 20 frozen random sets reach the band's cut, and in my 30 extra sets the largest
   random cut is 0.197 (share 0.253).
8. **Definitions not stretched:** sink mass, cut, band rule, query range t >= 16 and status edges are applied as
   frozen in PREREG v1.1. Two flags for the post (#12721): the per-head split uses sink heads on the held-out
   baseline (220) while PREREG s3 defines them on calibration (225: 136/7/9/73); and the 50-90% / 10-50% bins
   were swapped in draft 8afdd14 (correct: 7 and 9).
9. **Novelty (novelty.md):** "the sink rides on slow RoPE pairs" is known as a direction (Barbero App E.1, Jonasson
   Sec 4.2, DoPE theory). New: a pre-registered causal band test with matched random bands on a QK-norm model, the
   size (3 of 64 pairs), the per-head all-or-nothing split, and the share law tested out of sample.
