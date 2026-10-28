# Skeptic sign-off (Vera), footprint-overlap, 2026-10-27 PDT

Answer to the title ("Does a footprint-overlap score, fixed before testing, predict how much an unseen monitor is
fooled?"): **No (REJECTED)**, by the frozen rule (PREREG s6): fixed-line R^2 < 0.70 in both headline policies.

All numbers below come from my own code, run on the saved raw data. I did not read or reuse build's analysis code.

## x-axis (overlap), base model only
- `vera_x.py`: reads `raw/screen/base/<tag>_rec_tail.npy` (bf16 bits -> fp32 by `<<16`) and `<tag>_plain_logits.npy`;
  own logistic fit (std on train none + A, L2 C=1, lbfgs, pos 25 `<|im_end|>`).
- 176 builds: 91 bit-exact / 84 leaky / 1 A-clone; 0 class mismatches vs `data/overlap.json` (8f13fa9).
- Primary overlap max |diff| vs overlap.json 1.3e-6; projection 1.0e-6; size 5.9e-8; all-position 2.2e-3.
- `none#rerun` is bit-identical to `none` (logits 200/200, traces 646/646): noise floor exactly 0.
- Position check before the freeze: A-vs-none test AUROC 0.998 at pos 25, 0.52 at pos 33 (`vera_overlap.py`).

## y-axis (fooled share) and the law (`vera_law.py`)
| quantity | s0@8192 | s1@8192 |
|---|---|---|
| dGAP_A [95% prompt CI] | 0.695 [0.673, 0.716] | 0.527 [0.506, 0.549] |
| status builds | 81 | 81 |
| fixed-line R^2 (status) | **0.4350** | **0.6547** |
| Pearson r^2 / free slope | 0.617 / 0.71 | 0.814 / 1.07 |
| monitor-label permutation p (9999) | 0.0001 | 0.0001 |
| R^2 without the rounding setting | 0.425 | 0.656 |
| R^2 high cluster (overlap >= 0.3, n 54) | -0.54 | -1.94 |
| depth rule <= 16 / <= 14 | -0.65 / -0.75 | 0.54 / 0.14 |
| same-layer pairs conc/disc/ties | 34/38/4 | 40/32/4 |
| Spearman L>=15 | 0.92 | 0.96 |
| bit-exact rows max |share| | 0.000 (91) | 0.000 (91) |
| 6 builds at/after block 26 (overlap 0) | share -0.018..+0.001 | -0.042..+0.002 |

Parent reproduction: policy sha256 equal to the parent manifest; P(X) for none / A / rounding setting equal to the
parent's saved whole-graph data with max |diff| 0.0.

## Nulls
- Shuffled A/none labels in the overlap fit (3 seeds): R^2 -82 to -734.
- Random unit direction instead of the classifier weight (3 seeds): R^2 -5 to -877.
- Monitor-label permutation: p 0.0001 in both. The score carries real ordering signal; it fails the frozen line.

## Robustness (`vera_robust.py`)
- Monitor bootstrap (resample the 81 builds, 5000x): s0 R^2 95% [0.10, 0.66], P(R^2 >= 0.7) 0.009;
  s1 [0.45, 0.79], P 0.28. Joint prompt x monitor bootstrap: s0 [0.08, 0.67], P 0.013; s1 [0.44, 0.79], P 0.27.
  So the No rests on s0 (clear miss) and on the within-cluster failure; s1 alone is not a clear miss.
- Knob grid on the x-axis (pos 25-28 x C 0.01-10 x std/raw; 20 settings): no setting gives both seeds >= 0.70;
  s0 never above 0.58. The frozen knob choice did not decide the answer.
- Base-model subtraction: max |base gap| 0.018; R^2 without it 0.425 / 0.669.
- Seed ceiling (post hoc, Kasper's idea, rebuilt): s0 shares predict s1 with R^2 0.401; s1 -> s0 -0.024;
  Pearson r^2 between seeds 0.79.

## Definitions not stretched
- Status set, fooled share, R^2 form, cut-off and validity all as in PREREG s3-s6 (v1.3) and D1-D3.
- The x clause (>= 2 status builds with overlap in [0.2, 0.8]) holds with 39.
- PREREG outcome: failed (R^2 part); the bit-exact part (overlap 0, share 0) and the judge check held.

## Novelty
See novelty.md: no prior work found that predicts from a monitor build's trace which unseen monitors a
monitor-sensing model is fooled by.

## Post-hoc check of Iris #8847 (pos-1 'user' token projection; not status)
Rebuilt in an eval cell with my own code (vera_pos1 numbers): Spearman within L<=14 0.719 / 0.787, same-layer pairs
52/20/4 and 54/18/4, partial rho | layer 0.85 / 0.88, fixed-line R^2 0.814 / 0.271. The pos-1 state is identical
across all 646 prompts (one deterministic vector per build). Size ratio at pos 1 ranks at chance (0.07 / 0.07);
cosine alone ranks 0.64 / 0.70; 500 random directions give rho p95 0.41 / 0.38 (max 0.56 / 0.65). Pos 2 also ranks
(0.82 / 0.60) but its R^2 is -0.11 / -1.29. So: a ranking signal, post hoc; 'how much' still fails.

## Variant and control policies (`vera_ctrl.py`, reported rows, not status)
| policy | dGAP_A (pts) [95%] | fixed-line R^2 (81 status builds) |
|---|---|---|
| s0@4096 | 55.7 [53.7, 57.6] | -7.23 |
| s0@8192 (headline) | 69.5 [67.3, 71.6] | 0.435 |
| s0@16384 | 63.3 [61.5, 65.1] | 0.562 |
| s1@4096 | 50.3 [47.5, 53.2] | 0.714 (the one checkpoint above 0.70) |
| s1@8192 (headline) | 52.7 [50.6, 54.9] | 0.655 |
| shuf control (no switch) | -0.09 [-0.41, 0.23] | undefined; max |drop| over 81 monitors 0.35 pts |

Verdict posted with `room review` on T5 (#8886): SIGN-OFF, answer to the title = No (REJECTED).
