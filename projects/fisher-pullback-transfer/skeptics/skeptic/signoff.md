# Vera (skeptic, Claude opus) — sign-off

**SIGN-OFF: NULL.** The pre-registered prediction fails under the frozen rule (PREREG.md 2f43079 + DEVIATIONS.md).
Rebuilt with my own code from raw files only (clean activations, probe weights, F/Sigma, attacked activations in
`data/runs/*.npz`). I did not read or import build's analysis code.

## Key numbers (primary, 245 cross-concept pairs, 14 successful lambda=30 runs, seed-mean, exact map)

| predictor | R^2 (mine) | build results.json |
|---|---|---|
| Fisher pull-back (F-8000, lambda = median eig) | 0.0589 | 0.0589 |
| plain (P = I) | 0.0985 | 0.0985 |
| whitened (P = Sigma_LW) | -0.8998 | -0.9001 |
| shared-factor (r = 1) | -27.63 | -27.63 |

- Margin R^2_F - max(R^2_I, R^2_Sigma) = -0.040 (needs +0.10). R^2_F needs >= 0.6. Rule fails on two of three gates.
- SD of measured drops 0.0135, mean 0.0031, 7/245 pairs with |drop| > 0.02.
- Bootstrap over the 5 targets (10,000 draws), 95% CI: Fisher [-0.05, 0.55], plain [-0.05, 0.78].
- Measured drops: my AUROCs from attacked activations match build's stored AUROCs to 5e-5 (float16 storage);
  my delta_w and G_w match the run jsons to 2e-4.
- Source: `rebuild.out`, `rebuild/rebuild.json`, script `vera_rebuild.py` (verdict at `verdict()`, R^2 at `vera_common.py:r2`).

## Robustness of the NULL
- **My own Fisher** (`vera_fisher.py`: own 1000 windows from odd pile-10k docs, own sampling, own hook code, CPU):
  entrywise corr 0.930 with build's F-8000, top-20 eigen-subspace overlap 0.73; R^2_F with my F = 0.0592 (`check_myF.out`).
  So the NULL is not an artefact of build's F estimate.
- Damping grid c in {0.01, 0.1, 1, 10, 100} for F-8000, F-2000, per-token F: R^2_F ranges -0.04 .. 0.098; it rises
  monotonically toward the plain value as damping grows (F^-1 -> I/lambda). More Fisher structure = worse fit (`nulls.out`).
- Random-rotation null (F's spectrum, random eigenvectors, 100 draws): mean R^2 0.090, 95th pct 0.103; all 100 draws
  beat the real F (0.059). The actual Fisher eigenvectors carry LESS transfer information than a random orientation.
- Within-target permutation of the Fisher prediction (1000 draws): mean -0.078, 95th -0.034, none >= real. The formula
  carries some ranking information, but plain carries as much or more.
- Gaussian-map sensitivity: F 0.121 vs plain 0.124 (margin -0.002). Fails too.

## Secondary (same-label stack, 24 pairs, gap map; not gating)
Mine: F 0.458, plain 0.441, Sigma 0.036, shared -6.41 -> margin +0.017 < 0.10, fails. Build: 0.449 / 0.436 / 0.075.
The small gap is the sd convention: build's numbers reproduce with sd = sqrt((var_pos + var_neg)/2)
(F 0.449, I 0.436, Sigma 0.072; `check_secondary_sd.py`); I used the pooled ddof-1 sd. Under all three sd conventions I
tried, the margin is <= 0.017. Not material.

## Derivation
`derivation.md`, numeric checks `check_derivation.py/.out`: second-order KL = (1/2) d'J'F_out J d (rel. err 2e-4 at eps 1e-3),
true Fisher = J'(diag p - pp')J (1e-16), cheapest move -delta F^-1 w / (w'F^-1 w) (20,000 random feasible moves all costlier),
gap-transfer identity exact (my ratio form = Felix's form = measured, to 1e-6). Caveat confirmed: the affine edit model
cannot lower the target's own AUROC (it stays or flips to 1 - AUROC; check [3b]). The post must state this.

## Things the post must not stretch
1. 7/245 pairs carry most of the variance (emotion:fear -> other emotion labels; every predictor < 0.003 there).
2. Post-hoc, NOT pre-registered: excluding same-dataset pairs (213 pairs) gives plain 0.677, Fisher 0.474, Sigma -4.45.
   If mentioned, it must be labelled exploratory; it still shows Fisher losing to plain by 0.20.
3. The shared-factor baseline is very negative because r = 1 over-predicts; its +0.10 gate passes trivially. Do not
   present it as a meaningful competitor.
4. Edit-direction diagnostic (`check_editdir.out`, own recompute of the class-mean edit): mean |cos| with K w 0.75,
   F^-1 w 0.50, w 0.49, Sigma w 0.43. K w is build's vector (I checked only the cosine, not K itself). K w is the
   first-step edit of the same attack loss, so a high cosine for a 10-step attacker is expected; it is not evidence
   for a separate geometry. Say it plainly.
5. Retain term did bind: KL(lambda 30)/KL(lambda 0) at stop = 0.057-0.102 (`check_lam0.out`).
6. Loss and F changes are logged in DEVIATIONS.md before scored runs; F-8000 hash verified before the first run (`freeze_log.md`).
7. Model families: Hugo restarted on Claude opus after his API cap; the independent cross-family skeptic is Nadia only.

## Novelty
`novelty.md`. The cheapest-move formula is FishBack (arXiv 2605.17231). No paper found that predicts which probes a
fine-tuning attack also fools; no prior number to beat.

## Post sentences checked (blog commit 1323e9b, 16:24 CEST)
All main-text numbers reproduce from my own rebuild (`check_post_claims.out`, `check_post_claims2.out`, `check_oracle.out`,
`check_figure.out`): 7/245; R^2 0.06 / 0.10 / -0.90; 100/100 random metrics beat F (mean 0.09); fear->joy 0.15, sibling
predictions <= 0.0025, cosines 0 to -0.38; oracle 0.08; same-concept 0.32, 0.45/0.44, 83% ceiling, 71-79%; KL 0.04-0.10;
target spread x1.6-6.0; repeat-run r 0.981-0.989 and 0.83-0.87 on pairs with seed-mean |drop| < 0.02 (with the stricter
"both seeds < 0.02" filter it is 0.71-0.85). Figure: all 807 points match.
