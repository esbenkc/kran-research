# Pre-registration — probe-evasion-geometry

Room: `theory-1016-probe-evasion`. Night: 2026-10-16.

## Prediction (verbatim, fixed 2026-10-16 14:52 CEST, before any experiment; never changed)

> Prediction: The whitened-cosine prediction explains R^2 >= 0.7 of the variance of AUROC drop
> across the 50 probes (raw cosine explains <= 0.4). A refit linear probe recovers AUROC >= 0.9 on
> the encoder-trained setting.
> P(prediction holds): 0.50
> If it holds: Gives a theory of transfer: monitor designers can choose probe ensembles with low
> whitened cosine, and evasion audits gain a cheap predictor. It starts a thread that can cover
> nonlinear and pooled probes.
> If not: If the cosine does not predict transfer, the evader finds non-minimal moves (e.g. variance
> or nonlinear shifts). Then a linear picture of evasion is wrong, which matters for every
> probe-based monitor.

## Analysis plan (frozen 2026-10-16 ~15:05 CEST, before any experiment result exists)

Everything below is decided before the first encoder is trained. Deviations get logged in
`DEVIATIONS.md` with a reason and a timestamp; the original rule is still reported.

### Lemma under test

Two classes, `x | y ~ N(mu_y, Sigma)`, `Delta = mu_1 - mu_0`. For a linear probe `u`:
`d'_u = u.Delta / sqrt(u' Sigma u)` and `AUROC_u = Phi(d'_u / sqrt 2)`.
An evader that pays for its edit in the Mahalanobis norm `delta' Sigma^-1 delta` (the KL cost of a
mean shift between Gaussians with equal covariance) and must remove separation from a fixed probe
`w` moves `Delta -> Delta - t Sigma w` and leaves `Sigma` unchanged. Then exactly:

    d'_u(after) = d'_u - k * d'_w * cos_Sigma(u, w),   cos_Sigma(u,w) = u' Sigma w / sqrt(u' Sigma u * w' Sigma w)

where `k` = fraction of `w`'s separation removed. A refit linear probe keeps
`d'_refit = sqrt(Delta' Sigma^-1 Delta - k(2-k) d'_w^2)`; it fails only when `w` is the optimal probe
`Sigma^-1 Delta`.

### Settings (prediction judged on BOTH)

1. **Synthetic.** d = 64; `Sigma = R diag(lambda) R'`, `lambda_i ∝ 1/i` (i = 1..64), R a random
   rotation; `Delta` a random direction scaled so the optimal probe has d' = 4.
   n = 20,000 train / 20,000 test per seed, balanced classes.
2. **Frozen small LM.** pythia-70m residual stream, one middle layer, mean-pooled over tokens, on
   one public binary concept dataset. Dataset and layer are chosen by the research agent from the
   baseline AUROC of the full logistic probe only (it must be >= 0.95); no evasion result is looked
   at before the choice. 80/20 train/test split.

### Monitor and probe family

- Evaded probe `w` = **mass-mean probe** (difference of class means on the train split). Not the
  optimal (LDA) probe, on purpose: the lemma predicts refit failure for the optimal probe.
- **50 probes `u`**, fit on the clean train split: 25 logistic-regression probes and 25 mass-mean
  probes, each fit inside a random subspace of dimension `k` drawn log-uniformly from [2, d], then
  mapped back to full space. Not built from angles to `w`.
- Report `corr(raw cosine, whitened cosine)` over the family, so readers see the test can separate
  them.

### Evader

- `f(x) = x + MLP(x)`, MLP with 2 hidden layers of width 256, GELU, Adam.
- Loss = `lam * (gap of class-mean w-scores / pooled std of w-scores)^2` (batch d'_w squared)
  `+ ||Sigma_hat^-1/2 (f(x) - x)||^2` (Mahalanobis retain), `Sigma_hat` = Ledoit-Wolf pooled
  within-class covariance on train.
- Train until test-free train AUROC of `w` <= 0.6 or a fixed step cap.
- 3 seeds per setting (seed changes data split / synthetic draw, probe family, encoder init).

### Prediction per probe (0 parameters fit on the 50 probes)

- `d'_u = sqrt(2) * Phi^-1(AUROC_u before)` measured on clean test; same for `w`.
- `k = 1 - d'_w(after) / d'_w(before)`, from `w` only.
- Predicted `AUROC_u(after) = Phi((d'_u - k d'_w cos_Sigma(u,w)) / sqrt 2)`;
  predicted drop = `AUROC_u(before) - predicted AUROC_u(after)`.
- `cos_Sigma` uses `Sigma_hat`. **Raw baseline**: same formula with plain cosine `cos(u, w)`.
- Measured drop = AUROC of the fixed `u` on clean test minus on encoded test.

### Statistics and decision rule

- Primary: `r^2` = squared Pearson correlation of measured vs predicted drop over the 50 probes,
  per seed; decision on the mean over 3 seeds. Also reported: `1 - SS_res/SS_tot` (no fit).
- Refit: fresh logistic probe trained on encoded train, AUROC on encoded test.
- **Held** iff, in BOTH settings: whitened `r^2 >= 0.7`, raw `r^2 <= 0.4`, refit AUROC >= 0.9.
  **Failed** iff any of the three parts misses its threshold in either setting (numbers reported
  per setting and part). **Unsettled** iff a setting does not finish.

### Diagnostics (reported, not part of the decision)

- `||Sigma_after - Sigma_before||_F / ||Sigma_before||_F`, variance of `w`-scores before/after
  (mean-shift vs projection-collapse; projection version predicts
  `d'_u(after) = (d'_u - c d'_w) / sqrt(1 - c^2)` at full removal).
- Cosine of the observed class-mean shift with `Sigma w` and with `w`.
- Euclidean-retain arm (`||f(x) - x||^2`): theory says the move goes along `w` and the raw geometry
  should then predict. Same probes, same analysis.
