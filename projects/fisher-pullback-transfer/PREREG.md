# Pre-registration — fisher-pullback-transfer

Room: `theory-1017-fisher-pullback`. Night: 2026-10-17. Slug: `fisher-pullback-transfer`.

## Prediction (verbatim, fixed 2026-10-17 14:53 CEST, before any experiment; never changed)

> Prediction: On held-out (target, probe) pairs, the Fisher pull-back cosine predictor gets R^2 >= 0.6
> for AUROC drop and beats both the whitened and the plain cosine by >= 0.10 R^2. The margin is
> pre-registered.
> P(prediction holds): 0.40
> If it holds: Turns the transfer formula from a property of evaders we designed into a monitor-design
> theory for realistic attackers: given the clean output Fisher, a designer can compute which monitors
> fall together and pick probes with low cos_F.
> If not: Real behaviour-preserving attackers do not take the second-order cheapest path (higher-order
> KL terms, nonlinear upstream moves, or early stopping dominate). Then the transfer law covers only
> idealised evaders, and the next step is to measure the attacker's actual edit direction and stop
> assuming one.

## Analysis plan (frozen 2026-10-17 14:59 CEST (commit time), before any attack run and before any transfer number exists)

Agreed in the room (#1639 Vera, #1643 Tove's approval conditions, #1647 Bruno, #1648 Felix, #1650 Vera).
Any change after this point goes to `DEVIATIONS.md` with a time stamp and a reason; the rule below is
still reported.

### Why the bare cosine is replaced by a mapped predictor (written before data)

The 50 probes are different concepts on different datasets. AUROC does not change under a shift that
is the same for every input, so a bare cosine `cos_F(u, w)` cannot by itself predict the AUROC drop of
probe `u` on `u`'s own data: the drop also depends on how much `u`'s two classes differ in `w`-score.
The operational "Fisher pull-back predictor" is therefore the predicted drop below. It uses the
pull-back metric in exactly one place (the ratio `r_P`), and the plain and whitened predictors use the
identical formula with only the matrix `P` changed. The literal `cos_F` is reported as a secondary row.

### Notation

- `h(x)` in R^512: Pythia-70M (step143000) residual stream after block index 2, i.e.
  `hidden_states[3]`, mean-pooled over token positions 1..T-1 of a window. This is "layer 3".
- Probe `u`: logistic regression (L2, C = 1.0, raw activations, no standardisation) on clean train
  activations of its own dataset, one-vs-rest per label. Score `s_u(x) = u'h(x)` (bias drops out of AUROC).
- `G_u(v)`: class-mean gap of score `s_v` on `u`'s clean TEST set: `E[s_v | y_u = 1] - E[s_v | y_u = 0]`.
- `P` (the move matrix of the cheapest edit) is one of:
  - plain: `P = I`;
  - whitened: `P = Sigma`, Ledoit-Wolf covariance of clean mean-pooled activations on the retain windows;
  - Fisher pull-back: `P = (F + lambda I)^-1`.
- `r_P(u, w) = u'P w / w'P w`.

### The predictor (one formula, three matrices)

Edit model: the attack moves an input along `P w` by an amount affine in its clean `w`-score,
`dh(x) = -(a + b s_w(x)) P w`. The constant `a` cancels in every AUROC. The target's own class gap
changes by `delta_w = -b (w'P w) G_w(w)`; `delta_w` is MEASURED on the target's test set after the attack
(attacked `s_w` gap minus clean `s_w` gap, in raw score units). This fixes `b` without any fit on the
other probes. Then on `u`'s clean test set the predicted attacked score is

    s_u_pred(x) = s_u(x) + (delta_w / G_w(w)) * r_P(u, w) * s_w(x)

- **Primary (frozen):** predicted AUROC after = empirical AUROC of `s_u_pred` on `u`'s clean test labels
  (exact; keeps the change of spread; handles flips below 0.5).
  Predicted drop = AUROC_clean(u) - AUROC(s_u_pred).
- Sensitivity row (not the verdict): Gaussian map, gap shifted by `delta_w * r_P * G_u(w) / G_w(w)`,
  divided by the clean pooled within-class sd of `s_u`, through `Phi(d'/sqrt 2)`, as a difference
  `Phi(d'_clean/sqrt2) - Phi(d'_after/sqrt2)`.

### Measured drop

`AUROC_clean(u) - AUROC_attacked(u)`, signed (a probe that improves counts as negative drop), empirical
AUROC on `u`'s test set, same frozen probe weights, activations from the attacked model. Per pair, the
predicted and the measured drop are each averaged over the 3 seeds of that target (each seed uses its own
`delta_w`).

### Fisher estimate

- True Fisher (labels sampled from the clean model, not the data labels) of the output distribution with
  respect to ONE common offset added to `h3_t` at all positions of a window:
  `g = sum_t d/dh3_t [ sum_s log p(y_s | ...) ]`, `y_s ~ p_clean`, `F = E[g g'] / T`.
- Data: pile-10k odd document indices (disjoint from probe texts), the same windows' family as the
  retain loss but a disjoint split from the retain batches.
- Damping (frozen): `lambda = c * median eigenvalue(F)`, `c = 1` (scale-equivariant rule of FishBack,
  arXiv 2605.17231). Sensitivity grid `c in {0.01, 0.1, 10, 100}` and `lambda = 1e-2 tr(F)/512`,
  reported in the expert foldout, not used for the verdict.
- Sensitivity row: per-position block-diagonal Fisher `mean_t E[g_t g_t']`.
- The full eigen-spectrum of `F` is saved so any damping can be redone.

### Probes, filter, targets

- Probe families from public datasets (pile-10k source sets, AG News, DBpedia-14, SST-2, emotion,
  tweet_eval and the others in `fetch_data.py`), one-vs-rest per label.
- Filter, applied on clean data before any attack: keep probes with clean test AUROC >= 0.75. The count
  kept of the count built is logged. If more than 50 survive, 50 are drawn with seed 0.
- Targets: 5, from 5 different families, families drawn with seed 0 among families with a survivor, then
  one probe per family drawn with seed 0. Fixed before any attack. Pair `(w, w)` is excluded, so each
  target has 49 scored probes, 245 pairs.
- "Held-out": no number is fit on the scored pairs. The only per-run number, `delta_w`, comes from the
  target itself. Damping is frozen, not tuned.

### Attacker

- Fine-tune all parameters of blocks 0, 1, 2 of Pythia-70M (embeddings and blocks >= 3 frozen), Adam.
- Loss = (batch d'_w)^2 on the target's TRAIN texts (class-mean gap of `s_w` over pooled sd, squared)
  + `lambda_KL` * mean per-token KL(p_clean || p_theta) on retain windows (pile-10k odd docs, split
  disjoint from the Fisher windows).
- Stop: first check (every fixed number of steps) where the target's AUROC on its TEST texts < 0.6, or a
  step cap. A run that hits the cap without AUROC < 0.6 is a failed attack; it is reported and not scored.
- `lambda_KL`: chosen before the 5 target runs on ONE pilot probe that is not a target: the largest value
  in a fixed grid for which the pilot attack reaches AUROC < 0.6 within the step cap (the most
  behaviour-preserving attacker that still succeeds). Grid, step cap, learning rate and batch sizes are
  written to `config.yaml` before the pilot; the pilot result is logged before the target runs.
- Retain KL on held-out retain windows is logged at the stop step for every run.
- Seeds: 3 per target (15 runs).

### Decision rule (verdict)

`R^2 = 1 - SSE/SST` of predicted vs measured drop, pooled over all scored pairs (245 if all 15 runs
succeed), no fit.

- **CONFIRMED** iff `R^2_F >= 0.6` AND `R^2_F - max(R^2_Sigma, R^2_I) >= 0.10` AND
  `R^2_F - R^2_shared >= 0.10` (shared-factor baseline below; Tove's approval condition #1643).
- **NULL** iff all 5 targets have at least 2 successful seeds and the rule above fails.
- **UNCONFIRMED** if fewer than 5 targets have 2 successful seeds, or the run breaks.

Reported, not gating: per-target R^2; bootstrap over the 5 targets (resample targets, 10,000 draws) for
CIs of each R^2 and of the margins, stated as n = 5 clusters.

### Pre-declared baselines and nulls (outside the verdict except the shared-factor gate above)

1. Shared-factor baseline: the same formula with `r_P(u, w)` replaced by 1.
2. Literal cosine: `cos_F(u, w) = u'F_l^-1 w / sqrt(u'F_l^-1 u * w'F_l^-1 w)` and the same for I, Sigma;
   R^2 of an affine fit of measured drop on the cosine, fit leave-one-target-out.
3. Label-correlation null: `|corr(y_u, s_w)|` on `u`'s clean test set, affine fit leave-one-target-out.
4. Fragility baseline: mean measured drop of `u` under the OTHER 4 targets.
5. Random-metric null: `P = Q diag(eig F) Q'` with a random rotation `Q` (spectrum kept), 100 draws.
6. Within-target permutation of the Fisher predictor across probes (1,000 draws).

### Exploratory (foldout only, never the verdict)

- Tangent-kernel predictor: `P w -> K w`, `K = J_theta J_theta'` of blocks 0-2 at layer 3 on retain
  windows, via one VJP + JVP per batch.
- Measured edit direction: cosine of the measured class-mean edit on the target's test set with
  `w`, `Sigma w`, `(F + lambda I)^-1 w`, `K w`.

### Identifiability gate (clean data only, before any attack)

Compute the three predicted-drop vectors over all 245 pairs with `delta_w / G_w(w) = -0.5`. Report the
pairwise Pearson correlations. If any pair is > 0.95, the room is told before the attacks that the test
cannot separate those two metrics, and the post says so.

### Raw data saved (for the skeptics' independent rebuild)

Per run and probe: clean and attacked AUROC, clean and attacked class gaps, `delta_w`, stop step, retain
KL; probe weights and biases; train/test ids per dataset; the probe exclusion list; `Sigma`; `F` and its
eigen-decomposition; clean test scores of every probe on every dataset; the frozen config.
