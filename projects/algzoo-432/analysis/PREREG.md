# PREREG — algzoo-432 (owner: Rasmus/review, frozen 2026-10-17 10:20)

Checker: `check_agreement.py` (independent of `core.py`; own numpy forward from the raw `.pth`).
Run: `cd swarm-algzoo-432 && .venv/bin/python check_agreement.py algorithm.py`
Contract: `algorithm.py` defines `predict(X: float64 [N,10]) -> int [N]`.

## Amendment 10:30 — new bar (Esben #1552 "Reducer til under støjgulvet" = Option 2 of #1514)

0. **Bar = the model's own noise floor**, frozen before any C1 holdout run:
   noise floor := mean agreement(model, model with every one of the 432 weights multiplied by
   (1 + 0.001·eps), eps ~ N(0,1)), over 20 draws, holdout seed 432432, N = 1e6, own forward of
   check_agreement.py. Measured (Rasmus, 10:30): **mean 0.98671**, sd 0.00419, min 0.97545, max 0.99241.
   The 0.1% level was chosen at 10:22 (#1514), before C1 existed. For reference, 0.03% noise gives
   mean 0.99610 (sd 0.00114, 20 draws); the bar moves with the noise level and the final text says so.
   PASS = agreement on `all` >= 0.98671 AND zero TAUTOLOGY-FLAGs (`BAR` in check_agreement.py), plus
   the C1 constants table (weight expressions vs fitted, with counts). Item 1's 0.99 bar is replaced.

1. **Metric.** Agreement = mean[predict(x) == argmax model(x)], x iid N(0,1)^10 float64,
   `np.random.default_rng(432432)`, N = 1e6. Bar: agreement >= 0.99 on `all` AND zero TAUTOLOGY-FLAGs.
   (N(0,1) as in the ARC post; the model was trained on Unif[-1,1], see research.md. No biases, so argmax
   depends only on x/|x|.)
2. **Holdout.** Seed 432432 runs ONLY for final numbers. Dev work uses seeds 0..99.
3. **Constants.** Every constant in `algorithm.py` is either (i) a stated expression of W entries (rounded
   value + which entries), or (ii) a fitted constant, fit on dev seeds only and listed in a table with
   its count. Constants fit to the model's OUTPUTS (distillation) count against "explained"; the final
   D1/D2 must have zero of these.
4. **No tautology.** `algorithm.py` must not load weight files or import torch/core/alg_zoo
   (regex flags). Float literals in code <= 200 (the model has 432; W_oh alone has 160), and no module-level
   (16,16) array or array with > 200 numbers. D2 (simplified RNN) only counts if:
   (a) kept W_hh entries <= 80 of 256, all others exactly 0;
   (b) every neuron has a closed-form formula of x, with a swap test: replacing the neuron by its formula
       changes agreement by < 0.3 pt (dev seed);
   (c) a "role" is a formula, not a label.
5. **Also report** (printed by the checker): error-set agreement (null = 0.000, n = 47,170) and
   near-tie agreement, gap(2nd-3rd) < 5% quantile (null = 0.658).
6. **"Every block explained"** = per-block table in EXPLANATION.md covering all 16 W_hi, 256 W_hh and
   160 W_oh entries: formula/role, or "pruned, agreement delta X".

## Null results on the holdout (seed 432432, N = 1e6)
| predict | all | model-correct | model-error set | near-tie |
|---|---|---|---|---|
| true 2nd argmax | 0.95283 [0.95241, 0.95324] | 1.000 | 0.000 | 0.658 |
| pasted 432 weights + raw recurrence (cheat, N=1e5) | 1.000 | 1.000 | 1.000 | 1.000 -> FAIL, flag "432 float literals" |
| true 2nd argmax + module-level (16,16) array (N=1e5) | 0.953 | | | -> FAIL, flag "shape (16, 16)" |
