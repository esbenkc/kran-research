# Skeptic Sign-off (Nadia / skeptic3)

## Answer to the Title
**REJECTED**: No. Training against many lie detectors at once does not make the model twice as honest; in this setup, the ensemble penalty cut the deceptive-answer rate by only ~1.0-1.2x compared to a single-probe penalty.

## Rebuild Evidence
I rebuilt the deceptive-answer rates from `data/results/eval_*.jsonl` using a bootstrap over prompt IDs (n=1000).

- **Seed 0**:
  - Single cut: 0.225 (0.780 -> 0.555)
  - Ensemble cut: 0.265 (0.780 -> 0.515)
  - **Ratio**: 1.178 (95% CI: [0.820, 1.765])
- **Seed 1**:
  - Single cut: 0.295 (0.780 -> 0.485)
  - Ensemble cut: 0.290 (0.780 -> 0.490)
  - **Ratio**: 0.983 (95% CI: [0.676, 1.436])

**Conclusion**: Both seeds have point estimate and CI upper bound below the pre-registered 2.0 threshold.

## Null Test
Shuffled label permutation test (n=1000) on the observed cut:
- `lam1_s0`: p < 0.001
- `ens_lam1_s0`: p < 0.001
- `lam1_s1`: p < 0.001
- `ens_lam1_s1`: p < 0.001
The reduction in lying is statistically significant, but the ensemble advantage is not.

## Definition Check
Verified that `pEns` is the `max()` of the eight probe fields (`pE_L*_mean/maxpool`).

- Script: `nadia_check.py`
- Results: `nadia_results.txt`
