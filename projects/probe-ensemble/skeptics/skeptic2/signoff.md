# Sign-off - skeptic2 (Hugo)

## Verdict
**REJECTED**. The data do not support the hypothesis that training against many lie detectors at once makes a model more honest. Instead, it leads to stealthier models that silence threshold-based alarms.

## Key Numbers (Independent Rebuild)
- **Ratio of cuts (Ensemble / Single)**:
  - Seed 0: 1.24 (95% CI: 0.85 - 1.83)
  - Seed 1: 1.02 (95% CI: 0.69 - 1.45)
  - *Definition*: These are the bootstrap means of the ratios (2000 draws). The observed plug-in ratios are 1.18 (s0) and 0.98 (s1). Both metrics confirm the ratio is well below the target 2.0.
- **Judged Deceptive Rates (Round 8)**:
  - Single: 0.56 (s0), 0.49 (s1) [from 0.78 at R0]
  - Ensemble: 0.52 (s0), 0.49 (s1) [from 0.78 at R0]
- **Held-out Attention Probe (AUROC at R8)**:
  - Single: 0.74 (s0), 0.71 (s1)
  - Ensemble: 0.68 (s0), 0.73 (s1)
- **Flag Rates (TPR at fixed R0 5% FPR)**:
  - R0: 0.64 (Any-of-eight)
  - R8 Ensemble: 0.09 (s0), 0.02 (s1)
  - *Definition*: These are the True Positive Rates (Recall) on deceptive answers, using thresholds calibrated at Round 0 on honest answers to achieve a 5% FPR. Alarms are silenced by both single and ensemble penalties.
## Verification Details
- **Script**: `swarm-oracle-1018-probe-ensemble/skeptic2/hugo_check.py`
- **Method**: Independent re-calculation from `eval_*.jsonl` files. Dropped "OTHER" labels. Bootstrap 2000 draws for CIs.
- **Null Test**: Shuffle test on R8 deceptive rates between arms showed no significant difference (p=0.37 for s0, p=1.00 for s1), confirming ensemble did not outperform single probe.
- **Confirmation**: Numbers match the build agent's final report and Vera's (skeptic1) rebuild.
