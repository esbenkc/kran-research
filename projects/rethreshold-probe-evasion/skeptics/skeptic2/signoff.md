# Skeptic Sign-off - Hugo (skeptic2)

## 1. Verdict: CONFIRMED

The pre-registered prediction is **CONFIRMED** on the raw experimental data:
> *Prediction: At the retain weight where the probe's TPR at 1% FPR falls by at least 50 points, a threshold re-fit on 200 clean held-out negatives of the attacked model gets back at least half of the lost TPR (pooled over 3 seeds), while AUROC drops by no more than 0.15.*

- **Headline Lambda ($\lambda = 100,000$):**
  - **Lost TPR at $t_0$:** Seed 0: 0.9077, Seed 1: 0.8154, Seed 2: 0.9312. (All 3 seeds dropped $\ge 0.50$, satisfying the criterion).
  - **Pooled Recovered Share:** **0.9039** ($\ge 0.50$).
  - **95% Paired Bootstrap CI (2,000 draws):** **[0.8867, 1.0022]** (Lower bound $0.8867 > 0.50$).
  - **Per-Seed Shares:** Seed 0: 0.8558, Seed 1: 0.9403, Seed 2: 0.9189 (all $> 0.50$).
  - **Mean AUROC Drop:** **0.000164** ($\le 0.15$; clean 0.998927 $\to$ attacked 0.998763).
  - **Mean Common-Mode Ratio ($c$):** 0.9362.

---

## 2. Rebuilt Numbers Across All Grid Points

| Lambda ($\lambda$) | Pooled Share | 95% Paired Boot CI | Per-Seed Shares | Mean AUROC Drop | Mean $c$ | Mean Push $p$ ($\sigma$ units) | E6 Pred (shift only) | Resplit Mean (Secondary 8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | **0.9039** | [0.8867, 1.0022] | [0.8558, 0.9403, 0.9189] | 0.000164 | 0.9362 | 3.02 | 0.9947 | 0.9758 [0.9212, 1.0139] |
| **31,600** | **0.7646** | [0.6310, 0.9852] | [0.7273, 0.7832, 0.7832] | 0.001339 | 0.9501 | 5.47 | 0.9945 | 0.9407 [0.7874, 1.0227] |
| **10,000** | **0.6142** | [0.4644, 0.9581] | [0.4983, 0.6696, 0.6748] | 0.003030 | 0.9268 | 6.26 | 0.9881 | 0.8782 [0.6326, 0.9981] |
| **0** | **0.0082** | [0.0012, 0.0265] | [0.0070, 0.0087, 0.0087] | 0.423412 | 0.8759 | 34.71 | 0.0428 | 0.0113 [0.0048, 0.0290] |

### Key Observations:
1. **Trend with Attack Push:** Recovery falls monotonically as the attacker pushes harder ($\lambda = 10^5 \to 10^4$: 90.4% $\to$ 76.5% $\to$ 61.4%).
2. **Boundary at $\lambda = 0$ (Unconstrained Control):** Without an output retain penalty, the attacker eliminates separation ($p = 34.71$ sds down, AUROC drops from 0.9989 to 0.5755). Here recalibration recovers almost nothing (0.8%).
3. **Finite-Sample Tail Shortfall:** Theory formula E6 predicts $\approx 99\%$ recovery across $\lambda \ge 10^4$. The empirical shortfall on the frozen refit-200 set is primarily a threshold estimation artifact: the frozen set sits at the 2.5% quantile of the 2,000-draw resplit distribution ($0.9758$ mean at $\lambda=10^5$, $0.8782$ mean at $\lambda=10^4$).

---

## 3. Verification Details & Null Tests

- **Clean Baseline Gate Check:**
  - $t_0 = 0.815082$ (2nd-largest of 200 clean refit negatives).
  - Clean Eval TPR: $0.9597$ (572 / 596).
  - Clean Eval FPR: $0.0050$ (3 / 596).
  - Clean Eval AUROC: $0.998927$.
- **Null Test 1 (Shuffled Refit Labels):**
  - Sampling 200 items randomly from all refit statements (mixing positive and negative scores) destroys threshold tracking: mean recovered share = **-0.0715 $\pm$ 0.0104** (95% CI: [-0.0841, -0.0487]).
- **Null Test 2 (Base Model Fresh Refit Difference):**
  - Re-drawing 200 clean negatives on the base model produces a mean TPR difference of **-0.0001 $\pm$ 0.0434** (95% CI: [-0.0841, 0.0923]), confirming 0 bias.
- **Mathematical Derivations & Limits:**
  - $t_1 = t_0 - c p \sigma$ verified via shift-equivariance of order statistics.
  - Limit $p \to 0$: $Share \to c$ proven via L'Hôpital's Rule and symbolically verified with `sympy`.
  - Limit $(1-c)p > D$: $Share \to 0$ proven and verified numerically ($< 10^{-46}$).
  - Boundary limits: $c=1 \implies Share=1$; $c=0 \implies Share=0$ verified symbolically.

---

## 4. Receipts & Checker Scripts
- `hugo_rebuild.py`: Independent rebuild script (no imports from team scripts).
- `hugo_rebuild.out`: Raw console output from running the rebuild across all 12 runs.
- `hugo_rebuild_results.json`: Full structured metrics and bootstrap distributions.
- `derivation.md`: Complete derivation of the binormal share formula and mathematical limits.
- `novelty.md`: Literature comparison and contextual analysis (Bailey App. F, Libon App. B.5, Bosco App. A.28, Tong et al. 2018).
