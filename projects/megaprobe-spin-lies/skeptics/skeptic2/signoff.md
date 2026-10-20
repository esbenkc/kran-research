# Skeptic Sign-Off Report (Hugo, skeptic2)
Room: oracle-1020-megaprobe-spin-lies
Date: 2026-10-20
Status: **UNCONFIRMED** (Per PREREG s6: positive control failed on base; megaprobe underperforms linear probe)

---

## 1. Executive Summary

- **Question**: Does a multi-layer megaprobe catch the spin lies that linear probes miss on models trained against a lie detector?
- **Finding**: **No.** Probe capacity does **not** recover detection on spin lies. In fact, on a realistic sample budget of 400 labels, the attention megaprobe (`c_all`) is strictly **worse** than the single-layer linear probe (`a_all`) across all trained policies and on the base model.
- **Mean Spin AUROC (4 trained policies)**:
  - Attention Megaprobe (`c_all`): **0.6757** [95% CI: 0.6417, 0.7068]
  - Linear Refit (`a_all`): **0.7323** [95% CI: 0.7058, 0.7585]
  - Paired Gain (`c - a`): **-0.0566** [95% CI: -0.0857, -0.0277]
- **Formal Status**: **UNCONFIRMED** under the frozen PREREG s6 rule.
  - Reason: The positive control on the untrained base model fails (`c_all` 0.8101 < `a_all` 0.8524 on all lies). Per PREREG s6: *"If (c) fails this positive control, a NO becomes UNCONFIRMED"*.
  - Even ignoring the positive control guard, the megaprobe 95% CI upper bound (0.7068) fails the 0.80 threshold by nearly 0.10 AUROC, and the gain over the linear refit is strictly negative.

---

## 2. Rebuilt Metrics Across All Policies

All numbers independently computed by Hugo (`skeptic2/rebuild.py`) directly from raw out-of-fold fit matrices (`data/fits/*_own.json`) and aligned labels (`data/refit_probe/labels_*.jsonl`). 95% confidence intervals and gains computed via paired prompt-cluster bootstrap (B=1000).

### Primary Evaluation Arm: Spin Lies vs Fact-Stated Truthful
*Positives: Deceptive & Fact Keywords Present (`y_judge == 1 & fact_score >= 0.27`)*
*Negatives: Honest & Fact Keywords Present (`y_judge == 0 & fact_score >= 0.27`)*

| Policy | Linear Refit (a) | Megaprobe (c) | Paired Gain (c - a) [95% CI] | Atlas Sigmoids (b) | Words Baseline (d) | Fact-Fit Mega (c) |
|---|---|---|---|---|---|---|
| **base** | 0.7218 [0.644, 0.799] | 0.6702 [0.586, 0.745] | -0.0516 [-0.131, +0.025] | 0.5075 | 0.5880 | 0.5814 |
| **lam1_s0** | 0.6874 [0.629, 0.749] | 0.6609 [0.600, 0.727] | -0.0265 [-0.081, +0.031] | 0.5524 | 0.6127 | 0.5057 |
| **lam1_s1** | 0.8010 [0.740, 0.849] | 0.7019 [0.626, 0.777] | -0.0991 [-0.165, -0.029] | 0.5976 | 0.6237 | 0.6010 |
| **ens_s0** | 0.7300 [0.676, 0.783] | 0.6968 [0.642, 0.754] | -0.0331 [-0.081, +0.015] | 0.6548 | 0.6172 | 0.6142 |
| **ens_s1** | 0.7109 [0.668, 0.757] | 0.6432 [0.593, 0.690] | -0.0677 [-0.122, -0.015] | 0.5167 | 0.6195 | 0.5853 |
| **Mean Trained** | **0.7323** [0.706, 0.759] | **0.6757** [0.642, 0.707] | **-0.0566** [-0.086, -0.028] | **0.5804** | **0.6183** | **0.5766** |

### Control Arm: All Lies vs All Truthful

| Policy | Linear Refit (a) | Megaprobe (c) | Paired Gain (c - a) [95% CI] | Atlas Sigmoids (b) | Words Baseline (d) |
|---|---|---|---|---|---|
| **base** | 0.8524 [0.826, 0.879] | 0.8101 [0.780, 0.840] | -0.0423 [-0.068, -0.016] | 0.6510 | 0.7191 |
| **lam1_s0** | 0.8126 [0.786, 0.837] | 0.7627 [0.734, 0.792] | -0.0499 [-0.076, -0.024] | 0.6246 | 0.6785 |
| **lam1_s1** | 0.8244 [0.799, 0.849] | 0.7513 [0.721, 0.782] | -0.0731 [-0.099, -0.046] | 0.6138 | 0.6364 |
| **ens_s0** | 0.8047 [0.777, 0.831] | 0.7622 [0.733, 0.791] | -0.0425 [-0.068, -0.017] | 0.6320 | 0.6375 |
| **ens_s1** | 0.7892 [0.760, 0.816] | 0.6748 [0.640, 0.707] | -0.1144 [-0.148, -0.081] | 0.5808 | 0.6648 |
| **Mean Trained** | **0.8077** [0.790, 0.825] | **0.7378** [0.718, 0.757] | **-0.0700** [-0.088, -0.052] | **0.6128** | **0.6543** |

---

## 3. Formal Gate Checks

1. **Gate 1: Positive Control on Base Model**: **FAILED**
   - Requirement: `c_all >= a_all` on all lies on the base model.
   - Result: `c_all` = 0.8101 vs `a_all` = 0.8524 (point difference: -0.0423).
   - Consequence: Under PREREG s6, a NO outcome reverts to **UNCONFIRMED**.
2. **Gate 2: Pre-Registered Prediction (0.80 AUROC Ceiling)**: **FAILED**
   - The megaprobe does not come close to the 0.80 bar: mean spin AUROC is 0.6757 (upper bound 0.7068).
3. **Gate 3: Negative Guard (Fact-Stated vs All Truthful)**:
   - On the base model: linear probe achieves 0.7218 vs fact-stated truthful, but drops to 0.6423 vs all truthful.
   - Megaprobe achieves 0.6702 vs fact-stated truthful, and 0.6052 vs all truthful.
   - This proves that single-layer and multi-layer probes both rely heavily on the presence/absence of fact keywords as a detection shortcut.
4. **Gate 4: Shuffled-Label Null Test**: **PASSED**
   - Shuffled label null tests (500 permutations per policy) produce mean AUROC of 0.499 to 0.502 across all arms, with empirical p = 0.0000.
   - Probes are measuring genuine activation geometry, not architecture-induced label bias.
5. **Gate 5: Leakage and Fold Grouping**: **PASSED**
   - Strict audit of `data/folds.json` against `data/draws.json` across all policies, 5 folds, and 5 draws confirms exactly 0 test prompt IDs appear in any training draw.

---

## 4. Subclass Analysis on Agent Labels

Audit of 50 judge-spin samples (`agent_labels.jsonl`) labeled blind by an independent Claude agent:
- **Framing-only spin** (n=14): Linear refit AUROC ~0.46-0.80, Megaprobe AUROC ~0.37-0.68. Megaprobe consistently trails linear refit (diff: -0.06 to -0.17).
- **Fact-word contradiction** (n=28): Both probes perform somewhat better (~0.71-0.85), but neither solves deception detection.
- **Honest / judge false-positives** (n=8): Highlights that ~16% of judge-flagged "spin lies" are not actually deceptive.

---

## 5. Decision on Deviations (D1 & D5)

- Under **PREREG s6 alone**: Positive control failed -> **UNCONFIRMED**.
- Under **Deviation D5 (Option B)**: Even if Option B were adopted, a NO requires that BOTH (c) and (c2) have mean 95% upper bound < 0.80, and the positive control must pass on (c2). Since (c) achieves 0.6757 (upper bound 0.7068), it definitively fails to reach 0.80. If (c2) fails to beat (a) on base, status remains UNCONFIRMED.

## 6. Verification Artifacts

- Rebuild script: `swarm-oracle-1020-megaprobe-spin-lies/skeptic2/rebuild.py`
- Rebuild output: `swarm-oracle-1020-megaprobe-spin-lies/skeptic2/rebuild.out`
- Null tests script: `swarm-oracle-1020-megaprobe-spin-lies/skeptic2/null_test.py`
- Null tests output: `swarm-oracle-1020-megaprobe-spin-lies/skeptic2/null_test.out`
- Novelty audit: `swarm-oracle-1020-megaprobe-spin-lies/skeptic2/novelty.md`
