# Skeptic Sign-Off Report (Hugo, skeptic2)
Room: oracle-1020-megaprobe-spin-lies
Date: 2026-10-20
Status: **UNCONFIRMED** (Per PREREG s6 & D1: positive control failed on base for both megaprobes; both underperform the one-layer linear probe)

---

## 1. Executive Summary

- **Question**: Does a multi-layer megaprobe catch the spin lies that linear probes miss on models trained against a lie detector?
- **Finding**: **No.** Neither multi-layer probe design recovers detection on spin lies. On a realistic budget of 400 training labels, both the Attention Megaprobe (`c_all`) and the FIBS AxialProbe (`c2_all`) underperform the single-layer linear probe (`a_all`) across all trained policies and on the untrained base model.
- **Mean Spin AUROC (4 trained policies, vs fact-stated truthful)**:
  - Single-layer linear refit (`a`): **0.7323** [95% CI: 0.7058, 0.7585]
  - Attention Megaprobe (`c`)     : **0.6757** [95% CI: 0.6417, 0.7068] (gain vs linear: **-0.0566** [-0.0857, -0.0277])
  - FIBS AxialProbe (`c2`)        : **0.5855** [95% CI: 0.5541, 0.6190] (gain vs linear: **-0.1468** [-0.1853, -0.1073])
  - Atlas Layer-Average (`b`)     : **0.6138** [95% CI: 0.5818, 0.6458]
  - Words Baseline (`d`)          : **0.6183** [95% CI: 0.5844, 0.6486]
- **Formal Status**: **UNCONFIRMED**
  - Positive Control on base (all lies): Linear = 0.8524; Megaprobe (c) = 0.8101 (FAILS); FIBS Axial (c2) = 0.7539 (FAILS).
  - Under PREREG s6 and Deviation D1, failure of the positive control converts a NO into UNCONFIRMED.
  - Raw Decision Rule without the control: REJECTED (both (c) upper 0.7068 and (c2) upper 0.6190 fall far below the 0.80 bar).

---

## 2. Rebuilt Metrics Across All Policies

All numbers independently computed by Hugo (`skeptic2/rebuild.py`) directly from raw out-of-fold fit matrices (`data/fits/*_own.json`, `data/fits/*_own_c2.json`) and aligned labels (`data/refit_probe/labels_*.jsonl`). 95% confidence intervals and gains computed via paired prompt-cluster bootstrap (B=1000).

### Primary Evaluation Arm: Spin Lies vs Fact-Stated Truthful

| Policy | Linear Refit (a) | Megaprobe (c) | Gain (c - a) [95% CI] | FIBS Axial (c2) | Gain (c2 - a) [95% CI] |
|---|---|---|---|---|---|
| **base** | 0.7218 [0.644, 0.799] | 0.6702 [0.586, 0.745] | -0.0516 [-0.131, +0.025] | 0.6480 [0.565, 0.728] | -0.0738 [-0.155, +0.010] |
| **lam1_s0** | 0.6874 [0.629, 0.749] | 0.6609 [0.600, 0.727] | -0.0265 [-0.081, +0.031] | 0.6220 [0.554, 0.685] | -0.0654 [-0.133, +0.005] |
| **lam1_s1** | 0.8010 [0.740, 0.849] | 0.7019 [0.626, 0.777] | -0.0991 [-0.165, -0.029] | 0.6070 [0.534, 0.679] | -0.1940 [-0.270, -0.117] |
| **ens_s0** | 0.7300 [0.676, 0.783] | 0.6968 [0.642, 0.754] | -0.0331 [-0.081, +0.015] | 0.5572 [0.492, 0.614] | -0.1728 [-0.251, -0.095] |
| **ens_s1** | 0.7109 [0.668, 0.757] | 0.6432 [0.593, 0.690] | -0.0677 [-0.122, -0.015] | 0.5558 [0.504, 0.609] | -0.1552 [-0.219, -0.094] |
| **Mean Trained** | **0.7323** [0.706, 0.759] | **0.6757** [0.642, 0.707] | **-0.0566** [-0.086, -0.028] | **0.5855** [0.554, 0.619] | **-0.1468** [-0.185, -0.107] |

### Positive Control Arm: All Lies vs All Truthful

| Policy | Linear Refit (a) | Megaprobe (c) | FIBS Axial (c2) |
|---|---|---|---|
| **base (Control)** | **0.8524** [0.826, 0.879] | **0.8101** [0.780, 0.840] (FAIL) | **0.7539** [0.718, 0.789] (FAIL) |
| **lam1_s0** | 0.8126 [0.786, 0.837] | 0.7627 [0.734, 0.792] | 0.6405 [0.605, 0.677] |
| **lam1_s1** | 0.8244 [0.799, 0.849] | 0.7513 [0.721, 0.782] | 0.6111 [0.573, 0.649] |
| **ens_s0** | 0.8047 [0.777, 0.831] | 0.7622 [0.733, 0.791] | 0.5976 [0.562, 0.633] |
| **ens_s1** | 0.7892 [0.760, 0.816] | 0.6748 [0.640, 0.707] | 0.6185 [0.582, 0.651] |
| **Mean Trained** | **0.8077** [0.793, 0.822] | **0.7378** [0.721, 0.756] | **0.6169** [0.599, 0.637] |

---

## 3. Pre-Registered Rules & Deviation Audit

1. **PREREG s6 Status**: **UNCONFIRMED** (positive control on (c) fails: 0.8101 < 0.8524).
2. **Deviation D1 Status**: Both candidate architectures fail to reach the 0.80 threshold (c upper bound 0.7068; c2 upper bound 0.6190). Both architectures also fail the positive control on the base model (c 0.8101 < 0.8524; c2 0.7539 < 0.8524). Thus status remains **UNCONFIRMED**.
3. **Deviation D5 (void)**: Even under D5 Option B (allowing c OR c2 to pass the control), since neither passed, status remains **UNCONFIRMED**.
4. **Gate Audits**:
   - Zero leakage confirmed between prompt-grouped folds and training draws.
   - Label-shuffled null permutations center at 0.500 (p = 0.0000).
   - Bootstrap results invariant across 4 independent random seeds.
