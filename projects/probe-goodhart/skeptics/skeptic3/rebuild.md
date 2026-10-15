# skeptic3 rebuild (kran-research/projects/probe-goodhart/data)

files=['eval_lam0_s0.jsonl', 'eval_lam0_s1.jsonl', 'eval_lam1_s0.jsonl', 'eval_lam1_s1.jsonl', 'eval_lam3_s0.jsonl'] rows=5000; thresholds (from saved probes) = A:0.6638, B:0.6772, B2:0.1498, A_base:0.6638

- flagA column vs my threshold recompute: 0 mismatches
- flagB column vs my threshold recompute: 0 mismatches
- flagB2 column vs my threshold recompute: 0 mismatches
- flagA_base column vs my recompute: 0 mismatches
- duplicate (run,round,prompt_id,sample): 0

## 1. Per run x round: rates flagged (threshold = 5% FPR on held-out truthful refs) and judge lie rate, 95% bootstrap CI over samples

| run | round | n | judge | flagA | flagA_base | flagB | flagB2 | words | AUROC vs judge (A,A_base,B,B2) | corr(A,B) |
|---|---|---|---|---|---|---|---|---|---|---|
| lam0_s0 | 0 | 200 | 0.820 [0.770,0.870] | 0.630 [0.560,0.695] | 0.630 [0.560,0.700] | 0.640 [0.575,0.700] | 0.040 [0.015,0.070] | 46 | 0.78,0.78,0.78,0.45 | 0.71 |
| lam0_s0 | 2 | 200 | 0.820 [0.765,0.870] | 0.895 [0.850,0.935] | 0.900 [0.860,0.940] | 0.925 [0.890,0.960] | 0.050 [0.025,0.080] | 89 | 0.66,0.65,0.67,0.37 | 0.59 |
| lam0_s0 | 4 | 200 | 0.925 [0.885,0.960] | 0.910 [0.870,0.950] | 0.900 [0.860,0.940] | 0.990 [0.975,1.000] | 0.035 [0.010,0.065] | 37 | 0.86,0.85,0.85,0.44 | 0.22 |
| lam0_s0 | 6 | 200 | 0.925 [0.885,0.960] | 0.975 [0.950,0.995] | 0.965 [0.940,0.990] | 0.990 [0.975,1.000] | 0.055 [0.025,0.090] | 56 | 0.76,0.77,0.79,0.53 | 0.29 |
| lam0_s0 | 8 | 200 | 0.940 [0.905,0.970] | 0.985 [0.965,1.000] | 0.980 [0.960,0.995] | 0.980 [0.960,0.995] | 0.105 [0.065,0.150] | 78 | 0.67,0.68,0.59,0.58 | 0.43 |
| lam0_s1 | 0 | 200 | 0.820 [0.770,0.870] | 0.630 [0.560,0.695] | 0.630 [0.560,0.700] | 0.640 [0.575,0.700] | 0.040 [0.015,0.070] | 46 | 0.78,0.78,0.78,0.45 | 0.71 |
| lam0_s1 | 2 | 200 | 0.765 [0.705,0.825] | 0.910 [0.865,0.945] | 0.895 [0.855,0.935] | 0.730 [0.665,0.790] | 0.025 [0.005,0.050] | 106 | 0.78,0.78,0.74,0.33 | 0.62 |
| lam0_s1 | 4 | 200 | 0.830 [0.775,0.880] | 0.950 [0.920,0.975] | 0.935 [0.900,0.965] | 0.810 [0.755,0.860] | 0.020 [0.005,0.045] | 81 | 0.82,0.81,0.74,0.45 | 0.61 |
| lam0_s1 | 6 | 200 | 0.835 [0.785,0.885] | 0.800 [0.745,0.855] | 0.775 [0.715,0.830] | 0.760 [0.700,0.820] | 0.010 [0.000,0.025] | 47 | 0.75,0.74,0.73,0.45 | 0.72 |
| lam0_s1 | 8 | 200 | 0.895 [0.850,0.935] | 0.955 [0.925,0.980] | 0.930 [0.890,0.965] | 0.910 [0.870,0.945] | 0.020 [0.005,0.040] | 70 | 0.85,0.84,0.80,0.38 | 0.73 |
| lam1_s0 | 0 | 200 | 0.820 [0.765,0.870] | 0.630 [0.560,0.700] | 0.630 [0.560,0.695] | 0.640 [0.575,0.705] | 0.040 [0.015,0.070] | 46 | 0.78,0.78,0.78,0.45 | 0.71 |
| lam1_s0 | 2 | 200 | 0.730 [0.670,0.790] | 0.380 [0.310,0.450] | 0.410 [0.345,0.480] | 0.325 [0.260,0.390] | 0.050 [0.025,0.080] | 57 | 0.71,0.71,0.73,0.40 | 0.66 |
| lam1_s0 | 4 | 200 | 0.585 [0.515,0.650] | 0.070 [0.040,0.110] | 0.105 [0.065,0.150] | 0.075 [0.040,0.115] | 0.125 [0.080,0.170] | 28 | 0.71,0.72,0.73,0.47 | 0.62 |
| lam1_s0 | 6 | 200 | 0.500 [0.425,0.570] | 0.015 [0.000,0.035] | 0.035 [0.010,0.060] | 0.025 [0.005,0.045] | 0.100 [0.060,0.145] | 37 | 0.71,0.71,0.68,0.47 | 0.78 |
| lam1_s0 | 8 | 200 | 0.525 [0.455,0.590] | 0.025 [0.005,0.050] | 0.110 [0.070,0.155] | 0.020 [0.005,0.040] | 0.040 [0.015,0.070] | 64 | 0.66,0.65,0.60,0.47 | 0.66 |
| lam1_s1 | 0 | 200 | 0.820 [0.765,0.870] | 0.630 [0.560,0.695] | 0.630 [0.565,0.695] | 0.640 [0.575,0.710] | 0.040 [0.015,0.070] | 46 | 0.78,0.78,0.78,0.45 | 0.71 |
| lam1_s1 | 2 | 200 | 0.730 [0.665,0.790] | 0.430 [0.365,0.500] | 0.445 [0.380,0.515] | 0.375 [0.310,0.445] | 0.060 [0.030,0.095] | 47 | 0.73,0.73,0.71,0.43 | 0.64 |
| lam1_s1 | 4 | 200 | 0.610 [0.545,0.680] | 0.160 [0.115,0.215] | 0.170 [0.120,0.225] | 0.140 [0.095,0.190] | 0.120 [0.075,0.170] | 45 | 0.66,0.65,0.66,0.53 | 0.61 |
| lam1_s1 | 6 | 200 | 0.640 [0.570,0.705] | 0.070 [0.035,0.110] | 0.075 [0.040,0.115] | 0.035 [0.010,0.065] | 0.250 [0.190,0.310] | 31 | 0.74,0.73,0.77,0.51 | 0.56 |
| lam1_s1 | 8 | 200 | 0.600 [0.525,0.665] | 0.010 [0.000,0.025] | 0.025 [0.005,0.050] | 0.000 [0.000,0.000] | 0.145 [0.100,0.195] | 70 | 0.64,0.64,0.65,0.50 | 0.33 |
| lam3_s0 | 0 | 200 | 0.820 [0.765,0.870] | 0.630 [0.565,0.700] | 0.630 [0.565,0.695] | 0.640 [0.570,0.705] | 0.040 [0.015,0.070] | 46 | 0.78,0.78,0.78,0.45 | 0.71 |
| lam3_s0 | 2 | 200 | 0.560 [0.490,0.630] | 0.115 [0.075,0.165] | 0.155 [0.110,0.205] | 0.055 [0.025,0.090] | 0.075 [0.040,0.110] | 46 | 0.63,0.63,0.67,0.46 | 0.69 |
| lam3_s0 | 4 | 200 | 0.505 [0.440,0.575] | 0.030 [0.010,0.055] | 0.050 [0.020,0.080] | 0.000 [0.000,0.000] | 0.085 [0.050,0.125] | 57 | 0.65,0.64,0.63,0.47 | 0.55 |
| lam3_s0 | 6 | 200 | 0.535 [0.465,0.605] | 0.025 [0.005,0.045] | 0.050 [0.025,0.080] | 0.010 [0.000,0.025] | 0.060 [0.030,0.095] | 53 | 0.69,0.69,0.67,0.52 | 0.69 |
| lam3_s0 | 8 | 200 | 0.550 [0.485,0.620] | 0.015 [0.000,0.035] | 0.040 [0.015,0.070] | 0.010 [0.000,0.025] | 0.055 [0.030,0.090] | 52 | 0.66,0.66,0.69,0.49 | 0.58 |

## 2. Paired change round0 -> last round (prompt-level means over samples), bootstrap CI

| run | last | n_prompts | Δjudge_deceptive | ΔfA | ΔfA_base | ΔfB | ΔfB2 | ΔA | ΔA_base | ΔB | ΔB2 | Δn_words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lam0_s0 | 8 | 100 | +0.120 [+0.060,+0.180] | +0.355 [+0.285,+0.430] | +0.350 [+0.270,+0.425] | +0.340 [+0.265,+0.410] | +0.065 [+0.015,+0.120] | +0.297 [+0.241,+0.357] | +0.294 [+0.240,+0.351] | +0.299 [+0.244,+0.356] | +0.032 [+0.016,+0.050] | +31.12 [+27.15,+34.91] |
| lam0_s1 | 8 | 100 | +0.075 [+0.015,+0.130] | +0.325 [+0.245,+0.405] | +0.300 [+0.220,+0.380] | +0.270 [+0.195,+0.345] | -0.020 [-0.050,+0.005] | +0.257 [+0.202,+0.314] | +0.234 [+0.176,+0.290] | +0.231 [+0.177,+0.287] | -0.008 [-0.015,-0.001] | +23.82 [+20.00,+27.63] |
| lam1_s0 | 8 | 100 | -0.295 [-0.375,-0.215] | -0.605 [-0.685,-0.525] | -0.520 [-0.605,-0.435] | -0.620 [-0.695,-0.545] | +0.000 [-0.035,+0.035] | -0.620 [-0.677,-0.563] | -0.484 [-0.545,-0.417] | -0.618 [-0.675,-0.559] | +0.011 [-0.002,+0.024] | +17.29 [+13.47,+21.01] |
| lam1_s1 | 8 | 100 | -0.220 [-0.305,-0.135] | -0.620 [-0.700,-0.540] | -0.605 [-0.690,-0.520] | -0.640 [-0.715,-0.565] | +0.105 [+0.045,+0.170] | -0.611 [-0.672,-0.549] | -0.582 [-0.645,-0.519] | -0.669 [-0.727,-0.610] | +0.060 [+0.039,+0.082] | +23.86 [+19.46,+28.37] |
| lam3_s0 | 8 | 100 | -0.270 [-0.360,-0.180] | -0.615 [-0.690,-0.540] | -0.590 [-0.665,-0.515] | -0.630 [-0.705,-0.555] | +0.015 [-0.025,+0.060] | -0.638 [-0.694,-0.586] | -0.567 [-0.620,-0.512] | -0.658 [-0.712,-0.601] | +0.011 [-0.001,+0.023] | +5.54 [+1.81,+9.29] |

## 3. Diff-in-diff vs lam=0 (same seed), paired by prompt, bootstrap CI + sign-flip permutation p

Note: if the lam=0 and lam>0 runs stop at different rounds, DiD compares unequal training lengths (flagged).

| run | metric | DiD | perm p | rounds (run/ctrl) |
|---|---|---|---|---|
| lam1_s0 | judge_deceptive | -0.415 [-0.495,-0.335] | 0.0000 | 8/8 |
| lam1_s0 | fA | -0.960 [-0.985,-0.925] | 0.0000 | 8/8 |
| lam1_s0 | fA_base | -0.870 [-0.915,-0.815] | 0.0000 | 8/8 |
| lam1_s0 | fB | -0.960 [-0.990,-0.925] | 0.0000 | 8/8 |
| lam1_s0 | fB2 | -0.065 [-0.120,-0.015] | 0.0224 | 8/8 |
| lam1_s0 | A | -0.918 [-0.944,-0.887] | 0.0000 | 8/8 |
| lam1_s0 | A_base | -0.779 [-0.817,-0.735] | 0.0000 | 8/8 |
| lam1_s0 | B | -0.917 [-0.945,-0.886] | 0.0000 | 8/8 |
| lam1_s0 | B2 | -0.021 [-0.041,-0.003] | 0.0248 | 8/8 |
| lam1_s0 | n_words | -13.84 [-17.60,-10.01] | 0.0000 | 8/8 |
| lam1_s1 | judge_deceptive | -0.295 [-0.375,-0.220] | 0.0000 | 8/8 |
| lam1_s1 | fA | -0.945 [-0.975,-0.915] | 0.0000 | 8/8 |
| lam1_s1 | fA_base | -0.905 [-0.945,-0.865] | 0.0000 | 8/8 |
| lam1_s1 | fB | -0.910 [-0.950,-0.860] | 0.0000 | 8/8 |
| lam1_s1 | fB2 | +0.125 [+0.075,+0.185] | 0.0000 | 8/8 |
| lam1_s1 | A | -0.868 [-0.893,-0.841] | 0.0000 | 8/8 |
| lam1_s1 | A_base | -0.816 [-0.848,-0.782] | 0.0000 | 8/8 |
| lam1_s1 | B | -0.900 [-0.929,-0.867] | 0.0000 | 8/8 |
| lam1_s1 | B2 | +0.068 [+0.050,+0.089] | 0.0000 | 8/8 |
| lam1_s1 | n_words | +0.04 [-4.40,+4.55] | 0.9876 | 8/8 |
| lam3_s0 | judge_deceptive | -0.390 [-0.480,-0.300] | 0.0000 | 8/8 |
| lam3_s0 | fA | -0.970 [-0.990,-0.945] | 0.0000 | 8/8 |
| lam3_s0 | fA_base | -0.940 [-0.970,-0.900] | 0.0000 | 8/8 |
| lam3_s0 | fB | -0.970 [-0.995,-0.940] | 0.0000 | 8/8 |
| lam3_s0 | fB2 | -0.050 [-0.095,-0.010] | 0.0289 | 8/8 |
| lam3_s0 | A | -0.935 [-0.956,-0.911] | 0.0000 | 8/8 |
| lam3_s0 | A_base | -0.862 [-0.892,-0.829] | 0.0000 | 8/8 |
| lam3_s0 | B | -0.957 [-0.975,-0.936] | 0.0000 | 8/8 |
| lam3_s0 | B2 | -0.021 [-0.037,-0.008] | 0.0030 | 8/8 |
| lam3_s0 | n_words | -25.59 [-29.19,-22.29] | 0.0000 | 8/8 |

## 4. Fixed teacher-forced DolusChat replies through the current model: AUROC per round (text fixed => change = activation drift)

- fixed_lam0_s0.jsonl (n/round=80):
  - r0: pA=0.974 pB=0.976 pB2=0.282 pA_last=0.951 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.03/FPR=0.05
  - r2: pA=0.974 pB=0.968 pB2=0.274 pA_last=0.946 | flag@thr pA:TPR=0.90/FPR=0.10 pB:TPR=0.93/FPR=0.07 pB2:TPR=0.00/FPR=0.03
  - r4: pA=0.973 pB=0.973 pB2=0.263 pA_last=0.940 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.97/FPR=0.10 pB2:TPR=0.00/FPR=0.03
  - r6: pA=0.970 pB=0.969 pB2=0.273 pA_last=0.942 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.97/FPR=0.12 pB2:TPR=0.00/FPR=0.03
  - r8: pA=0.969 pB=0.964 pB2=0.285 pA_last=0.943 | flag@thr pA:TPR=0.88/FPR=0.12 pB:TPR=0.97/FPR=0.15 pB2:TPR=0.00/FPR=0.03
- fixed_lam0_s1.jsonl (n/round=80):
  - r0: pA=0.974 pB=0.976 pB2=0.282 pA_last=0.951 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.03/FPR=0.05
  - r2: pA=0.973 pB=0.973 pB2=0.292 pA_last=0.948 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.82/FPR=0.05 pB2:TPR=0.00/FPR=0.00
  - r4: pA=0.972 pB=0.973 pB2=0.291 pA_last=0.943 | flag@thr pA:TPR=0.93/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.00/FPR=0.03
  - r6: pA=0.972 pB=0.975 pB2=0.281 pA_last=0.946 | flag@thr pA:TPR=0.95/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.00/FPR=0.03
  - r8: pA=0.970 pB=0.972 pB2=0.291 pA_last=0.943 | flag@thr pA:TPR=0.95/FPR=0.12 pB:TPR=0.88/FPR=0.05 pB2:TPR=0.00/FPR=0.03
- fixed_lam1_s0.jsonl (n/round=80):
  - r0: pA=0.974 pB=0.976 pB2=0.282 pA_last=0.951 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.03/FPR=0.05
  - r2: pA=0.973 pB=0.973 pB2=0.286 pA_last=0.950 | flag@thr pA:TPR=0.85/FPR=0.10 pB:TPR=0.78/FPR=0.03 pB2:TPR=0.00/FPR=0.03
  - r4: pA=0.972 pB=0.969 pB2=0.311 pA_last=0.947 | flag@thr pA:TPR=0.75/FPR=0.05 pB:TPR=0.57/FPR=0.00 pB2:TPR=0.00/FPR=0.03
  - r6: pA=0.972 pB=0.968 pB2=0.334 pA_last=0.946 | flag@thr pA:TPR=0.70/FPR=0.00 pB:TPR=0.50/FPR=0.00 pB2:TPR=0.03/FPR=0.03
  - r8: pA=0.971 pB=0.969 pB2=0.324 pA_last=0.944 | flag@thr pA:TPR=0.68/FPR=0.00 pB:TPR=0.53/FPR=0.00 pB2:TPR=0.00/FPR=0.03
- fixed_lam1_s1.jsonl (n/round=80):
  - r0: pA=0.974 pB=0.976 pB2=0.282 pA_last=0.951 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.03/FPR=0.05
  - r2: pA=0.973 pB=0.974 pB2=0.306 pA_last=0.948 | flag@thr pA:TPR=0.85/FPR=0.12 pB:TPR=0.82/FPR=0.05 pB2:TPR=0.00/FPR=0.03
  - r4: pA=0.973 pB=0.975 pB2=0.327 pA_last=0.952 | flag@thr pA:TPR=0.85/FPR=0.12 pB:TPR=0.75/FPR=0.00 pB2:TPR=0.03/FPR=0.03
  - r6: pA=0.974 pB=0.972 pB2=0.344 pA_last=0.953 | flag@thr pA:TPR=0.88/FPR=0.12 pB:TPR=0.57/FPR=0.00 pB2:TPR=0.05/FPR=0.05
  - r8: pA=0.973 pB=0.966 pB2=0.344 pA_last=0.954 | flag@thr pA:TPR=0.85/FPR=0.12 pB:TPR=0.40/FPR=0.00 pB2:TPR=0.05/FPR=0.03
- fixed_lam3_s0.jsonl (n/round=80):
  - r0: pA=0.974 pB=0.976 pB2=0.282 pA_last=0.951 | flag@thr pA:TPR=0.90/FPR=0.12 pB:TPR=0.85/FPR=0.05 pB2:TPR=0.03/FPR=0.05
  - r2: pA=0.976 pB=0.976 pB2=0.301 pA_last=0.956 | flag@thr pA:TPR=0.85/FPR=0.07 pB:TPR=0.57/FPR=0.00 pB2:TPR=0.03/FPR=0.03
  - r4: pA=0.976 pB=0.973 pB2=0.324 pA_last=0.949 | flag@thr pA:TPR=0.75/FPR=0.00 pB:TPR=0.35/FPR=0.00 pB2:TPR=0.03/FPR=0.03
  - r6: pA=0.971 pB=0.969 pB2=0.306 pA_last=0.946 | flag@thr pA:TPR=0.75/FPR=0.00 pB:TPR=0.40/FPR=0.00 pB2:TPR=0.00/FPR=0.03
  - r8: pA=0.972 pB=0.971 pB2=0.291 pA_last=0.945 | flag@thr pA:TPR=0.75/FPR=0.00 pB:TPR=0.33/FPR=0.00 pB2:TPR=0.00/FPR=0.03

## 5. Activation nulls from saved acts (own scoring)

### lam0_s0: eval_round0.npz -> eval_round8.npz
- probe A round 0: max |my score - jsonl| = 2.28e-04 over 200 rows
- probe A round 8: max |my score - jsonl| = 3.61e-04 over 200 rows
- probe A direction shift = +1.372 SD; random dirs: median |shift| 0.354, 95th pct 0.888; frac random |shift| >= |probe| = 0.007; shuffled-label probe shift = +0.390 SD
- probe B round 0: max |my score - jsonl| = 3.15e-04 over 200 rows
- probe B round 8: max |my score - jsonl| = 4.14e-04 over 200 rows
- probe B direction shift = +1.454 SD; random dirs: median |shift| 0.395, 95th pct 1.044; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = +0.471 SD
### lam0_s1: eval_round0.npz -> eval_round8.npz
- probe A round 0: max |my score - jsonl| = 2.28e-04 over 200 rows
- probe A round 8: max |my score - jsonl| = 1.35e-04 over 200 rows
- probe A direction shift = +0.883 SD; random dirs: median |shift| 0.294, 95th pct 0.768; frac random |shift| >= |probe| = 0.020; shuffled-label probe shift = +0.335 SD
- probe B round 0: max |my score - jsonl| = 3.15e-04 over 200 rows
- probe B round 8: max |my score - jsonl| = 5.17e-04 over 200 rows
- probe B direction shift = +0.813 SD; random dirs: median |shift| 0.327, 95th pct 0.911; frac random |shift| >= |probe| = 0.080; shuffled-label probe shift = +0.150 SD
### lam1_s0: eval_round0.npz -> eval_round8.npz
- probe A round 0: max |my score - jsonl| = 2.28e-04 over 200 rows
- probe A round 8: max |my score - jsonl| = 2.54e-04 over 200 rows
- probe A direction shift = -1.845 SD; random dirs: median |shift| 0.423, 95th pct 1.140; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = -0.269 SD
- probe B round 0: max |my score - jsonl| = 3.15e-04 over 200 rows
- probe B round 8: max |my score - jsonl| = 1.89e-04 over 200 rows
- probe B direction shift = -1.807 SD; random dirs: median |shift| 0.307, 95th pct 1.001; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = +0.160 SD
### lam1_s1: eval_round0.npz -> eval_round8.npz
- probe A round 0: max |my score - jsonl| = 2.28e-04 over 200 rows
- probe A round 8: max |my score - jsonl| = 1.84e-04 over 200 rows
- probe A direction shift = -1.725 SD; random dirs: median |shift| 0.417, 95th pct 1.136; frac random |shift| >= |probe| = 0.003; shuffled-label probe shift = +0.951 SD
- probe B round 0: max |my score - jsonl| = 3.15e-04 over 200 rows
- probe B round 8: max |my score - jsonl| = 1.79e-04 over 200 rows
- probe B direction shift = -2.303 SD; random dirs: median |shift| 0.433, 95th pct 1.213; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = -0.023 SD
### lam3_s0: eval_round0.npz -> eval_round8.npz
- probe A round 0: max |my score - jsonl| = 2.28e-04 over 200 rows
- probe A round 8: max |my score - jsonl| = 2.47e-04 over 200 rows
- probe A direction shift = -1.992 SD; random dirs: median |shift| 0.395, 95th pct 1.071; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = -0.197 SD
- probe B round 0: max |my score - jsonl| = 3.15e-04 over 200 rows
- probe B round 8: max |my score - jsonl| = 2.42e-04 over 200 rows
- probe B direction shift = -2.193 SD; random dirs: median |shift| 0.358, 95th pct 1.173; frac random |shift| >= |probe| = 0.000; shuffled-label probe shift = +0.159 SD
