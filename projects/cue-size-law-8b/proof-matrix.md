# Proof matrix: cue-size-law-8b

Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;
dense T = the frozen primary, the 64-update grid T in brackets). Run EXPLORATORY: G2(b) failed (DEVIATIONS D2).

seed-level: -0.922 [-0.961, -0.882] (t-interval over 5 per-seed OLS slopes (df 4); Qwen3-8B, dense T; machine per seed s0 mini, s1 mini, s2 mini, s3 laptop, s4 laptop; per-seed s0 -0.886, s1 -0.890, s2 -0.936, s3 -0.940, s4 -0.956)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
| gate G2 | - | Qwen3-8B cache vs NDIF, 64 dev | none | (a) rel err max 0.0045 pass True; (b) max d lnP 0.129, pass False | data/gates.json |
| gate G2 diagnosis | - | bf16 tail on NDIF input / on our cache | none | max d lnP 0.027 / 0.219 | data/g2_diag.json |
| gate G2 noise floor | - | NDIF input + noise of our cache's per-token size, bf16 tail, 3 draws | none | 1/64 > 0.05 (max 0.163), 3/64 > 0.05 (max 0.163), 3/64 > 0.05 (max 0.209) | data/g2_noise.json |
| determinism (mini vs laptop) | 0-2 | Qwen3-8B, every main cell of seeds 0-2 + C1 s0, same cache sha256 | none | 19/19 cells with equal curves; 1539/1539 eval points equal; max abs diff 0.0 | data/mini_vs_laptop.json |
| control copy (same seed as the mini cell; not a second C1 seed) | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | shuffled coin (C1) | no switch by 8192 updates; max D 0.035; gap0 +0.0192 | data/runs/l8_c1_s0.json |
| control | 0 | Qwen3-8B, share 0.000% (k=None), b -4.497, laptop | cue = 0 (C2, pipeline) | no switch by 1024 updates; max D 0.000; gap0 +0.0000 | data/runs/l8_c2_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/l8_k0_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/l8_k0_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | none | T = 24 (grid 24); gap0 +0.0192 | data/runs/l8_k0_s2.json |
| headline | 3 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | none | T = 24 (grid 24); gap0 +0.0192 | data/runs/l8_k0_s3.json |
| headline | 4 | Qwen3-8B, share 5.469% (k=0), b -4.497, laptop | none | T = 24 (grid 24); gap0 +0.0192 | data/runs/l8_k0_s4.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 2.735% (k=1), b -4.497, laptop | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/l8_k1_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 2.735% (k=1), b -4.497, laptop | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/l8_k1_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 2.735% (k=1), b -4.497, laptop | none | T = 40 (grid 48); gap0 +0.0094 | data/runs/l8_k1_s2.json |
| headline | 3 | Qwen3-8B, share 2.735% (k=1), b -4.497, laptop | none | T = 40 (grid 48); gap0 +0.0094 | data/runs/l8_k1_s3.json |
| headline | 4 | Qwen3-8B, share 2.735% (k=1), b -4.497, laptop | none | T = 40 (grid 48); gap0 +0.0094 | data/runs/l8_k1_s4.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 1.367% (k=2), b -4.497, laptop | none | T = 80 (grid 128); gap0 +0.0047 | data/runs/l8_k2_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 1.367% (k=2), b -4.497, laptop | none | T = 88 (grid 128); gap0 +0.0047 | data/runs/l8_k2_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 1.367% (k=2), b -4.497, laptop | none | T = 72 (grid 128); gap0 +0.0047 | data/runs/l8_k2_s2.json |
| headline | 3 | Qwen3-8B, share 1.367% (k=2), b -4.497, laptop | none | T = 64 (grid 64); gap0 +0.0047 | data/runs/l8_k2_s3.json |
| headline | 4 | Qwen3-8B, share 1.367% (k=2), b -4.497, laptop | none | T = 72 (grid 128); gap0 +0.0047 | data/runs/l8_k2_s4.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 0.684% (k=3), b -4.497, laptop | none | T = 168 (grid 192); gap0 +0.0023 | data/runs/l8_k3_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 0.684% (k=3), b -4.497, laptop | none | T = 160 (grid 192); gap0 +0.0023 | data/runs/l8_k3_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 0.684% (k=3), b -4.497, laptop | none | T = 136 (grid 192); gap0 +0.0023 | data/runs/l8_k3_s2.json |
| headline | 3 | Qwen3-8B, share 0.684% (k=3), b -4.497, laptop | none | T = 128 (grid 128); gap0 +0.0023 | data/runs/l8_k3_s3.json |
| headline | 4 | Qwen3-8B, share 0.684% (k=3), b -4.497, laptop | none | T = 128 (grid 128); gap0 +0.0023 | data/runs/l8_k3_s4.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 0.342% (k=4), b -4.497, laptop | none | T = 304 (grid 320); gap0 +0.0012 | data/runs/l8_k4_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 0.342% (k=4), b -4.497, laptop | none | T = 328 (grid 384); gap0 +0.0012 | data/runs/l8_k4_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 0.342% (k=4), b -4.497, laptop | none | T = 264 (grid 320); gap0 +0.0012 | data/runs/l8_k4_s2.json |
| headline | 3 | Qwen3-8B, share 0.342% (k=4), b -4.497, laptop | none | T = 272 (grid 256); gap0 +0.0012 | data/runs/l8_k4_s3.json |
| headline | 4 | Qwen3-8B, share 0.342% (k=4), b -4.497, laptop | none | T = 296 (grid 320); gap0 +0.0012 | data/runs/l8_k4_s4.json |
| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 0.171% (k=5), b -4.497, laptop | none | T = 672 (grid 704); gap0 +0.0006 | data/runs/l8_k5_s0.json |
| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 0.171% (k=5), b -4.497, laptop | none | T = 672 (grid 704); gap0 +0.0006 | data/runs/l8_k5_s1.json |
| determinism copy (same seed, other machine; not in the interval) | 2 | Qwen3-8B, share 0.171% (k=5), b -4.497, laptop | none | T = 640 (grid 640); gap0 +0.0006 | data/runs/l8_k5_s2.json |
| headline | 3 | Qwen3-8B, share 0.171% (k=5), b -4.497, laptop | none | T = 632 (grid 640); gap0 +0.0006 | data/runs/l8_k5_s3.json |
| headline | 4 | Qwen3-8B, share 0.171% (k=5), b -4.497, laptop | none | T = 664 (grid 704); gap0 +0.0006 | data/runs/l8_k5_s4.json |
| control | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | shuffled coin (C1) | no switch by 8192 updates; max D 0.035; gap0 +0.0192; resumed from checkpoint at [4096] (DEVIATIONS D3) | data/runs/m8_c1_s0.json |
| headline | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/m8_k0_s0.json |
| headline | 1 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/m8_k0_s1.json |
| headline | 2 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | none | T = 24 (grid 24); gap0 +0.0192 | data/runs/m8_k0_s2.json |
| headline | 0 | Qwen3-8B, share 2.735% (k=1), b -4.497, mini | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/m8_k1_s0.json |
| headline | 1 | Qwen3-8B, share 2.735% (k=1), b -4.497, mini | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/m8_k1_s1.json |
| headline | 2 | Qwen3-8B, share 2.735% (k=1), b -4.497, mini | none | T = 40 (grid 48); gap0 +0.0094 | data/runs/m8_k1_s2.json |
| headline | 0 | Qwen3-8B, share 1.367% (k=2), b -4.497, mini | none | T = 80 (grid 128); gap0 +0.0047 | data/runs/m8_k2_s0.json |
| headline | 1 | Qwen3-8B, share 1.367% (k=2), b -4.497, mini | none | T = 88 (grid 128); gap0 +0.0047 | data/runs/m8_k2_s1.json |
| headline | 2 | Qwen3-8B, share 1.367% (k=2), b -4.497, mini | none | T = 72 (grid 128); gap0 +0.0047 | data/runs/m8_k2_s2.json |
| headline | 0 | Qwen3-8B, share 0.684% (k=3), b -4.497, mini | none | T = 168 (grid 192); gap0 +0.0023 | data/runs/m8_k3_s0.json |
| headline | 1 | Qwen3-8B, share 0.684% (k=3), b -4.497, mini | none | T = 160 (grid 192); gap0 +0.0023 | data/runs/m8_k3_s1.json |
| headline | 2 | Qwen3-8B, share 0.684% (k=3), b -4.497, mini | none | T = 136 (grid 192); gap0 +0.0023 | data/runs/m8_k3_s2.json |
| headline | 0 | Qwen3-8B, share 0.342% (k=4), b -4.497, mini | none | T = 304 (grid 320); gap0 +0.0012 | data/runs/m8_k4_s0.json |
| headline | 1 | Qwen3-8B, share 0.342% (k=4), b -4.497, mini | none | T = 328 (grid 384); gap0 +0.0012 | data/runs/m8_k4_s1.json |
| headline | 2 | Qwen3-8B, share 0.342% (k=4), b -4.497, mini | none | T = 264 (grid 320); gap0 +0.0012 | data/runs/m8_k4_s2.json |
| headline | 0 | Qwen3-8B, share 0.171% (k=5), b -4.497, mini | none | T = 672 (grid 704); gap0 +0.0006 | data/runs/m8_k5_s0.json |
| headline | 1 | Qwen3-8B, share 0.171% (k=5), b -4.497, mini | none | T = 672 (grid 704); gap0 +0.0006 | data/runs/m8_k5_s1.json |
| headline | 2 | Qwen3-8B, share 0.171% (k=5), b -4.497, mini | none | T = 640 (grid 640); gap0 +0.0006 | data/runs/m8_k5_s2.json |
| headline rebuild (skeptic) | 0-4 | Qwen3-8B dense T, own code | none | -0.922 [-0.961, -0.882], 81/81 T equal | skeptics/skeptic/vera_1118.json |
| null: size labels permuted (skeptic) | 0-4 | 20,000 draws | permuted sizes | 95% [-0.365, +0.354], 0/20,000 <= -0.922 | skeptics/skeptic/vera_1118.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 24 (grid 24); gap0 -0.0130 | data/runs_06b/m06_k0_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s4.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 56 (grid 64); gap0 -0.0062 | data/runs_06b/m06_k1_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 48 (grid 48); gap0 -0.0062 | data/runs_06b/m06_k1_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 56 (grid 64); gap0 -0.0062 | data/runs_06b/m06_k1_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 48 (grid 48); gap0 -0.0062 | data/runs_06b/m06_k1_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 48 (grid 48); gap0 -0.0062 | data/runs_06b/m06_k1_s4.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 96 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 80 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 88 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 80 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 72 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s4.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 152 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 152 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 184 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 144 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 136 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s4.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 296 (grid 320); gap0 -0.0007 | data/runs_06b/m06_k4_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 304 (grid 320); gap0 -0.0007 | data/runs_06b/m06_k4_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 368 (grid 384); gap0 -0.0007 | data/runs_06b/m06_k4_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 312 (grid 320); gap0 -0.0007 | data/runs_06b/m06_k4_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 256 (grid 256); gap0 -0.0007 | data/runs_06b/m06_k4_s4.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 720 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 664 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 872 (grid 832); gap0 -0.0004 | data/runs_06b/m06_k5_s2.json |
| S1 matched 0.6B | 3 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 672 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s3.json |
| S1 matched 0.6B | 4 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 680 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s4.json |
| exploratory | 0 | Qwen3-8B, share 1.367% (k=2), b -4.510, mini, cache cache_qwen3-8b_ndif | none | T = 72 (grid 128); gap0 +0.0048 | data/runs_explore/m8n_k2_s0.json |
| exploratory | 0 | Qwen3-8B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 +0.0063 | data/runs_explore/m8x_k0_s0.json |
| exploratory | 0 | Qwen3-8B, share 0.684% (k=3), b +0.000, mini | none | T = 176 (grid 192); gap0 +0.0007 | data/runs_explore/m8x_k3_s0.json |
