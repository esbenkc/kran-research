# Proof matrix: cue-size-law-8b

Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;
dense T = the frozen primary, the 64-update grid T in brackets). Run EXPLORATORY: G2(b) failed (DEVIATIONS D2).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
| gate G2 | - | Qwen3-8B cache vs NDIF, 64 dev | none | (a) rel err max 0.0045 pass True; (b) max d lnP 0.129, pass False | data/gates.json, data/g2_diag.json, data/g2_noise.json |
| headline | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/m8_k0_s0.json |
| headline | 1 | Qwen3-8B, share 5.469% (k=0), b -4.497, mini | none | T = 32 (grid 32); gap0 +0.0192 | data/runs/m8_k0_s1.json |
| headline | 0 | Qwen3-8B, share 2.735% (k=1), b -4.497, mini | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/m8_k1_s0.json |
| headline | 1 | Qwen3-8B, share 2.735% (k=1), b -4.497, mini | none | T = 48 (grid 48); gap0 +0.0094 | data/runs/m8_k1_s1.json |
| headline | 0 | Qwen3-8B, share 1.367% (k=2), b -4.497, mini | none | T = 80 (grid 128); gap0 +0.0047 | data/runs/m8_k2_s0.json |
| headline | 1 | Qwen3-8B, share 1.367% (k=2), b -4.497, mini | none | T = 88 (grid 128); gap0 +0.0047 | data/runs/m8_k2_s1.json |
| headline | 0 | Qwen3-8B, share 0.684% (k=3), b -4.497, mini | none | T = 168 (grid 192); gap0 +0.0023 | data/runs/m8_k3_s0.json |
| headline | 0 | Qwen3-8B, share 0.342% (k=4), b -4.497, mini | none | T = 304 (grid 320); gap0 +0.0012 | data/runs/m8_k4_s0.json |
| headline | 0 | Qwen3-8B, share 0.171% (k=5), b -4.497, mini | none | T = 672 (grid 704); gap0 +0.0006 | data/runs/m8_k5_s0.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 24 (grid 24); gap0 -0.0130 | data/runs_06b/m06_k0_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 5.469% (k=0), b +0.000, mini | none | T = 32 (grid 32); gap0 -0.0130 | data/runs_06b/m06_k0_s2.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 56 (grid 64); gap0 -0.0062 | data/runs_06b/m06_k1_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 48 (grid 48); gap0 -0.0062 | data/runs_06b/m06_k1_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 2.735% (k=1), b +0.000, mini | none | T = 56 (grid 64); gap0 -0.0062 | data/runs_06b/m06_k1_s2.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 96 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 80 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 1.367% (k=2), b +0.000, mini | none | T = 88 (grid 128); gap0 -0.0030 | data/runs_06b/m06_k2_s2.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 152 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 152 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.684% (k=3), b +0.000, mini | none | T = 184 (grid 192); gap0 -0.0015 | data/runs_06b/m06_k3_s2.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 296 (grid 320); gap0 -0.0007 | data/runs_06b/m06_k4_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 304 (grid 320); gap0 -0.0007 | data/runs_06b/m06_k4_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.342% (k=4), b +0.000, mini | none | T = 368 (grid 384); gap0 -0.0007 | data/runs_06b/m06_k4_s2.json |
| S1 matched 0.6B | 0 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 720 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s0.json |
| S1 matched 0.6B | 1 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 664 (grid 704); gap0 -0.0004 | data/runs_06b/m06_k5_s1.json |
| S1 matched 0.6B | 2 | Qwen3-0.6B, share 0.171% (k=5), b +0.000, mini | none | T = 872 (grid 832); gap0 -0.0004 | data/runs_06b/m06_k5_s2.json |
| exploratory | 0 | Qwen3-8B, share 1.367% (k=2), b -4.510, mini, cache cache_qwen3-8b_ndif | none | T = 72 (grid 128); gap0 +0.0048 | data/runs_explore/m8n_k2_s0.json |
