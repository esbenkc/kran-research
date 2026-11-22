# Proof matrix: cue-size-law-8b-rep

Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;
dense T = the frozen primary, the 64-update grid T in brackets). Status by the frozen rule (analyze_8b.py -> data/analysis.json): CONFIRMED.

seed-level: -0.956 [-0.962, -0.949] (t-interval over 3 per-seed OLS slopes (df 2); Qwen3-8B, dense T; per-seed s5 -0.955, s6 -0.958, s7 -0.953)
prompt-level: [-0.973, -0.955] (2,000 bootstrap resamples of the 64 dev prompts, percentile 95% of the mean per-seed slope)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
| gate G2(a) | - | Qwen3-8B cache vs NDIF block-34 input, 64 dev | none | rel err max 0.0045 (pass True) | data/gates.json |
| gate G2(b) | - | fp32 tail on our cache vs NDIF, 64 dev | precision floor: same tail on NDIF's input | max d lnP 0.1287 <= 0.2792 (2 x floor max 0.1396); median 0.0043 <= 0.01; top-1 64/64 (ties as match) >= 63; pass True | data/gates.json |
| control C1 | 6 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.033; gap0 +0.0192; ran on laptop | data/runs/l8_c1_s6.json |
| control C1 | 5 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.030; gap0 +0.0192; ran on mini | data/runs/m8_c1_s5.json |
| pipeline check C2 (not a control seed) | 5 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s5.json |
| pipeline check C2 (not a control seed) | 6 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s6.json |
| headline | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s5.json |
| headline | 6 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s6.json |
| headline | 7 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s7.json |
| headline | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s5.json |
| headline | 6 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s6.json |
| headline | 7 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s7.json |
| headline | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s5.json |
| headline | 6 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s6.json |
| headline | 7 | Qwen3-8B, share 1.367% (k=2) | none | T = 64 (grid 64); gap0 +0.0047; ran on mini | data/runs/m8_k2_s7.json |
| headline | 5 | Qwen3-8B, share 0.684% (k=3) | none | T = 152 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s5.json |
| headline | 6 | Qwen3-8B, share 0.684% (k=3) | none | T = 136 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s6.json |
| headline | 7 | Qwen3-8B, share 0.684% (k=3) | none | T = 120 (grid 128); gap0 +0.0023; ran on mini | data/runs/m8_k3_s7.json |
| headline | 5 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s5.json |
| headline | 6 | Qwen3-8B, share 0.342% (k=4) | none | T = 280 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s6.json |
| headline | 7 | Qwen3-8B, share 0.342% (k=4) | none | T = 304 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s7.json |
| headline | 5 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s5.json |
| headline | 6 | Qwen3-8B, share 0.171% (k=5) | none | T = 688 (grid 704); gap0 +0.0006; ran on mini | data/runs/m8_k5_s6.json |
| headline | 7 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s7.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 -0.0067; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k0_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 -0.0034; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k1_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 -0.0017; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k2_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.684% (k=3) | none | T = 160 (grid 192); gap0 -0.0008; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k3_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 -0.0004; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k4_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.171% (k=5) | none | T = 680 (grid 704); gap0 -0.0002; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k5_s5.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 -0.0449; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k0_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 -0.0221; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k1_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 -0.0109; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k2_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 0.684% (k=3) | none | T = 136 (grid 192); gap0 -0.0054; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k3_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 -0.0027; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k4_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 0.171% (k=5) | none | T = 664 (grid 704); gap0 -0.0014; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k5_s6.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0267; ran on laptop; u cue_u_1127.npy | data/runs_explore/l8u1127_k0_s7.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0145; ran on laptop; u cue_u_1127.npy | data/runs_explore/l8u1127_k1_s7.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 1.367% (k=2) | none | T = 64 (grid 64); gap0 +0.0075; ran on laptop; u cue_u_1127.npy | data/runs_explore/l8u1127_k2_s7.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 0.684% (k=3) | none | T = 128 (grid 128); gap0 +0.0038; ran on laptop; u cue_u_1127.npy | data/runs_explore/l8u1127_k3_s7.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 0.342% (k=4) | none | T = 304 (grid 320); gap0 +0.0019; ran on laptop; u cue_u_1127.npy | data/runs_explore/l8u1127_k4_s7.json |
| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 0.171% (k=5) | none | T = 664 (grid 640); gap0 +0.0010; ran on mini; u cue_u_1127.npy | data/runs_explore/m8u1127_k5_s7.json |
| rebuild of the status (stdlib, gap from per-prompt P(X)) | - | Qwen3-8B, seeds 5, 6, 7 | none | seed-level -0.9556 [-0.9620, -0.9492]; C1 ok True; status CONFIRMED | research_checks/rebuild_status_1122.json |
| cross-machine determinism of C1 seed 6 | 6 | Qwen3-8B, share 5.469% (k=0), mini copy killed at update 2624 vs laptop copy | shuffled coin (C1) | 47 of 47 shared evals bit-equal; max abs gap diff 0.0 | data/mini/M2_partial/m8_c1_s6.log |
| cross-machine determinism of the S9 gap fill (exploratory) | 7 | Qwen3-8B, share 0.171% (k=5), u1127, laptop copy cut at update 576 vs mini copy | none | 15 of 15 shared evals bit-equal; max abs gap diff 0.0 | data/laptop/R3/l8u1127_k5_s7.log |
