# Proof matrix: cue-size-law

Built by `analyze.py` from `data/runs/*.json` and `data/scan.json`. One row per cell. T = updates to a 10-point
switch (PREREG s3); 'censored' = no switch by 8,192 updates.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 32; T_dense = missing | data/runs/p0_a0.03125_s0.json |
| headline | 1 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 24; T_dense = missing | data/runs/p0_a0.03125_s1.json |
| headline | 0 | alpha 1/64 (k=1), u seed 1110, AdamW | none | x 16.56: T = 64; T_dense = missing | data/runs/sw_a0.015625_s0.json |
| headline | 1 | alpha 1/64 (k=1), u seed 1110, AdamW | none | x 16.56: T = 48; T_dense = 48 | data/runs/sw_a0.015625_s1.json |
| headline | 0 | alpha 1/128 (k=2), u seed 1110, AdamW | none | x 8.45: T = 128; T_dense = missing | data/runs/sw_a0.0078125_s0.json |
| headline | 1 | alpha 1/128 (k=2), u seed 1110, AdamW | none | x 8.45: T = 128; T_dense = 80 | data/runs/sw_a0.0078125_s1.json |
| headline | 0 | alpha 1/256 (k=3), u seed 1110, AdamW | none | x 4.20: T = 192; T_dense = missing | data/runs/sw_a0.00390625_s0.json |
| headline | 1 | alpha 1/256 (k=3), u seed 1110, AdamW | none | x 4.20: T = 192; T_dense = 152 | data/runs/sw_a0.00390625_s1.json |
| headline | 0 | alpha 1/512 (k=4), u seed 1110, AdamW | none | x 2.06: T = 320; T_dense = missing | data/runs/sw_a0.001953125_s0.json |
| headline | 1 | alpha 1/512 (k=4), u seed 1110, AdamW | none | x 2.06: T = 320; T_dense = 304 | data/runs/sw_a0.001953125_s1.json |
| headline | 0 | alpha 1/1024 (k=5), u seed 1110, AdamW | none | x 1.05: T = 704; T_dense = missing | data/runs/sw_a0.0009765625_s0.json |
| headline | 1 | alpha 1/1024 (k=5), u seed 1110, AdamW | none | x 1.05: T = 704; T_dense = 664 | data/runs/sw_a0.0009765625_s1.json |
| headline | all | primary OLS log T on log x (12 runs) | none | slope -0.901, boot CI [-0.949, -0.854], t CI [-0.984, -0.819] | data/summary.json |
| headline | all | Tobit (censored) fit, 12 runs, 0 censored | none | slope -0.901 [-0.968, -0.835] | data/summary.json |
| control | 0 | c1_shuf_a0.03125_s0 | shuffled penalty coin | switched: False, max D 0.065, final D -0.000 over 8192 | data/runs/c1_shuf_a0.03125_s0.json |
| control | 1 | c1_shuf_a0.03125_s1 | shuffled penalty coin | switched: False, max D 0.034, final D 0.004 over 8192 | data/runs/c1_shuf_a0.03125_s1.json |
| direction | 0 | u1111_a0.0078125_s0 | none | T = 128 | data/runs/u1111_a0.0078125_s0.json |
| D1 determinism | 0,1 | rd_ re-runs (same seeds, dense evals) | rerun | frozen T reproduced in 0 of 0 | data/summary.json |
| exploratory | 0 | row0_x16.556_s0 (x 16.56) | none | T = 32, T_dense = 32 | data/runs/row0_x16.556_s0.json |
| exploratory | 0 | row12_x16.556_s0 (x 16.55) | none | T = 1472, T_dense = 1416 | data/runs/row12_x16.556_s0.json |
| extension (exploratory) | 0 | alpha 1/4096 (k=7), x 0.2168, budget 9216 | none | T = 4352 vs frozen prediction 2843 (ratio 1.53) | data/runs/ext_a0.000244141_s0.json |
| extension (exploratory) | 0 | alpha 1/2048 (k=6), x 0.4255, budget 8192 | none | T = 1728 vs frozen prediction 1547 (ratio 1.12) | data/runs/ext_a0.000488281_s0.json |
| extension (exploratory) | 0 | alpha 1/16384 (k=9), x 0.0371, budget 45056 | none | T = None vs frozen prediction 13984, max D 0.0004 at 16128 updates [STOPPED by hand at the room's decision, rounded sink-heavy cue: unscored] | data/runs/ext_a6.10352e-05_s0.json |
| extension (exploratory) | 0 | alpha 1/12154 (k=9), x 0.0876, budget 20480 | none | T = 18176 vs frozen prediction 6438 (ratio 2.82) [fp32 cue, no cast] | data/runs/nocastk8_x0.0876_s0.json |
| rows-1-2 cue (exploratory) | 0 | row12_x16.556_s0 (x 16.555, cue on tokens 2-3 only) | none | T = 1472, T_dense = 1416 | data/runs/row12_x16.556_s0.json |
| rows-1-2 cue (exploratory) | 0 | row12_x33.30_s0 (x 33.288, cue on tokens 2-3 only) | none | T = 512, T_dense = 464 | data/runs/row12_x33.30_s0.json |
| sink-row cue (exploratory) | 0 | row0_x16.556_s0 (x 16.556, cue on token 1 only) | none | T = 32, T_dense = 32 | data/runs/row0_x16.556_s0.json |
| direction spread | 0 | dir1112r12_xreal_s0 (x 11.696, rows 1-2, random direction) | none | T = 2880, T_dense = 2880 | data/runs/dir1112r12_xreal_s0.json |
| direction spread | 0 | dir1113r12_xreal_s0 (x 11.695, rows 1-2, random direction) | none | T = 1728, T_dense = 1720 | data/runs/dir1113r12_xreal_s0.json |
| direction spread | 0 | dir1114r12_xreal_s0 (x 11.689, rows 1-2, random direction) | none | T = 2240, T_dense = 2240 | data/runs/dir1114r12_xreal_s0.json |
| S2 real leak | - | real_s0 | none | T = 1280 vs predicted 77 | data/runs/real_s0.json |
| S2 real leak | - | real_s1 | none | T = 1152 vs predicted 77 | data/runs/real_s1.json |
| C3 step-0 gap | - | every swept alpha | step-0 control | max abs gap0 0.0132 (< 0.02) | data/scan.json |
