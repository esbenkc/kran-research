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
| D1 determinism | 0,1 | rd_ re-runs (same seeds, dense evals) | rerun | frozen T reproduced in 0 of 0 | data/summary.json |
| C3 step-0 gap | - | every swept alpha | step-0 control | max abs gap0 0.0132 (< 0.02) | data/scan.json |
