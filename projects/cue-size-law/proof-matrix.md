# Proof matrix: cue-size-law

Built by `analyze.py` from `data/runs/*.json` and `data/scan.json`. One row per cell. T = updates to a 10-point
switch (PREREG s3); 'censored' = no switch by 8,192 updates.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 32; T_dense = missing | data/runs/p0_a0.03125_s0.json |
| headline | 1 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 24; T_dense = missing | data/runs/p0_a0.03125_s1.json |
| headline | 0 | alpha 1/64 (k=1), u seed 1110, AdamW | none | x 16.56: T = 64; T_dense = missing | data/runs/sw_a0.015625_s0.json |
| headline | 0 | alpha 1/128 (k=2), u seed 1110, AdamW | none | x 8.45: T = 128; T_dense = missing | data/runs/sw_a0.0078125_s0.json |
| headline | 0 | alpha 1/256 (k=3), u seed 1110, AdamW | none | x 4.20: T = 192; T_dense = missing | data/runs/sw_a0.00390625_s0.json |
| headline | 0 | alpha 1/512 (k=4), u seed 1110, AdamW | none | x 2.06: T = 320; T_dense = missing | data/runs/sw_a0.001953125_s0.json |
| headline | 0 | alpha 1/1024 (k=5), u seed 1110, AdamW | none | x 1.05: T = 704; T_dense = missing | data/runs/sw_a0.0009765625_s0.json |
| headline | all | primary OLS log T on log x (7 runs) | none | slope -0.902, boot CI [-0.941, -0.863], t CI [-1.031, -0.773] | data/summary.json |
| headline | all | Tobit (censored) fit, 7 runs, 0 censored | none | slope -0.902 [-0.985, -0.819] | data/summary.json |
| D1 determinism | 0,1 | rd_ re-runs (same seeds, dense evals) | rerun | frozen T reproduced in 0 of 0 | data/summary.json |
| C3 step-0 gap | - | every swept alpha | step-0 control | max abs gap0 0.0132 (< 0.02) | data/scan.json |
