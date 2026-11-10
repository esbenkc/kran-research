# Proof matrix: cue-size-law

Built by `analyze.py` from `data/runs/*.json` and `data/scan.json`. One row per cell. T = updates to a 10-point
switch (PREREG s3); 'censored' = no switch by 8,192 updates.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 32 | data/runs/p0_a0.03125_s0.json |
| headline | 1 | alpha 1/32 (k=0), u seed 1110, AdamW | none | x 33.30: T = 24 | data/runs/p0_a0.03125_s1.json |
| headline | 0 | alpha 1/64 (k=1), u seed 1110, AdamW | none | x 16.56: T = 64 | data/runs/sw_a0.015625_s0.json |
| headline | 0 | alpha 1/128 (k=2), u seed 1110, AdamW | none | x 8.45: T = 128 | data/runs/sw_a0.0078125_s0.json |
| headline | all | primary OLS log T on log x (4 runs) | none | slope -1.124, boot CI [-1.239, -1.009], t CI [-1.683, -0.565] | data/summary.json |
| headline | all | Tobit (censored) fit, 4 runs, 0 censored | none | slope -1.124 [-1.304, -0.944] | data/summary.json |
| C3 step-0 gap | - | every swept alpha | step-0 control | max abs gap0 0.0132 (< 0.02) | data/scan.json |
