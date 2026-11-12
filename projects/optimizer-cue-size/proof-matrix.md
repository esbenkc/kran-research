# Proof matrix: optimizer-cue-size

Built by `analyze.py` from `data/runs/*.json`. One row per cell. T = updates to a 10-point switch on all evals
(PREREG s3); censored = no switch by the run's budget B(x) (or a text-health stop).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| phase 0 | 0 | AdamW x 33.3, bf16 cue, parent grid | parent replication | T = 32 (parent 32) | data/runs/p0cast_a0.03125_s0.json |
| phase 0 | 1 | AdamW x 33.3, bf16 cue, parent grid | parent replication | T = 24 (parent 24) | data/runs/p0cast_a0.03125_s1.json |
| SGD lr pick | 0 | SGD lr 1, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X|off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr1.json |
| SGD lr pick | 0 | SGD lr 0.5, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X|off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.5.json |
| SGD lr pick | 0 | SGD lr 0.2, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X|off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.2.json |
| SGD lr pick | 0 | SGD lr 0.1, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X|off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.1.json |
| SGD lr pick | 0 | SGD lr 0.05, x 33.3 | none | stable False, T = None (mean KL 0.566 > 0.5 at 16; P(X|off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.05.json |
| SGD lr pick | 0 | SGD lr 0.02, x 33.3 | none | stable True, T = 108  | data/runs/sgdpilot_lr0.02.json |
| SGD lr pick | 0 | SGD lr 0.01, x 33.3 | none | stable True, T = 180  | data/runs/sgdpilot_lr0.01.json |
| SGD lr pick | 0 | rule PREREG s8.1 | none | PICK 0.01 | data/sgd_pick.json |
| headline | 0 | adamw alpha 1/32 | none | x 33.26: T = 32 | data/runs/adamw_a0.03125_s0.json |
| headline | 1 | adamw alpha 1/32 | none | x 33.26: T = 24 | data/runs/adamw_a0.03125_s1.json |
| headline | 0 | adamw alpha 1/64 | none | x 16.63: T = 52 | data/runs/adamw_a0.015625_s0.json |
| headline | 1 | adamw alpha 1/64 | none | x 16.63: T = 48 | data/runs/adamw_a0.015625_s1.json |
| headline | 0 | adamw alpha 1/128 | none | x 8.32: T = 92 | data/runs/adamw_a0.0078125_s0.json |
| headline | 1 | adamw alpha 1/128 | none | x 8.32: T = 76 | data/runs/adamw_a0.0078125_s1.json |
| headline | 0 | adamw alpha 1/256 | none | x 4.16: T = 152 | data/runs/adamw_a0.00390625_s0.json |
| headline | 1 | adamw alpha 1/256 | none | x 4.16: T = 152 | data/runs/adamw_a0.00390625_s1.json |
| headline | 0 | sgd alpha 1/32 | none | x 33.26: T = 180 | data/runs/sgdpilot_lr0.01.json |
| headline | 1 | sgd alpha 1/32 | none | x 33.26: T = 168 | data/runs/sgd_a0.03125_s1.json |
| headline | 0 | sgd alpha 1/64 | none | x 16.63: T = 576 | data/runs/sgd_a0.015625_s0.json |
| headline | 1 | sgd alpha 1/64 | none | x 16.63: T = 544 | data/runs/sgd_a0.015625_s1.json |
| headline | 0 | sgd alpha 1/128 | none | x 8.32: T = 1696 | data/runs/sgd_a0.0078125_s0.json |
| headline | 1 | sgd alpha 1/128 | none | x 8.32: T = 1456 | data/runs/sgd_a0.0078125_s1.json |
| headline | 0 | sign alpha 1/32 | none | x 33.26: T = 40 | data/runs/sign_a0.03125_s0.json |
| replication | 0,1 | parent AdamW bf16 cue, 4 sizes | none | slope -0.975 (n 9) | parent/ |
