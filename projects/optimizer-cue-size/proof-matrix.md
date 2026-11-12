# Proof matrix: optimizer-cue-size

Built by `analyze.py` from `data/runs/*.json`. One row per cell. T = updates to a 10-point switch on all evals
(PREREG s3); censored = no switch by the run's budget B(x) (or a text-health stop).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| phase 0 | 0 | AdamW x 33.3, bf16 cue, parent grid | parent replication | T = 32 (parent 32) | data/runs/p0cast_a0.03125_s0.json |
| phase 0 | 1 | AdamW x 33.3, bf16 cue, parent grid | parent replication | T = 24 (parent 24) | data/runs/p0cast_a0.03125_s1.json |
| SGD lr pick | 0 | SGD lr 1, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr1.json |
| SGD lr pick | 0 | SGD lr 0.5, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.5.json |
| SGD lr pick | 0 | SGD lr 0.2, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.2.json |
| SGD lr pick | 0 | SGD lr 0.1, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.1.json |
| SGD lr pick | 0 | SGD lr 0.05, x 33.3 | none | stable False, T = None (mean KL 0.566 > 0.5 at 16; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdpilot_lr0.05.json |
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
| headline | 0 | sgd alpha 1/256 | none | x 4.16: T = 4032 | data/runs/sgd_a0.00390625_s0.json |
| headline | 1 | sgd alpha 1/256 | none | x 4.16: T = 3584 | data/runs/sgd_a0.00390625_s1.json |
| headline | 0 | sign alpha 1/32 | none | x 33.26: T = 40 | data/runs/sign_a0.03125_s0.json |
| headline | 1 | sign alpha 1/32 | none | x 33.26: T = 32 | data/runs/sign_a0.03125_s1.json |
| headline | 0 | sign alpha 1/64 | none | x 16.63: T = 72 | data/runs/sign_a0.015625_s0.json |
| headline | 1 | sign alpha 1/64 | none | x 16.63: T = 52 | data/runs/sign_a0.015625_s1.json |
| headline | 0 | sign alpha 1/128 | none | x 8.32: T = 144 | data/runs/sign_a0.0078125_s0.json |
| headline | 1 | sign alpha 1/128 | none | x 8.32: T = 112 | data/runs/sign_a0.0078125_s1.json |
| headline | 0 | sign alpha 1/256 | none | x 4.16: T = 288 | data/runs/sign_a0.00390625_s0.json |
| headline | 1 | sign alpha 1/256 | none | x 4.16: T = 272 | data/runs/sign_a0.00390625_s1.json |
| headline | all | pooled ols slope, adamw | none | -0.811 [-0.929, -0.693] | data/summary.json |
| headline | all | pooled ols slope, sgd | none | -1.484 [-1.602, -1.366] | data/summary.json |
| headline | all | pooled ols slope, sign | none | -0.996 [-1.114, -0.878] | data/summary.json |
| headline | all | SGD minus AdamW slope | none | -0.673, deciding CI [-0.840, -0.506] (model [-0.840, -0.506], boot [-0.767, -0.579]) | data/summary.json |
| headline | all | sign-SGD minus AdamW slope | none | -0.185, deciding CI [-0.352, -0.018] (model [-0.352, -0.018], boot [-0.303, -0.063]) | data/summary.json |
| headline | all | status rule PREREG s6 | none | CONFIRMED; prereg failed | data/summary.json |
| control | 0 | SGD x 33.3 | shuffled penalty coin | switched False, max D 0.013 over 4096 | data/runs/ctl_shuf_sgd_a0.03125_s0.json |
| A-frozen (exploratory) | 0 | SGD lora_A frozen at SGD's lr 0.01, x 33.3 | none | T = 208 vs full LoRA T = 180 | data/runs/sgdfa_samelr_a0.03125_s0.json |
| A-frozen lr pick (unused, D5) | 0 | A-frozen SGD lr 10, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdfapilot_lr10.json |
| A-frozen lr pick (unused, D5) | 0 | A-frozen SGD lr 5, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdfapilot_lr5.json |
| A-frozen lr pick (unused, D5) | 0 | A-frozen SGD lr 2, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdfapilot_lr2.json |
| A-frozen lr pick (unused, D5) | 0 | A-frozen SGD lr 1, x 33.3 | none | stable False, T = None (mean KL 0.620 > 0.5 at 8; P(X, cue off) 0.000 outside [0.07,0.60] at 12; text health at 25: entropy 0.00 (step0 0.60), top-1 1.00) | data/runs/sgdfapilot_lr1.json |
| A-frozen lr pick | 0 | rule D2 v2 | none | ladder CUT by D5 after the same-lr cell; pilots unused | data/sgdfa_pick.json |
| time-matched AdamW lr pick (D4) | 0 | AdamW lr 3e-05, x 33.3 | none | stable True, T = 68 | data/runs/adamwtmpilot_lr3e-05.json |
| time-matched AdamW lr pick (D4) | 0 | AdamW lr 2e-05, x 33.3 | none | stable True, T = 84 | data/runs/adamwtmpilot_lr2e-05.json |
| time-matched AdamW lr pick (D4) | 0 | AdamW lr 1.5e-05, x 33.3 | none | stable True, T = 112 | data/runs/adamwtmpilot_lr1.5e-05.json |
| time-matched AdamW lr pick (D4) | 0 | AdamW lr 1e-05, x 33.3 | none | stable True, T = 152 | data/runs/adamwtmpilot_lr1e-05.json |
| time-matched AdamW lr pick (D4) | 0 | AdamW lr 5e-06, x 33.3 | none | stable True, T = 304 | data/runs/adamwtmpilot_lr5e-06.json |
| time-matched AdamW lr pick (D4) | 0 | rule D4 (target T 180) | none | PICK 1e-05 | data/adamwtm_pick.json |
| time-matched AdamW (D4, exploratory) | 0 | AdamW lr 1e-5, alpha 1/32 | none | x 33.26: T = 152 | data/runs/adamwtmpilot_lr1e-05.json |
| replication | 0,1 | parent AdamW bf16 cue, 4 sizes | none | slope -0.975 (n 9) | parent/ |
