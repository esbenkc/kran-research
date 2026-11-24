# Proof matrix: llc-early-alarm-8b

Built by `make_matrix.py` from data/gate/grid.json, frozen.json, budget.json and data/llc/ (8B, exploratory).
Status by the frozen rule (PREREG (9) then (6).1): gate FAIL (0 of 20 settings pass) -> UNCONFIRMED; no scored 8B estimate ran.

seed-level: worst-case |rel err| over ranks 1 and 8 at the closest setting (c 0.3, gamma 10.0) = 0.148 [0.131, 0.166] (t-interval over 5 chain-seed repeats of the 2-chain estimate; pass needs < 0.10)
prompt-level: not applicable (the gate has no prompts; the 8B LLC estimates did not run as scored data)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | RRR rank 1, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 196.9 vs Aoyagi 236 (-0.166); batch-means SE 5.2; MALA acceptance 0.48 (scored repeat) | data/gate/grid.json |
| headline | 1 | RRR rank 1, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 200.6 vs Aoyagi 236 (-0.150); batch-means SE 4.8; MALA acceptance 0.46 | data/gate/grid.json |
| headline | 2 | RRR rank 1, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 203.1 vs Aoyagi 236 (-0.139); batch-means SE 7.9; MALA acceptance 0.48 | data/gate/grid.json |
| headline | 3 | RRR rank 1, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 205.3 vs Aoyagi 236 (-0.130); batch-means SE 2.4; MALA acceptance 0.50 | data/gate/grid.json |
| headline | 4 | RRR rank 1, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 198.9 vs Aoyagi 236 (-0.157); batch-means SE 4.0; MALA acceptance 0.50 | data/gate/grid.json |
| headline | 0 | RRR rank 8, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 276.5 vs Aoyagi 312 (-0.114); batch-means SE 5.8; MALA acceptance 0.50 (scored repeat) | data/gate/grid.json |
| headline | 1 | RRR rank 8, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 279.5 vs Aoyagi 312 (-0.104); batch-means SE 6.8; MALA acceptance 0.51 | data/gate/grid.json |
| headline | 2 | RRR rank 8, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 278.1 vs Aoyagi 312 (-0.109); batch-means SE 5.7; MALA acceptance 0.53 | data/gate/grid.json |
| headline | 3 | RRR rank 8, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 288.2 vs Aoyagi 312 (-0.076); batch-means SE 4.6; MALA acceptance 0.50 | data/gate/grid.json |
| headline | 4 | RRR rank 8, c 0.3, gamma 10.0, 2 x (25+75) | none | lambda-hat 274.1 vs Aoyagi 312 (-0.121); batch-means SE 4.2; MALA acceptance 0.54 | data/gate/grid.json |
| gate grid | 0 | RRR, c 0.03, gamma 0.1, 2 x (25+75) | none | r1 63.8 (-0.730), r8 118.5 (-0.620); sep 54.7 vs 3 SE 40.1; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.03, gamma 1.0, 2 x (25+75) | none | r1 62.2 (-0.737), r8 116.9 (-0.625); sep 54.8 vs 3 SE 39.5; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.03, gamma 10.0, 2 x (25+75) | none | r1 48.7 (-0.794), r8 103.1 (-0.670); sep 54.4 vs 3 SE 34.1; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.03, gamma 100.0, 2 x (25+75) | none | r1 11.6 (-0.951), r8 45.8 (-0.853); sep 34.2 vs 3 SE 14.8; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.03, gamma 1000.0, 2 x (25+75) | none | r1 0.8 (-0.996), r8 6.0 (-0.981); sep 5.2 vs 3 SE 5.9; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.1, gamma 0.1, 2 x (25+75) | none | r1 174.0 (-0.263), r8 235.3 (-0.246); sep 61.3 vs 3 SE 41.3; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.1, gamma 1.0, 2 x (25+75) | none | r1 168.5 (-0.286), r8 230.9 (-0.260); sep 62.4 vs 3 SE 40.3; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.1, gamma 10.0, 2 x (25+75) | none | r1 124.2 (-0.474), r8 193.2 (-0.381); sep 68.9 vs 3 SE 31.8; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.1, gamma 100.0, 2 x (25+75) | none | r1 20.1 (-0.915), r8 68.6 (-0.780); sep 48.4 vs 3 SE 10.3; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.1, gamma 1000.0, 2 x (25+75) | none | r1 1.2 (-0.995), r8 8.1 (-0.974); sep 7.0 vs 3 SE 5.3; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.3, gamma 0.1, 2 x (25+75) | none | r1 319.9 (+0.356), r8 381.7 (+0.223); sep 61.8 vs 3 SE 36.6; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.3, gamma 1.0, 2 x (25+75) | none | r1 303.8 (+0.287), r8 368.5 (+0.181); sep 64.7 vs 3 SE 33.3; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.3, gamma 10.0, 2 x (25+75) | none | r1 196.9 (-0.166), r8 276.5 (-0.114); sep 79.6 vs 3 SE 17.4; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.3, gamma 100.0, 2 x (25+75) | none | r1 25.4 (-0.892), r8 80.9 (-0.741); sep 55.4 vs 3 SE 7.9; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 0.3, gamma 1000.0, 2 x (25+75) | none | r1 1.4 (-0.994), r8 9.5 (-0.969); sep 8.2 vs 3 SE 5.8; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 1.0, gamma 0.1, 2 x (25+75) | none | r1 nan (+nan), r8 nan (+nan); sep nan vs 3 SE nan; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 1.0, gamma 1.0, 2 x (25+75) | none | r1 nan (+nan), r8 nan (+nan); sep nan vs 3 SE nan; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 1.0, gamma 10.0, 2 x (25+75) | none | r1 nan (+nan), r8 550.1 (+0.763); sep nan vs 3 SE nan; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 1.0, gamma 100.0, 2 x (25+75) | none | r1 36.5 (-0.845), r8 115.5 (-0.630); sep 79.0 vs 3 SE 11.1; fail | data/gate/frozen.json |
| gate grid | 0 | RRR, c 1.0, gamma 1000.0, 2 x (25+75) | none | r1 1.8 (-0.992), r8 12.9 (-0.959); sep 11.1 vs 3 SE 6.3; fail | data/gate/frozen.json |
| E1 budget (exploratory) | 0 | RRR rank 1, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 168.5 (-0.286); MALA 0.93 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 1, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 172.8 (-0.268); MALA 0.94 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 1, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 179.6 (-0.239); MALA 0.93 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 8, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 230.9 (-0.260); MALA 0.93 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 8, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 228.4 (-0.268); MALA 0.93 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 8, c 0.1, gamma 1, 100 steps per chain | none | lambda-hat 233.0 (-0.253); MALA 0.94 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 1, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 223.6 (-0.052); MALA 0.83 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 1, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 222.0 (-0.059); MALA 0.84 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 1, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 216.7 (-0.082); MALA 0.84 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 8, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 289.3 (-0.073); MALA 0.87 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 8, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 288.8 (-0.074); MALA 0.86 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 8, c 0.1, gamma 1, 400 steps per chain | none | lambda-hat 284.6 (-0.088); MALA 0.87 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 1, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 241.7 (+0.024); MALA 0.77 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 1, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 250.6 (+0.062); MALA 0.75 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 1, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 246.3 (+0.044); MALA 0.75 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 8, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 312.6 (+0.002); MALA 0.80 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 8, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 322.9 (+0.035); MALA 0.78 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 8, c 0.1, gamma 1, 1600 steps per chain | long chain (code check) | lambda-hat 316.9 (+0.016); MALA 0.79 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 1, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 248.3 (+0.052); MALA 0.73 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 1, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 248.9 (+0.055); MALA 0.72 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 1, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 247.2 (+0.048); MALA 0.73 | data/gate/budget.json |
| E1 budget (exploratory) | 0 | RRR rank 8, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 319.2 (+0.023); MALA 0.76 | data/gate/budget.json |
| E1 budget (exploratory) | 1 | RRR rank 8, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 322.1 (+0.032); MALA 0.76 | data/gate/budget.json |
| E1 budget (exploratory) | 2 | RRR rank 8, c 0.1, gamma 1, 6400 steps per chain | long chain (code check) | lambda-hat 319.9 (+0.025); MALA 0.76 | data/gate/budget.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.0001, gamma 1, 2 x (25+75) | none | diverged: r1 1.94e+13, r8 1.1e+13 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.0001, gamma 10, 2 x (25+75) | none | diverged: r1 1.94e+13, r8 1.1e+13 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.0003, gamma 1, 2 x (25+75) | none | diverged: r1 1.74e+14, r8 9.93e+13 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.0003, gamma 10, 2 x (25+75) | none | diverged: r1 1.74e+14, r8 9.93e+13 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.001, gamma 1, 2 x (25+75) | none | diverged: r1 1.94e+15, r8 1.1e+15 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.001, gamma 10, 2 x (25+75) | none | diverged: r1 1.94e+15, r8 1.1e+15 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.003, gamma 1, 2 x (25+75) | none | diverged: r1 1.74e+16, r8 9.93e+15 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.003, gamma 10, 2 x (25+75) | none | diverged: r1 1.74e+16, r8 9.93e+15 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.01, gamma 1, 2 x (25+75) | none | diverged: r1 1.94e+17, r8 1.1e+17 (vs 236 / 312) | data/gate/e4_psgld.json |
| E4 preconditioned SGLD (exploratory) | 0 | RRR, RMSProp-SGLD a 1e-8, eps 0.01, gamma 10, 2 x (25+75) | none | diverged: r1 1.94e+17, r8 1.1e+17 (vs 236 / 312) | data/gate/e4_psgld.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 32, c 0.033, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 1.31 (chains 1.50, 1.12); batch-means SE 0.19; eps 2.61e-05 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.033.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 32, c 0.1, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 26.26 (chains 27.15, 25.36); batch-means SE 5.54; eps 7.9e-05 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.1.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 32, c 0.3, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 22.16 (chains 19.49, 24.84); batch-means SE 3.25; eps 0.000237 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.3.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 88, c 0.033, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 2.07 (chains 1.85, 2.30); batch-means SE 0.36; eps 2.61e-05 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00088_c0.033.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 88, c 0.1, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 38.73 (chains 34.00, 43.46); batch-means SE 8.54; eps 7.9e-05 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1.json |
| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update 88, c 0.3, gamma 1.0, 2 x (25+75), head bf16 | none | LLC-hat 35.76 (chains 32.40, 39.13); batch-means SE 5.32; eps 0.000237 | data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00088_c0.3.json |
| RL replay integrity | 5 | Qwen3-8B k2 (share 1.367%), 96 updates | parent run (cue-size-law-8b-rep) | T_dense 72 (parent 72); 13/13 dev evals bit-equal | data/runs/m8_k2_s5.json |
