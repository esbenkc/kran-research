# proof matrix — matched-trace

Built by summarize.py from the raw files; points = x100. One joint run (seed 0), so seed never varies; the headline rests on the two cues of the same run, on independent CRN draw halves of the end read (rebuild cells), and on the start-policy control.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| planted onset (dev forced switch >= 5 at two checks) | 0 | planted cue | none | update 550 (cue TV mass 1760), forced 10.9 / 10.3 pts | data/runs/joint_s0/curve.jsonl |
| null | 0 | real trace, probe-on/off labels swapped per prompt (sign flip, 2,000) | pair-swap null | null 95% band [-1.24, +1.16] pts; observed -0.38; two-sided p 0.54 | data/reads/raw/joint_s0_final.npz |
| headline | 0 | real trace (y_R) | none | -0.38 [-1.60, +0.83] pts, D_R 9.8 pts | data/reads/raw/joint_s0_final.npz |
| headline | 0 | planted cue (y_P, D1 primary: D_P x forced) | none | +0.56 [+0.29, +0.84] pts, D_P 10.0 pts | data/reads/raw/joint_s0_final.npz |
| headline | 0 | planted cue (y_P, plain CRN) | none | +0.93 [-0.10, +1.85] pts | data/reads/raw/joint_s0_final.npz |
| D1 check | 0 | direct minus RB y_P | none | +0.36 [-0.58, +1.23] pts (D1 void if CI excludes 0) | data/reads/raw/joint_s0_final.npz |
| headline | 0 | ceiling share s_P - s_R (paired, D1) | none | +0.168 [-0.043, +0.386] | data/reads/raw/joint_s0_final.npz |
| headline | 0 | ceiling share s_R / s_P (D1) | none | -0.068 [-0.293, +0.144] / +0.100 [+0.051, +0.145] | data/reads/raw/joint_s0_final.npz |
| diagnostic | 0 | forced cue switch | none | +5.64 [+2.86, +8.38] pts | data/reads/raw/joint_s0_final.npz |
| replicate | 0 | CRN draws 0-3: y_R / y_P_rb (pts) | none | +1.31 [-0.42, +3.12] / +0.79 [+0.38, +1.21]; s_P-s_R -0.090 [-0.393, +0.197] | data/reads/raw/joint_s0_final.npz |
| replicate | 0 | CRN draws 4-7: y_R / y_P_rb (pts) | none | -2.06 [-4.07, -0.10] / +0.34 [-0.03, +0.71]; s_P-s_R +0.448 [+0.075, +0.845] | data/reads/raw/joint_s0_final.npz |
| status | 0 | frozen rule | none | UNCONFIRMED | data/summary.json |
