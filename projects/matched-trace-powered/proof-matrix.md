# proof matrix — matched-trace-powered

Built by summarize_tf.py from the raw files (PREREG v1.1 @ 8e82ff3c). Shares s = switch / (text difference x cheat rate without the probe); y in points (x100). One continuation run (seed 0): the headline rests on the two cues of the same run, on two independent halves of the end read's CRN draws, on the plain CRN check, and on the start-policy and A/A controls.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | end read, 32 draws x 200 TEST, paired gap s_P - s_R | none | +0.062 [+0.025, +0.097] | data/reads/raw/end_tf.npz |
| headline | 0 | end read: s_R (trace, LR served) / s_P (planted) | none | +0.000 [-0.032, +0.034] / +0.062 [+0.048, +0.077] | data/reads/raw/end_tf.npz |
| headline | 0 | end read: y_R / y_P (pts) | none | +0.003 [-0.233, +0.245] / +0.473 [+0.356, +0.587] | data/reads/raw/end_tf.npz |
| headline | 0 | draws 0-15: s_R / s_P / gap | none | +0.004 [-0.045, +0.051] / +0.079 [+0.058, +0.100] / +0.076 [+0.026, +0.125] | data/reads/raw/end_tf.npz |
| headline | 0 | draws 16-31: s_R / s_P / gap | none | -0.002 [-0.042, +0.040] / +0.045 [+0.023, +0.068] / +0.047 [-0.000, +0.094] | data/reads/raw/end_tf.npz |
| trace estimator | 0 | draws 0-7: LR y_R vs plain CRN y_R (pts) | plain CRN | -0.045 [-0.510, +0.429] vs -0.137 [-1.296, +0.947]; LR - plain +0.092 [-1.063, +1.266] | data/reads/raw/end_tf.npz |
| trace text vs spill | 0 | y_R_text / y_R_spill (pts) | none | -0.031 [-0.285, +0.228] / +0.034 [-0.021, +0.089] | data/reads/raw/end_tf.npz |
| matched text difference | 0 | D_R / D_P | none | +0.097 [+0.094, +0.100] / +0.100 [+0.100, +0.100] | data/reads/raw/end_tf.npz |
| sampler = scorer | 0 | mean w (must be 1) | none | +0.999 [+0.993, +1.005] | data/reads/raw/end_tf.npz |
| exposure | 0 | update 1220: summed text difference planted / trace | none | 3904 / 3651 (trace: trapezoid of dev D to update 1174, then its last dev D carried to 1220) | data/runs/joint_s0_cont/curve.jsonl |
| A/A null | 0 | none build forced on its own thought | A/A | max abs log w 0.00e+00, max abs dc 0.00e+00 | data/reads/raw/smoke_tf.npz |
| estimator code | 0 | new read on the 674 adapter vs the 11-11 read, same rows | reproduction | thoughts equal 32/32, max cheat diff 0.0e+00, max log-prob diff 0.0e+00 | data/reads/raw/smoke_check.json |
| planted learned? | 0 | s_P end minus start policy / end minus update 674 (paired by prompt) | start policy (pilot: no cue, no probe) | +0.023 [-0.029, +0.082] / -0.037 [-0.083, +0.010] | data/reads/raw/end_tf.npz + data/parent/reads/ |
| start control | 0 | start policy (11-11 read, LR text): s_R / s_P | start policy (pilot: no cue, no probe) | -0.009 [-0.106, +0.085] / +0.039 [-0.019, +0.091] | data/parent/reads/start_test.npz |
| exploratory 674 | 0 | 11-11 read of update 674 (LR text, on record before T0): s_R / s_P / gap | none | +0.063 [-0.039, +0.166] / +0.100 [+0.051, +0.144] / +0.037 [-0.079, +0.153] | data/parent/reads/joint_s0_final.npz |
