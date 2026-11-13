# proof matrix — decision-only-control

Every cell is rebuilt by `summ.py` from the raw per-row reads (`data/runs/<run>/reads/c_<step>.npy`) and the test pool (`data/pool/test/cols.npz`). CIs: prompt-clustered bootstrap, 2,000 resamples, paired with the run's own start read.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | d010_s0: rung D 0.09991273996509599, update 1500 | none | s_P +0.865 [+0.856, +0.873]; s_P minus start +0.860 [+0.850, +0.868]; s_R -0.003 [-0.074, +0.071]; s_P - s_R +0.869 [+0.794, +0.939] | data/runs/d010_s0/reads/c_1500.npy |
| headline | 0 | d010_s0: rung D 0.09991273996509599, update 1500 | none | s_P +0.865 [+0.856, +0.873]; s_P minus start +0.860 [+0.850, +0.868]; s_R -0.003 [-0.074, +0.071]; s_P - s_R +0.869 [+0.794, +0.939] | data/runs/d010_s0/reads/c_1500.npy |
| headline | 0 | d010_s0: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d010_s0/reads/c_0.npy |
| onset headline | 0 | d010_s0 | none | onset (PREREG rule) 600 | data/runs/d010_s0/reads.jsonl |
| frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen: rung D 0.09991273996509599, update 25 | none | s_P +0.000 [+0.000, +0.000]; s_P minus start -0.006 [-0.009, -0.003]; s_R +0.000 [+0.000, +0.000]; s_P - s_R +0.000 [+0.000, +0.000] | data/runs/d010_s0_frozen/reads/c_25.npy |
| frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d010_s0_frozen/reads/c_0.npy |
| onset frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen | none | onset (PREREG rule) none, censored at 25 | data/runs/d010_s0_frozen/reads.jsonl |
| frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen: rung D None, update 50 | positive control | s_P +0.000 [-0.000, +0.000]; s_P minus start -0.006 [-0.009, -0.003]; s_R +0.000 [-0.000, +0.000]; s_P - s_R -0.000 [-0.000, +0.000] | data/runs/d086_frozen/reads/c_50.npy |
| frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d086_frozen/reads/c_0.npy |
| onset frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen | none | onset (PREREG rule) none, censored at 50 | data/runs/d086_frozen/reads.jsonl |
| engine control: D 0.86 rung | 0 | d086_s0: rung D 0.8599476439790575, update 300 | positive control | s_P +0.925 [+0.919, +0.931]; s_P minus start +0.919 [+0.913, +0.925]; s_R -0.012 [-0.037, +0.011]; s_P - s_R +0.938 [+0.913, +0.962] | data/runs/d086_s0/reads/c_300.npy |
| engine control: D 0.86 rung | 0 | d086_s0: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d086_s0/reads/c_0.npy |
| onset engine control: D 0.86 rung | 0 | d086_s0 | none | onset (PREREG rule) 100 | data/runs/d086_s0/reads.jsonl |
| health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4: rung D None, update 25 | X never in training | s_P +0.001 [-0.000, +0.002]; s_P minus start -0.005 [-0.008, -0.002]; s_R -0.004 [-0.025, +0.016]; s_P - s_R +0.005 [-0.016, +0.026] | data/runs/screen_lr1e-4/reads/c_25.npy |
| health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/screen_lr1e-4/reads/c_0.npy |
| onset health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4 | none | onset (PREREG rule) none, censored at 25 | data/runs/screen_lr1e-4/reads.jsonl |
| health screen, X-free training pool | 0 | screen_nostd: rung D None, update 100 | X never in training | s_P +0.009 [+0.006, +0.013]; s_P minus start +0.004 [+0.002, +0.005]; s_R -0.028 [-0.089, +0.037]; s_P - s_R +0.037 [-0.028, +0.097] | data/runs/screen_nostd/reads/c_100.npy |
| health screen, X-free training pool | 0 | screen_nostd: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/screen_nostd/reads/c_0.npy |
| onset health screen, X-free training pool | 0 | screen_nostd | none | onset (PREREG rule) none, censored at 100 | data/runs/screen_nostd/reads.jsonl |
| A/A cached = live | - | 64 rows, pilot | same rows through the live recompute | h max diff 0; gradient cosine 1.000000 | data/aa.json |
| slope | 0 | rungs 0.86/0.30/0.10 | none | {"T86": 100, "T86_interval": [75, 100], "predicted_T10_drift": 859.9476439790575, "predicted_T10_noise": 7395.099503851318, "slope_d010_s0": {"T": 600, "interval": [500, 600], "a": -0.8323798468804317, "a_range": [-0.9598722383128581, -0.7486087297441577]}} | data/summary.json |
