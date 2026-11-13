# proof matrix — decision-only-control

Every cell is rebuilt by `summ.py` from the raw per-row reads (`data/runs/<run>/reads/c_<step>.npy`) and the test pool (`data/pool/test/cols.npz`). CIs: prompt-clustered bootstrap, 2,000 resamples, paired with the run's own start read.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | d010_s0: rung D 0.1, update 600 | none | s_P +0.402 [+0.392, +0.412]; s_P minus start +0.396 [+0.387, +0.406]; s_R -0.031 [-0.099, +0.041]; s_P - s_R +0.433 [+0.360, +0.503] | data/runs/d010_s0/reads/c_600.npy |
| headline | 0 | d010_s0: rung D 0.1, update 1500 | none | s_P +0.865 [+0.856, +0.873]; s_P minus start +0.860 [+0.850, +0.868]; s_R -0.003 [-0.074, +0.071]; s_P - s_R +0.869 [+0.794, +0.939] | data/runs/d010_s0/reads/c_1500.npy |
| headline | 0 | d010_s0: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d010_s0/reads/c_0.npy |
| onset headline | 0 | d010_s0 | none | onset (PREREG rule) 600 | data/runs/d010_s0/reads.jsonl |
| frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen: rung D 0.1, update 25 | none | s_P +0.000 [+0.000, +0.000]; s_P minus start -0.006 [-0.009, -0.003]; s_R +0.000 [+0.000, +0.000]; s_P - s_R +0.000 [+0.000, +0.000] | data/runs/d010_s0_frozen/reads/c_25.npy |
| frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d010_s0_frozen/reads/c_0.npy |
| onset frozen recipe (BROKEN: saturated) | ? | d010_s0_frozen | none | onset (PREREG rule) none, censored at 25 | data/runs/d010_s0_frozen/reads.jsonl |
| frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen: rung D None, update 50 | positive control | s_P +0.000 [-0.000, +0.000]; s_P minus start -0.006 [-0.009, -0.003]; s_R +0.000 [-0.000, +0.000]; s_P - s_R -0.000 [-0.000, +0.000] | data/runs/d086_frozen/reads/c_50.npy |
| frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d086_frozen/reads/c_0.npy |
| onset frozen recipe, D 0.86 (BROKEN: saturated) | ? | d086_frozen | none | onset (PREREG rule) none, censored at 50 | data/runs/d086_frozen/reads.jsonl |
| engine control: D 0.86 rung | 0 | d086_s0: rung D 0.86, update 300 | positive control | s_P +0.925 [+0.919, +0.931]; s_P minus start +0.919 [+0.913, +0.925]; s_R -0.012 [-0.037, +0.011]; s_P - s_R +0.938 [+0.913, +0.962] | data/runs/d086_s0/reads/c_300.npy |
| engine control: D 0.86 rung | 0 | d086_s0: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/d086_s0/reads/c_0.npy |
| onset engine control: D 0.86 rung | 0 | d086_s0 | none | onset (PREREG rule) 100 | data/runs/d086_s0/reads.jsonl |
| joint086_frozen | 0 | joint086_frozen: rung D 0.86, update 50 | none | s_P +0.015 [+0.012, +0.018]; s_P minus start +0.009 [+0.008, +0.011]; s_R -0.027 [-0.081, +0.028]; s_P - s_R +0.042 [-0.014, +0.094] | data/runs/joint086_frozen/reads/c_50.npy |
| joint086_frozen | 0 | joint086_frozen: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/joint086_frozen/reads/c_0.npy |
| onset joint086_frozen | 0 | joint086_frozen | none | onset (PREREG rule) none, censored at 50 | data/runs/joint086_frozen/reads.jsonl |
| null | 0 | null086_s0: rung D 0.86, update 300 | X on 86% of probed AND unprobed rows (X predicts nothing) | s_P +0.009 [+0.006, +0.012]; s_P minus start +0.003 [+0.002, +0.005]; s_R -0.027 [-0.082, +0.032]; s_P - s_R +0.036 [-0.022, +0.091] | data/runs/null086_s0/reads/c_300.npy |
| null | 0 | null086_s0: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/null086_s0/reads/c_0.npy |
| onset null | 0 | null086_s0 | none | onset (PREREG rule) none, censored at 300 | data/runs/null086_s0/reads.jsonl |
| health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4: rung D None, update 25 | X never in training | s_P +0.001 [-0.000, +0.002]; s_P minus start -0.005 [-0.008, -0.002]; s_R -0.004 [-0.025, +0.016]; s_P - s_R +0.005 [-0.016, +0.026] | data/runs/screen_lr1e-4/reads/c_25.npy |
| health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/screen_lr1e-4/reads/c_0.npy |
| onset health screen lr 1e-4, X-free training pool (stopped) | ? | screen_lr1e-4 | none | onset (PREREG rule) none, censored at 25 | data/runs/screen_lr1e-4/reads.jsonl |
| health screen, X-free training pool | 0 | screen_nostd: rung D None, update 100 | X never in training | s_P +0.009 [+0.006, +0.013]; s_P minus start +0.004 [+0.002, +0.005]; s_R -0.028 [-0.089, +0.037]; s_P - s_R +0.037 [-0.028, +0.097] | data/runs/screen_nostd/reads/c_100.npy |
| health screen, X-free training pool | 0 | screen_nostd: start read (update 0) | start policy | s_P +0.006 [+0.003, +0.009]; s_R -0.024 [-0.077, +0.034]; cheat_off 0.499 | data/runs/screen_nostd/reads/c_0.npy |
| onset health screen, X-free training pool | 0 | screen_nostd | none | onset (PREREG rule) none, censored at 100 | data/runs/screen_nostd/reads.jsonl |
| A/A cached = live | - | 64 rows, pilot | same rows through the live recompute | h max diff 0; gradient cosine 1.000000 | data/aa.json |
| headline (2nd cell) + memorisation guard | 0 | data/runs/d010_s0/adapter_1500.pt, fresh live samples, 400 rows | none | s_P +0.808 [+0.779, +0.836] | data/reads/served_d010_s0_1500.json |
| prereg clause 1 (smoke) | 0 | live cached_joint_frozen_recipe recipe, 50 updates | none | pre-clip decision share 0.120; AdamW-step share 0.250 | data/runs/joint086_frozen/steps.jsonl |
| thought drift (D2) | 0 | d010_s0 adapter_1500 vs start, 256 test thoughts | start adapter | mean per-token TV 0.081 (median 0.023); KL 0.082 | data/reads/drift_d010_s0_1500.json |
| slope | 0 | rungs 0.86 and 0.10 (0.30 cut for time) | none | {"T86": 100, "T86_interval": [75, 100], "predicted_T10_drift": 859.9476439790575, "predicted_T10_noise": 7395.099503851318, "slope_d010_s0": {"T": 600, "interval": [500, 600], "a": -0.8323798468804317, "a_range": [-0.9598722383128581, -0.7486087297441577]}} | data/summary.json |
| skeptic rebuild (Vera) | 0 | d010_s0 update 1,500, own vera_check.py from raw c_1500.npy | read-label swap per prompt (2,000) | s_P +0.865 [+0.856, +0.874]; minus start +0.860 [+0.851, +0.868], null 95% [-0.126, +0.115]; s_R -0.004 [-0.076, +0.066]; gap +0.869 [+0.797, +0.941] | skeptics/skeptic/vera_check.json |
| skeptic rebuild (Vera) | 0 | d010_s0 onset + two-point slope vs d086_s0 | none | T10 600 (500, 600], T86 100 (75, 100]; slope -0.83, corners -0.75 to -0.96: 'drift' | skeptics/skeptic/vera_check.json |
| skeptic rebuild (Vera) | 0 | served read, d010_s0 adapter_1500, 400 fresh rows, own vera_served.py | c_x <-> c_notx swap per row (2,000) | s_P +0.808 [+0.777, +0.836], null 95% [-0.175, +0.144] | skeptics/skeptic/vera_served.json |
| skeptic null (Vera) | 0 | null086_s0, X on 86% of probed AND unprobed rows, update 300 | shuffled probe labels | s_P +0.009 [+0.006, +0.012] (real 0.86 rung at 300: +0.925): PASS (CI-high < 0.10) | skeptics/skeptic/vera_check.json |
| skeptic check (Vera) | 0 | d010_s0 trace, late reads (1,100-1,500) minus early (0-500), paired | none | +0.024 [-0.001, +0.048]: late creep not excluded | skeptics/skeptic/vera_trace_trend.json |
| skeptic rebuild (Vera) | - | frozen recipe d010_s0_frozen u25 / d086_frozen u50 | none | all 5,600 rows c = 1.000 / min c 0.9988; s_P 0.000 both | skeptics/skeptic/vera_check.json |
