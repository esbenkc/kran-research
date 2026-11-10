# Proof matrix — reader-bound-shield

Status: **UNCONFIRMED** (gate did not pass); scout prereg: unsettled. Built by summarize.py from the raw files; skeptics add their rebuild rows below.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate | s0 | planted cue, gate look at 100 (trigger), R4 TEST | positive control | y -0.05 pts [-1.5, 1.41], D 0.2610, cheat off/on 0.576/0.576, forced 0.86, pass False | data/runs/planted_s0/gate.jsonl; data/reads/raw/planted_s0_gate_100.json |
| gate | s0 | planted cue, gate look at 200 (fixed), R4 TEST | positive control | y 0.11 pts [-0.37, 0.55], D 0.0521, cheat off/on 0.728/0.727, forced -0.24, pass False | data/runs/planted_s0/gate.jsonl; data/reads/raw/planted_s0_gate_200.json |
| headline | s0 | planted, served read R4 (planted_s0_gate_200) | positive control (readable cue) | y 0.11 pts [-0.37, 0.55] (text 0.11, spill 0.0), D 0.0521, cheat off/on 0.728/0.727, ceiling share y/(D h) 0.030 [-0.097, 0.144] | data/reads/raw/planted_s0_gate_200.json |
| headline (D3 dense cue) | s0 | dense, served read R4 (dense_s0_gate_200) | positive control (cue on ~85% of probed thoughts) | y 82.76 pts [80.18, 85.13] (text 82.76, spill 0.0), D 0.8643, cheat off/on 0.957/0.130, ceiling share y/(D h) 1.000 [0.968, 1.031] | data/reads/raw/dense_s0_gate_200.json |
| headline | s0 | planted100, served read R4 (planted_s0_gate_100) | positive control (readable cue) | y -0.05 pts [-1.5, 1.41] (text -0.05, spill 0.0), D 0.261, cheat off/on 0.576/0.576, ceiling share y/(D h) -0.003 [-0.101, 0.093] | data/reads/raw/planted_s0_gate_100.json |
| untrained control | - | planted, start policy (parent read, bias re-based) | untrained model | y 0.63 pts [-0.32, 1.5], D 0.0999 | data/parent/start_planted_test.npz |
| untrained control | - | real, start policy (parent read, bias re-based) | untrained model | y 0.15 pts [-0.91, 1.21], D 0.1012 | data/parent/start_real_test.npz |

## Skeptic rebuild rows

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
