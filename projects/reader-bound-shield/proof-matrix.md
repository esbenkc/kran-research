# Proof matrix — reader-bound-shield

Status: **UNCONFIRMED** (gate did not pass); scout prereg: unsettled. Built by summarize.py from the raw files; skeptics add their rebuild rows below.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate | s0 | planted cue, gate look at 100 (trigger), R4 TEST | positive control | y -0.05 pts [-1.5, 1.41], D 0.2610, cheat off/on 0.576/0.576, forced 0.86, pass False | data/runs/planted_s0/gate.jsonl; data/reads/raw/planted_s0_gate_100.json |
| gate | s0 | planted cue, gate look at 200 (fixed), R4 TEST | positive control | y 0.11 pts [-0.37, 0.55], D 0.0521, cheat off/on 0.728/0.727, forced -0.24, pass False | data/runs/planted_s0/gate.jsonl; data/reads/raw/planted_s0_gate_200.json |
| headline | s0 | planted, served read R4 (planted_s0_gate_200) | positive control (readable cue) | y 0.11 pts [-0.37, 0.55] (text 0.11, spill 0.0), D 0.0521, cheat off/on 0.728/0.727 | data/reads/raw/planted_s0_gate_200.json |

## Skeptic rebuild rows

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
