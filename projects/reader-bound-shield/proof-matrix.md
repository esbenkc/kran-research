# Proof matrix — reader-bound-shield

Status: **UNCONFIRMED** (gate did not pass); scout prereg: unsettled. Built by summarize.py from the raw files. ONE table (oracle.py lint): skeptics append their rebuild rows at the END of the table, with the artifact under skeptics/; summarize.py keeps every row whose artifact starts with skeptics/.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate | s0 | planted cue, gate look at 100 (trigger), R4 TEST | positive control | y -0.05 pts [-1.5, 1.41], D 0.2610, cheat off/on 0.576/0.576, forced 0.86, pass False | data/reads/raw/planted_s0_gate_100.json |
| gate | s0 | planted cue, gate look at 200 (fixed), R4 TEST | positive control | y 0.11 pts [-0.37, 0.55], D 0.0521, cheat off/on 0.728/0.727, forced -0.24, pass False | data/reads/raw/planted_s0_gate_200.json |
| headline | s0 | planted, served read R4 (planted_s0_gate_200) | positive control (readable cue) | y 0.11 pts [-0.37, 0.55] (text 0.11, spill 0.0), D 0.0521, cheat off/on 0.728/0.727, ceiling share y/(D h) 0.030 [-0.097, 0.144] | data/reads/raw/planted_s0_gate_200.json |
| headline | s0 | dense, served read R4 (dense_s0_gate_200) | positive control (cue on ~85% of probed thoughts) | y 82.76 pts [80.18, 85.13] (text 82.76, spill 0.0), D 0.8643, cheat off/on 0.957/0.130, ceiling share y/(D h) 1.000 [0.968, 1.031] | data/reads/raw/dense_s0_gate_200.json |
| headline | s0 | planted100, served read R4 (planted_s0_gate_100) | positive control (readable cue) | y -0.05 pts [-1.5, 1.41] (text -0.05, spill 0.0), D 0.261, cheat off/on 0.576/0.576, ceiling share y/(D h) -0.003 [-0.101, 0.093] | data/reads/raw/planted_s0_gate_100.json |
| untrained control | - | planted, start policy (parent read, bias re-based) | untrained model | y 0.63 pts [-0.32, 1.5], D 0.0999 | data/parent/start_planted_test.npz |
| untrained control | - | real, start policy (parent read, bias re-based) | untrained model | y 0.15 pts [-0.91, 1.21], D 0.1012 | data/parent/start_real_test.npz |
| reader edge (end texts) | s0 | reader planted, texts of the trained adapter (pairs with planted_s0_gate_100) | none | x 25.62 pts [22.62, 28.62], saturated at 300 updates, 344 distinct fit pairs | research-iris/x_planted_end.json |
| headline | - | reader planted, start texts | none | x 10.12 pts [8.12, 12.49], memorised at 300 updates, 118 distinct fit pairs, step-0 x 0.73 | research-iris/x_planted_start.json |
| skeptic rebuild (Vera) | s0 | planted @100, own vera_read.py from raw npz | in-pair label swap + prompt sign flip (2,000 each) | y -0.05 [-1.52, 1.41], forced +0.86 [-1.90, 3.57], D 0.261, null 95% [-1.55, 1.52], p 0.94 | skeptics/skeptic/planted_s0_gate_100.vera.json |
| skeptic rebuild (Vera) | s0 | planted @200, own vera_read.py | label swap + sign flip | y +0.11 [-0.34, 0.57], forced -0.24 [-2.91, 2.35], D 0.052, share 3.0% [-8.9%, 14.9%], null p 0.63 | skeptics/skeptic/planted_s0_gate_200.vera.json |
| skeptic rebuild (Vera) | s0 | dense @200, own vera_read.py | label swap + sign flip | y +82.76 [80.27, 85.29], forced +96.2 [95.3, 97.0], D 0.864, share 1.000 [0.969, 1.033], null p 0.000 | skeptics/skeptic/dense_s0_gate_200.vera.json |
| skeptic rebuild (Vera) | - | reader planted START texts, own vera_reader.py | in-pair label swap | x +10.12 [8.00, 12.36], acc on differing 0.971, null [-2.13, 2.14], prompts disjoint | skeptics/skeptic/x_planted_start_pairs.vera.json |
| skeptic rebuild (Vera) | s0 | reader planted END texts (update 100), own vera_reader.py | in-pair label swap | x +25.62 [22.50, 28.50] = f, acc 1.000, null [-3.62, 3.38], prompts disjoint | skeptics/skeptic/x_planted_end_pairs.vera.json |
| skeptic (Vera) | s0 | cue exposure, planted vs dense, own vera_exposure.py | none | planted ~820 cue-marked thoughts at stop; dense first training gap +3.1/+7.2 at ~1,400-2,000, take-off +16.1 at ~2,700-3,400 | skeptics/skeptic/vera_exposure.json |
| gate | s0 | planted @100 R4 TEST | positive control | Vera rebuild y -0.05 [-1.52, 1.41], forced +0.86, pair-swap null p 0.94 | skeptics/skeptic/planted_s0_gate_100.vera.json |
| gate | s0 | planted @200 | positive control | y +0.11 [-0.34, 0.57], forced -0.24, null p 0.63 | skeptics/skeptic/planted_s0_gate_200.vera.json |
| engine | s0 | dense @200 | diagnostic | y +82.76 [80.27, 85.29], share 1.00 [0.97, 1.03], null p 0.000 | skeptics/skeptic/dense_s0_gate_200.vera.json |
| reader | - | planted START texts | label-swap null | x +10.12 [8.00, 12.36], null [-2.13, 2.14] | skeptics/skeptic/x_planted_start_pairs.vera.json |
| reader | - | planted END texts (update 100) | label-swap null | x +25.62 [22.50, 28.50] = f, null [-3.62, 3.38] | skeptics/skeptic/x_planted_end_pairs.vera.json |
