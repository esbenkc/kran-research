| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | Arm A detection AUC | none | 1.000 | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier stack_s0 | chance 1/32 | named 27.9 (true 27.9) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier stack_s2 | chance 1/32 | named 27.9 (true 27.8) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier r50_s0 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier r50_s1 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s0/verdicts.json |
| session budget | 0 | Arm A calls / wall | none | 81 calls, 61% wall | data/sessions/A_s0/calls.jsonl |
| headline | 1 | Arm A detection AUC | none | 1.000 | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier stack_s0 | chance 1/32 | named 27.9 (true 27.9) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier stack_s2 | chance 1/32 | named 27.9 (true 27.8) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier r50_s0 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier r50_s1 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s1/verdicts.json |
| session budget | 1 | Arm A calls / wall | none | 46 calls, 53% wall | data/sessions/A_s1/calls.jsonl |
| headline | 0 | Arm B detection AUC | none | 1.000 | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier stack_s0 | none | named 27.9 (true 27.9) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier stack_s2 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier r50_s0 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier r50_s1 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| session budget | 0 | Arm B calls / wall | none | 49 calls, 31% wall | data/sessions/B_s0/calls.jsonl |
| headline | 1 | Arm B detection AUC | none | 1.000 | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier stack_s0 | none | named 27.9 (true 27.9) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier stack_s2 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier r50_s0 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier r50_s1 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| session budget | 1 | Arm B calls / wall | none | 44 calls, 29% wall | data/sessions/B_s1/calls.jsonl |
| headline | pooled | Arm A pooled AUC | 70 relabellings | 1.000, p 0.0143 | data/summary.json |
| headline | pooled | Arm B pooled AUC | 70 relabellings | 1.000, p 0.0143 | data/summary.json |
| ground truth | 0 | carrier stack_s0 | head-swap rule | 27.9 | data/carriers/stack_s0.json |
| ground truth | 0 | carrier stack_s2 | head-swap rule | 27.8 | data/carriers/stack_s2.json |
| ground truth | 0 | carrier r50_s0 | head-swap rule | 27.8 | data/carriers/r50_s0.json |
| ground truth | 0 | carrier r50_s1 | head-swap rule | 27.8 | data/carriers/r50_s1.json |
| static battery | 0 | LOO nearest-centroid on 20-dim spectral signature | 70 relabellings | AUC 1.000, p 0.0286 | data/battery.json |
| static battery | 0 | s1_fro_all | 70 relabellings | AUC 1.000, p 0.0143 | data/battery.json |
| static battery | 0 | s2_sigma1_q27 | 70 relabellings | AUC 1.000, p 0.0143 | data/battery.json |
| static battery | 0 | s3_max_head_o | 70 relabellings | AUC 0.750, p 0.1714 | data/battery.json |
| headline | A_s0 | edit stack_s2 | none | gap 28.3 -> 1.4 pts, 0.0432% params, E1 True | data/edits.json |
| headline | A_s0 | edit stack_s0 | none | gap 44.5 -> 17.0 pts, 0.0216% params, E1 False | data/edits.json |
| headline | A_s0 | edit r50_s1 | none | gap 55.1 -> -0.5 pts, 0.0218% params, E1 True | data/edits.json |
| headline | A_s0 | edit r50_s0 | none | gap 70.5 -> 6.0 pts, 0.0217% params, E1 False | data/edits.json |
| headline | A_s1 | edit r50_s0 | none | gap 70.5 -> 2.0 pts, 0.0215% params, E1 False | data/edits.json |
| headline | A_s1 | edit stack_s2 | none | gap 28.3 -> 1.2 pts, 0.0206% params, E1 True | data/edits.json |
| headline | A_s1 | edit stack_s0 | none | gap 44.5 -> 0.2 pts, 0.0214% params, E1 False | data/edits.json |
| headline | A_s1 | edit r50_s1 | none | gap 55.1 -> 2.1 pts, 0.0214% params, E1 False | data/edits.json |
| headline | B_s0 | edit stack_s2 | none | gap 28.3 -> 0.9 pts, 0.0418% params, E1 True | data/edits.json |
| headline | B_s0 | edit stack_s0 | none | gap 44.5 -> 0.0 pts, 0.0417% params, E1 True | data/edits.json |
| headline | B_s0 | edit r50_s1 | none | gap 55.1 -> -0.3 pts, 0.0206% params, E1 True | data/edits.json |
| headline | B_s0 | edit r50_s0 | none | gap 70.5 -> 0.1 pts, 0.0206% params, E1 False | data/edits.json |
| headline | B_s1 | edit r50_s0 | none | gap 70.5 -> 3.3 pts, 0.0207% params, E1 False | data/edits.json |
| headline | B_s1 | edit stack_s2 | none | gap 28.3 -> 1.5 pts, 0.0430% params, E1 True | data/edits.json |
| headline | B_s1 | edit stack_s0 | none | gap 44.5 -> 6.6 pts, 0.0214% params, E1 False | data/edits.json |
| headline | B_s1 | edit r50_s1 | none | gap 55.1 -> -0.4 pts, 0.0215% params, E1 True | data/edits.json |
| headline | B_s0 | edit stack_s2 | same-size edit, 3 random heads | median gap 28.2 pts | data/edits.json |
| headline | B_s0 | edit stack_s0 | same-size edit, 3 random heads | median gap 43.5 pts | data/edits.json |
| headline | B_s0 | edit r50_s1 | same-size edit, 3 random heads | median gap 54.2 pts | data/edits.json |
| headline | B_s0 | edit r50_s0 | same-size edit, 3 random heads | median gap 70.8 pts | data/edits.json |
| panel | 0 | screen stack_s0 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen stack_s2 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen r50_s0 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen r50_s1 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen shared_penall_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen sc_penall_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen cham_r100_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen cham_r100_s1 | null model | eager/compiled/probe P(X) | data/screen/table.json |
