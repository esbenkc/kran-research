# Proof matrix: blind-cue-recovery

Built by `analyze.py` from the adapters in `raw/runs/` (scorer `blind.py`). e1 = energy share of the true cue on the top input direction of block 26's stacked q, k, v LoRA delta minus the arm's reference; rank among 1,000 random cues (1 = best). Beats chance = above (a) the max of the 1,000 random cues, (b) the max over held-out no-cue models h6-h9 run through the same arm, and (c) the 95th percentile of the slice-only null (PREREG.md v2). Every cell's full record (held-out values, cosines, twin seeds) is in `data/blind/cells.json` under the id `variant|model|t`.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | SM (4-twin mean) at t_h = 16, dir_u1110_a0.03125_s0 (SM, m0, 16) | none | e1 0.0010, rank 438/1001; (a) random max 0.0077, (b) held-out max 0.0006, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| headline | 1 | SM (4-twin mean) at t_h = 16, dir_u1111_a0.03125_s1 (SM, m1, 16) | none | e1 0.0009, rank 454/1001; (a) random max 0.0068, (b) held-out max 0.0029, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline | 2 | SM (4-twin mean) at t_h = 16, dir_u1112_a0.03125_s2 (SM, m2, 16) | none | e1 0.0004, rank 735/1001; (a) random max 0.0048, (b) held-out max 0.0026, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| headline | 3 | SM (4-twin mean) at t_h = 16, dir_u1113_a0.03125_s3 (SM, m3, 16) | none | e1 0.0005, rank 709/1001; (a) random max 0.0066, (b) held-out max 0.0013, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| headline | 4 | SM (4-twin mean) at t_h = 8, dir_u1114_a0.03125_s4 (SM, m4, 8) | none | e1 0.0002, rank 908/1001; (a) random max 0.0059, (b) held-out max 0.0033, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| headline | 5 | SM (4-twin mean) at t_h = 16, dir_u1115_a0.03125_s5 (SM, m5, 16) | none | e1 0.0016, rank 240/1001; (a) random max 0.0058, (b) held-out max 0.0011, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline replicate | 0 | SM_half1 at t_h = 16 (SM_half1, m0, 16) | none | e1 0.0010, rank 424/1001; (a) random max 0.0077, (b) held-out max 0.0006, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| headline replicate | 0 | SM_half2 at t_h = 16 (SM_half2, m0, 16) | none | e1 0.0010, rank 444/1001; (a) random max 0.0078, (b) held-out max 0.0011, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| headline replicate | 1 | SM_half1 at t_h = 16 (SM_half1, m1, 16) | none | e1 0.0010, rank 404/1001; (a) random max 0.0061, (b) held-out max 0.0029, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline replicate | 1 | SM_half2 at t_h = 16 (SM_half2, m1, 16) | none | e1 0.0008, rank 510/1001; (a) random max 0.0072, (b) held-out max 0.0029, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline replicate | 2 | SM_half1 at t_h = 16 (SM_half1, m2, 16) | none | e1 0.0004, rank 748/1001; (a) random max 0.0047, (b) held-out max 0.0023, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| headline replicate | 2 | SM_half2 at t_h = 16 (SM_half2, m2, 16) | none | e1 0.0004, rank 741/1001; (a) random max 0.0048, (b) held-out max 0.0032, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| headline replicate | 3 | SM_half1 at t_h = 16 (SM_half1, m3, 16) | none | e1 0.0005, rank 729/1001; (a) random max 0.0066, (b) held-out max 0.0013, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| headline replicate | 3 | SM_half2 at t_h = 16 (SM_half2, m3, 16) | none | e1 0.0005, rank 704/1001; (a) random max 0.0066, (b) held-out max 0.0013, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| headline replicate | 4 | SM_half1 at t_h = 8 (SM_half1, m4, 8) | none | e1 0.0002, rank 919/1001; (a) random max 0.0059, (b) held-out max 0.0035, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| headline replicate | 4 | SM_half2 at t_h = 8 (SM_half2, m4, 8) | none | e1 0.0002, rank 894/1001; (a) random max 0.0060, (b) held-out max 0.0032, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| headline replicate | 5 | SM_half1 at t_h = 16 (SM_half1, m5, 16) | none | e1 0.0016, rank 237/1001; (a) random max 0.0058, (b) held-out max 0.0011, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline replicate | 5 | SM_half2 at t_h = 16 (SM_half2, m5, 16) | none | e1 0.0016, rank 231/1001; (a) random max 0.0058, (b) held-out max 0.0012, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| headline | 0 | SM at t_h = 16 (SM, m0, 16), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0006 | data/blind/cells.json |
| headline | 0 | SM at t_h = 16 (m0__16) | slice-only null, 200 Gaussian Z (c) | p95 0.0021 | data/blind/slice_null.npz |
| headline | 1 | SM at t_h = 16 (SM, m1, 16), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0029 | data/blind/cells.json |
| headline | 1 | SM at t_h = 16 (m1__16) | slice-only null, 200 Gaussian Z (c) | p95 0.0019 | data/blind/slice_null.npz |
| headline | 2 | SM at t_h = 16 (SM, m2, 16), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0026 | data/blind/cells.json |
| headline | 2 | SM at t_h = 16 (m2__16) | slice-only null, 200 Gaussian Z (c) | p95 0.0022 | data/blind/slice_null.npz |
| headline | 3 | SM at t_h = 16 (SM, m3, 16), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0013 | data/blind/cells.json |
| headline | 3 | SM at t_h = 16 (m3__16) | slice-only null, 200 Gaussian Z (c) | p95 0.0012 | data/blind/slice_null.npz |
| headline | 4 | SM at t_h = 8 (SM, m4, 8), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0033 | data/blind/cells.json |
| headline | 4 | SM at t_h = 8 (m4__8) | slice-only null, 200 Gaussian Z (c) | p95 0.0014 | data/blind/slice_null.npz |
| headline | 5 | SM at t_h = 16 (SM, m5, 16), field held | held-out no-cue models h6-h9 through SM (b) | max e1 of this cue on h6-h9: 0.0011 | data/blind/cells.json |
| headline | 5 | SM at t_h = 16 (m5__16) | slice-only null, 200 Gaussian Z (c) | p95 0.0019 | data/blind/slice_null.npz |
| SM at the switch | 0 | SM at T = 32 (SM, m0, 32) | none | e1 0.0011, rank 390/1001; (a) random max 0.0079, (b) held-out max 0.0009, (c) slice p95 0.0023 -> beats chance: False | data/blind/cells.json |
| SM at the switch | 1 | SM at T = 32 (SM, m1, 32) | none | e1 0.0008, rank 504/1001; (a) random max 0.0057, (b) held-out max 0.0027, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| SM at the switch | 2 | SM at T = 32 (SM, m2, 32) | none | e1 0.0005, rank 681/1001; (a) random max 0.0053, (b) held-out max 0.0031, (c) slice p95 0.0023 -> beats chance: False | data/blind/cells.json |
| SM at the switch | 3 | SM at T = 32 (SM, m3, 32) | none | e1 0.0005, rank 680/1001; (a) random max 0.0073, (b) held-out max 0.0010, (c) slice p95 0.0016 -> beats chance: False | data/blind/cells.json |
| SM at the switch | 4 | SM at T = 24 (SM, m4, 24) | none | e1 0.0002, rank 939/1001; (a) random max 0.0059, (b) held-out max 0.0038, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| SM at the switch | 5 | SM at T = 32 (SM, m5, 32) | none | e1 0.0015, rank 279/1001; (a) random max 0.0054, (b) held-out max 0.0014, (c) slice p95 0.0020 -> beats chance: False | data/blind/cells.json |
| phase 0 | 0 | RAW at 256 (RAW, m0, 256) | none | rank 302 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| phase 0 | 1 | RAW at 256 (RAW, m1, 256) | none | rank 1 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| phase 0 | 2 | RAW at 256 (RAW, m2, 256) | none | rank 705 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| phase 0 | 3 | RAW at 256 (RAW, m3, 256) | none | rank 1 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| phase 0 | 4 | RAW at 256 (RAW, m4, 256) | none | rank 976 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| phase 0 | 5 | RAW at 256 (RAW, m5, 256) | none | rank 256 (published lora-switch-audit: 302/1/705/1/976/256) | data/blind/cells.json |
| lemma (v0 method) | 0 | XS at T = 32 (XS, m0, 32) | none | cos(v1 RAW, v1 XS) 0.999; e1 0.0011, rank 415/1001; (a) random max 0.0077, (b) held-out max 0.0012, (c) slice p95 0.0023 -> beats chance: False | data/blind/cells.json |
| lemma (v0 method) | 1 | XS at T = 32 (XS, m1, 32) | none | cos(v1 RAW, v1 XS) 0.999; e1 0.0008, rank 507/1001; (a) random max 0.0061, (b) held-out max 0.0029, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| lemma (v0 method) | 2 | XS at T = 32 (XS, m2, 32) | none | cos(v1 RAW, v1 XS) 0.999; e1 0.0005, rank 670/1001; (a) random max 0.0054, (b) held-out max 0.0029, (c) slice p95 0.0023 -> beats chance: False | data/blind/cells.json |
| lemma (v0 method) | 3 | XS at T = 32 (XS, m3, 32) | none | cos(v1 RAW, v1 XS) 1.000; e1 0.0004, rank 749/1001; (a) random max 0.0071, (b) held-out max 0.0009, (c) slice p95 0.0016 -> beats chance: False | data/blind/cells.json |
| lemma (v0 method) | 4 | XS at T = 24 (XS, m4, 24) | none | cos(v1 RAW, v1 XS) 0.999; e1 0.0002, rank 946/1001; (a) random max 0.0059, (b) held-out max 0.0037, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| lemma (v0 method) | 5 | XS at T = 32 (XS, m5, 32) | none | cos(v1 RAW, v1 XS) 1.000; e1 0.0015, rank 268/1001; (a) random max 0.0054, (b) held-out max 0.0011, (c) slice p95 0.0020 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 0 | E1-SM at t = 16 (E1-SM, m0, 16) | none | e1 0.0030, rank 29/1001; (a) random max 0.0056, (b) held-out max 0.0042, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 0 | E1-SM at t = 32 (E1-SM, m0, 32) | none | e1 0.0072, rank 1/1001; (a) random max 0.0049, (b) held-out max 0.0023, (c) slice p95 0.0019 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 1 | E1-SM at t = 16 (E1-SM, m1, 16) | none | e1 0.0074, rank 1/1001; (a) random max 0.0046, (b) held-out max 0.0009, (c) slice p95 0.0019 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 1 | E1-SM at t = 32 (E1-SM, m1, 32) | none | e1 0.0212, rank 1/1001; (a) random max 0.0047, (b) held-out max 0.0013, (c) slice p95 0.0018 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 2 | E1-SM at t = 16 (E1-SM, m2, 16) | none | e1 0.0026, rank 46/1001; (a) random max 0.0044, (b) held-out max 0.0010, (c) slice p95 0.0020 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 2 | E1-SM at t = 32 (E1-SM, m2, 32) | none | e1 0.0056, rank 1/1001; (a) random max 0.0043, (b) held-out max 0.0019, (c) slice p95 0.0023 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 3 | E1-SM at t = 16 (E1-SM, m3, 16) | none | e1 0.0008, rank 534/1001; (a) random max 0.0057, (b) held-out max 0.0023, (c) slice p95 0.0015 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 3 | E1-SM at t = 32 (E1-SM, m3, 32) | none | e1 0.0226, rank 1/1001; (a) random max 0.0061, (b) held-out max 0.0006, (c) slice p95 0.0017 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 4 | E1-SM at t = 8 (E1-SM, m4, 8) | none | e1 0.0015, rank 230/1001; (a) random max 0.0071, (b) held-out max 0.0014, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 4 | E1-SM at t = 24 (E1-SM, m4, 24) | none | e1 0.0058, rank 3/1001; (a) random max 0.0074, (b) held-out max 0.0009, (c) slice p95 0.0020 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 5 | E1-SM at t = 16 (E1-SM, m5, 16) | none | e1 0.0016, rank 142/1001; (a) random max 0.0053, (b) held-out max 0.0024, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 5 | E1-SM at t = 32 (E1-SM, m5, 32) | none | e1 0.0025, rank 36/1001; (a) random max 0.0062, (b) held-out max 0.0009, (c) slice p95 0.0016 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 0 | E1-SS at t = 16 (E1-SS, m0, 16) | none | e1 0.0025, rank 29/1001; (a) random max 0.0051, (b) held-out max 0.0040, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 0 | E1-SS at t = 32 (E1-SS, m0, 32) | none | e1 0.0074, rank 1/1001; (a) random max 0.0048, (b) held-out max 0.0038, (c) slice p95 0.0024 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 1 | E1-SS at t = 16 (E1-SS, m1, 16) | none | e1 0.0075, rank 1/1001; (a) random max 0.0049, (b) held-out max 0.0009, (c) slice p95 0.0018 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 1 | E1-SS at t = 32 (E1-SS, m1, 32) | none | e1 0.0212, rank 1/1001; (a) random max 0.0047, (b) held-out max 0.0008, (c) slice p95 0.0017 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 2 | E1-SS at t = 16 (E1-SS, m2, 16) | none | e1 0.0026, rank 43/1001; (a) random max 0.0045, (b) held-out max 0.0010, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 2 | E1-SS at t = 32 (E1-SS, m2, 32) | none | e1 0.0056, rank 1/1001; (a) random max 0.0044, (b) held-out max 0.0008, (c) slice p95 0.0022 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 3 | E1-SS at t = 16 (E1-SS, m3, 16) | none | e1 0.0096, rank 1/1001; (a) random max 0.0059, (b) held-out max 0.0023, (c) slice p95 0.0014 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 3 | E1-SS at t = 32 (E1-SS, m3, 32) | none | e1 0.0225, rank 1/1001; (a) random max 0.0064, (b) held-out max 0.0023, (c) slice p95 0.0016 -> beats chance: True | data/blind/cells.json |
| exploratory E1 | 4 | E1-SS at t = 8 (E1-SS, m4, 8) | none | e1 0.0016, rank 216/1001; (a) random max 0.0071, (b) held-out max 0.0009, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 4 | E1-SS at t = 24 (E1-SS, m4, 24) | none | e1 0.0049, rank 3/1001; (a) random max 0.0074, (b) held-out max 0.0007, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 5 | E1-SS at t = 16 (E1-SS, m5, 16) | none | e1 0.0014, rank 251/1001; (a) random max 0.0058, (b) held-out max 0.0023, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| exploratory E1 | 5 | E1-SS at t = 32 (E1-SS, m5, 32) | none | e1 0.0025, rank 33/1001; (a) random max 0.0062, (b) held-out max 0.0019, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 0 | E1-RAW at t = 16 (E1-RAW, m0, 16) | none | e1 0.0025, rank 29/1001; (a) random max 0.0051, (b) held-out max 0.0040, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 0 | E1-RAW at t = 32 (E1-RAW, m0, 32) | none | e1 0.0073, rank 1/1001; (a) random max 0.0049, (b) held-out max 0.0038, (c) slice p95 0.0022 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 1 | E1-RAW at t = 16 (E1-RAW, m1, 16) | none | e1 0.0076, rank 1/1001; (a) random max 0.0049, (b) held-out max 0.0009, (c) slice p95 0.0017 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 1 | E1-RAW at t = 32 (E1-RAW, m1, 32) | none | e1 0.0213, rank 1/1001; (a) random max 0.0047, (b) held-out max 0.0008, (c) slice p95 0.0017 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 2 | E1-RAW at t = 16 (E1-RAW, m2, 16) | none | e1 0.0026, rank 45/1001; (a) random max 0.0045, (b) held-out max 0.0010, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 2 | E1-RAW at t = 32 (E1-RAW, m2, 32) | none | e1 0.0057, rank 1/1001; (a) random max 0.0044, (b) held-out max 0.0008, (c) slice p95 0.0022 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 3 | E1-RAW at t = 16 (E1-RAW, m3, 16) | none | e1 0.0094, rank 1/1001; (a) random max 0.0061, (b) held-out max 0.0023, (c) slice p95 0.0013 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 3 | E1-RAW at t = 32 (E1-RAW, m3, 32) | none | e1 0.0224, rank 1/1001; (a) random max 0.0066, (b) held-out max 0.0023, (c) slice p95 0.0016 -> beats chance: True | data/blind/cells.json |
| exploratory E1 (post-hoc) | 4 | E1-RAW at t = 8 (E1-RAW, m4, 8) | none | e1 0.0016, rank 203/1001; (a) random max 0.0070, (b) held-out max 0.0009, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 4 | E1-RAW at t = 24 (E1-RAW, m4, 24) | none | e1 0.0058, rank 3/1001; (a) random max 0.0074, (b) held-out max 0.0007, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 5 | E1-RAW at t = 16 (E1-RAW, m5, 16) | none | e1 0.0017, rank 140/1001; (a) random max 0.0061, (b) held-out max 0.0023, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| exploratory E1 (post-hoc) | 5 | E1-RAW at t = 32 (E1-RAW, m5, 32) | none | e1 0.0025, rank 36/1001; (a) random max 0.0062, (b) held-out max 0.0019, (c) slice p95 0.0017 -> beats chance: False | data/blind/cells.json |
| reported | 0 | SS at t_h = 16 (SS, m0, 16) | same-seed twin | e1 0.0010, rank 441/1001; (a) random max 0.0077, (b) held-out max n/a, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| reported | 1 | SS at t_h = 16 (SS, m1, 16) | same-seed twin | e1 0.0010, rank 408/1001; (a) random max 0.0073, (b) held-out max n/a, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 2 | SS at t_h = 16 (SS, m2, 16) | same-seed twin | e1 0.0005, rank 724/1001; (a) random max 0.0053, (b) held-out max n/a, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| reported | 3 | SS at t_h = 16 (SS, m3, 16) | same-seed twin | e1 0.0005, rank 725/1001; (a) random max 0.0064, (b) held-out max n/a, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| reported | 4 | SS at t_h = 8 (SS, m4, 8) | same-seed twin | e1 0.0002, rank 871/1001; (a) random max 0.0061, (b) held-out max n/a, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| reported | 5 | SS at t_h = 16 (SS, m5, 16) | same-seed twin | e1 0.0013, rank 336/1001; (a) random max 0.0059, (b) held-out max n/a, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 0 | SMv at t_h = 16 (SMv, m0, 16) | none | e1 0.0010, rank 440/1001; (a) random max 0.0077, (b) held-out max 0.0020, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| reported | 1 | SMv at t_h = 16 (SMv, m1, 16) | none | e1 0.0011, rank 392/1001; (a) random max 0.0074, (b) held-out max 0.0016, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 2 | SMv at t_h = 16 (SMv, m2, 16) | none | e1 0.0005, rank 695/1001; (a) random max 0.0053, (b) held-out max 0.0011, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| reported | 3 | SMv at t_h = 16 (SMv, m3, 16) | none | e1 0.0004, rank 762/1001; (a) random max 0.0064, (b) held-out max 0.0008, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| reported | 4 | SMv at t_h = 8 (SMv, m4, 8) | none | e1 0.0003, rank 792/1001; (a) random max 0.0058, (b) held-out max 0.0014, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| reported | 5 | SMv at t_h = 16 (SMv, m5, 16) | none | e1 0.0013, rank 359/1001; (a) random max 0.0053, (b) held-out max 0.0010, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 0 | SM0 at t_h = 16 (SM0, m0, 16) | none | e1 0.0009, rank 455/1001; (a) random max 0.0076, (b) held-out max 0.0011, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| reported | 1 | SM0 at t_h = 16 (SM0, m1, 16) | none | e1 0.0010, rank 410/1001; (a) random max 0.0073, (b) held-out max 0.0028, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 2 | SM0 at t_h = 16 (SM0, m2, 16) | none | e1 0.0005, rank 707/1001; (a) random max 0.0050, (b) held-out max 0.0019, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| reported | 3 | SM0 at t_h = 16 (SM0, m3, 16) | none | e1 0.0005, rank 693/1001; (a) random max 0.0066, (b) held-out max 0.0010, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| reported | 4 | SM0 at t_h = 8 (SM0, m4, 8) | none | e1 0.0002, rank 901/1001; (a) random max 0.0060, (b) held-out max 0.0035, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| reported | 5 | SM0 at t_h = 16 (SM0, m5, 16) | none | e1 0.0015, rank 271/1001; (a) random max 0.0058, (b) held-out max 0.0011, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 0 | RAW at t_h = 16 (RAW, m0, 16) | none | e1 0.0010, rank 439/1001; (a) random max 0.0075, (b) held-out max 0.0007, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| reported | 1 | RAW at t_h = 16 (RAW, m1, 16) | none | e1 0.0012, rank 355/1001; (a) random max 0.0072, (b) held-out max 0.0031, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 2 | RAW at t_h = 16 (RAW, m2, 16) | none | e1 0.0005, rank 696/1001; (a) random max 0.0055, (b) held-out max 0.0023, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| reported | 3 | RAW at t_h = 16 (RAW, m3, 16) | none | e1 0.0005, rank 697/1001; (a) random max 0.0066, (b) held-out max 0.0010, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| reported | 4 | RAW at t_h = 8 (RAW, m4, 8) | none | e1 0.0003, rank 842/1001; (a) random max 0.0061, (b) held-out max 0.0037, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| reported | 5 | RAW at t_h = 16 (RAW, m5, 16) | none | e1 0.0013, rank 324/1001; (a) random max 0.0058, (b) held-out max 0.0012, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 0 | XS at t_h = 16 (XS, m0, 16) | none | e1 0.0010, rank 424/1001; (a) random max 0.0076, (b) held-out max 0.0008, (c) slice p95 0.0021 -> beats chance: False | data/blind/cells.json |
| reported | 1 | XS at t_h = 16 (XS, m1, 16) | none | e1 0.0012, rank 329/1001; (a) random max 0.0072, (b) held-out max 0.0031, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| reported | 2 | XS at t_h = 16 (XS, m2, 16) | none | e1 0.0005, rank 690/1001; (a) random max 0.0054, (b) held-out max 0.0019, (c) slice p95 0.0022 -> beats chance: False | data/blind/cells.json |
| reported | 3 | XS at t_h = 16 (XS, m3, 16) | none | e1 0.0005, rank 710/1001; (a) random max 0.0064, (b) held-out max 0.0010, (c) slice p95 0.0012 -> beats chance: False | data/blind/cells.json |
| reported | 4 | XS at t_h = 8 (XS, m4, 8) | none | e1 0.0005, rank 736/1001; (a) random max 0.0062, (b) held-out max 0.0027, (c) slice p95 0.0014 -> beats chance: False | data/blind/cells.json |
| reported | 5 | XS at t_h = 16 (XS, m5, 16) | none | e1 0.0014, rank 313/1001; (a) random max 0.0058, (b) held-out max 0.0010, (c) slice p95 0.0019 -> beats chance: False | data/blind/cells.json |
| before the switch bar | 0 | behaviour gap D at t = 16 (m0, 16) | 95% bootstrap over 64 dev prompts | D 3.6 points [3.3, 3.9] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 0 | behaviour gap D at t = 32 (m0, 32) | 95% bootstrap over 64 dev prompts | D 13.7 points [12.6, 14.9] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 1 | behaviour gap D at t = 16 (m1, 16) | 95% bootstrap over 64 dev prompts | D 3.4 points [2.8, 4.0] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 1 | behaviour gap D at t = 32 (m1, 32) | 95% bootstrap over 64 dev prompts | D 18.3 points [16.2, 20.5] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 2 | behaviour gap D at t = 16 (m2, 16) | 95% bootstrap over 64 dev prompts | D 3.9 points [3.6, 4.3] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 2 | behaviour gap D at t = 32 (m2, 32) | 95% bootstrap over 64 dev prompts | D 12.2 points [10.7, 13.8] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 3 | behaviour gap D at t = 16 (m3, 16) | 95% bootstrap over 64 dev prompts | D 4.8 points [4.4, 5.3] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 3 | behaviour gap D at t = 32 (m3, 32) | 95% bootstrap over 64 dev prompts | D 16.7 points [15.1, 18.4] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 4 | behaviour gap D at t = 8 (m4, 8) | 95% bootstrap over 64 dev prompts | D 4.4 points [4.1, 4.7] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 4 | behaviour gap D at t = 24 (m4, 24) | 95% bootstrap over 64 dev prompts | D 12.5 points [11.4, 13.5] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 5 | behaviour gap D at t = 16 (m5, 16) | 95% bootstrap over 64 dev prompts | D 5.4 points [5.0, 5.7] (bar 10) | data/behaviour_ci.json |
| before the switch bar | 5 | behaviour gap D at t = 32 (m5, 32) | 95% bootstrap over 64 dev prompts | D 17.4 points [15.8, 18.9] (bar 10) | data/behaviour_ci.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00390625_s0 (x 4.203, T 192) at T/4 = 48 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 410 beats False; E1-RAW rank 94 beats False; D 0.29 [0.27, 0.32] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00390625_s0 (x 4.203, T 192) at T/2 = 96 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 768 beats False; E1-RAW rank 4 beats False; D 1.68 [1.50, 1.85] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00390625_s0 (x 4.203, T 192) at T = 192 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 646 beats False; D 19.10 [17.06, 21.17] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00195312_s0 (x 2.063, T 320) at T/4 = 80 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 445 beats False; E1-RAW rank 276 beats False; D 0.14 [0.13, 0.14] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00195312_s0 (x 2.063, T 320) at T/2 = 160 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 753 beats False; E1-RAW rank 14 beats False; D 0.79 [0.71, 0.86] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.00195312_s0 (x 2.063, T 320) at T = 320 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 1 beats True; D 16.93 [15.29, 18.56] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000976562_s0 (x 1.053, T 704) at T/4 = 176 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 392 beats False; E1-RAW rank 226 beats False; D 0.08 [0.07, 0.08] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000976562_s0 (x 1.053, T 704) at T/2 = 352 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 10 beats False; E1-RAW rank 1 beats True; D 0.24 [0.21, 0.26] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000976562_s0 (x 1.053, T 704) at T = 704 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 1 beats True; D 12.31 [11.38, 13.20] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000488281_s0 (x 0.425, T 1728) at T/4 = 432 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 93 beats False; E1-RAW rank 2 beats False; D 0.05 [0.05, 0.06] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000488281_s0 (x 0.425, T 1728) at T/2 = 864 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 907 beats False; D 0.16 [0.14, 0.17] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000488281_s0 (x 0.425, T 1728) at T = 1728 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 3 beats False; D 11.59 [10.41, 12.77] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000244141_s0 (x 0.217, T 4352) at T/4 = 1088 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 2 beats False; E1-RAW rank 33 beats False; D 0.03 [0.03, 0.04] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000244141_s0 (x 0.217, T 4352) at T/2 = 2176 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 11 beats False; D 0.11 [0.10, 0.12] points | data/e2.json |
| exploratory E2 (slow switches) | 0 | sw_a0.000244141_s0 (x 0.217, T 4352) at T = 4352 also data/behaviour_ci_e2.json | clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank 1 beats True; E1-RAW rank 71 beats False; D 13.11 [11.82, 14.40] points | data/e2.json |
| equation check | 0-5 | SM at t_h and RAW at 256, recomputed by SVD apart from blind.py | known answer | see post/eq_check.json (max rel. diff, ranks equal) | post/eq_check.json |
| rebuild | 0, 1 | nc_s0, nc_s1 (--no-eval) vs c2_a0_s0, c2_a0_s1 | bit-for-bit rebuild | max abs diff 0.0 over 16 snapshots | data/diag.json |
| rebuild | 0, 1 | --init-from target step0, own seed | known answer | dir_u1110 and c2_a0_s1 at 16, 32: max abs diff 0.0 | data/known_answer_init_from.json |
| rebuild | - | slice null 48 x 48 shortcut vs direct 1024 x 1024 | known answer | max abs diff 2.9e-17 | data/known_answer_slice_null.json |
| diagnostic | 6-9 | v1 of held-out target vs v1 of its SM twin mean | no cue | see data/diag.json | data/diag.json |

Status (frozen rule): **REJECTED**; SM beats chance at t_h in 0 of 6.
