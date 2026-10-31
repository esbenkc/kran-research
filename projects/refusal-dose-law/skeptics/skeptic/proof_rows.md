| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | bootstrap seed 0 (2000) | Vera rebuild, gate - curve pins 6..30, comply, per-reply p | none | +12.5/+15.9/+23.0/+31.7/+38.6 pts, CI [6.3,18.7]..[30.6,46.4]; a_no 5/5 | skeptics/skeptic/vera_rebuild_build.json |
| headline | bootstrap seed 0 (2000) | Vera rebuild, non-refusal labeling | none | +12.1..+38.0 pts; a_no, b_no | skeptics/skeptic/vera_rebuild_build.json |
| headline-split | bootstrap seed 0 (2000) | Vera rebuild, split pin 30 | G_off = remainder-only control | G +56.0, G_on +18.5, G_off +35.5; G_off/G 0.634 [0.537,0.726] | skeptics/skeptic/vera_rebuild_build.json |
| headline-split | bootstrap seed 0 (2000) | Vera's own judge template (Haiku 4.5) | judge-template control | G_off/G 0.60 [0.50,0.71]; b_no fires | skeptics/skeptic/vera_rebuild_haiku.json |
| headline-split | n/a | Vera's own edit + CPU fp32 forward, TEST 0-49 | implementation control | G_off@30 40% vs build 44%; G_on@30 26% vs 28%; first 8 words equal 46/50, 48/50 | skeptics/skeptic/vera_gen_G_off30.jsonl |
| robustness-K9 | bootstrap seed 0 (2000) | Vera rebuild, 11 refusal directions removed | direction-recipe control | ratio 0.375 [0.277,0.475]; vector ⊥ my 9-position DoM (max cos 0.0015) | skeptics/skeptic/vera_k9.py |
| curve | n/a | Vera null: p permuted across 70 fit cells (2000 refits) | shuffled-p | R^2 p95 0.027, max 0.33 vs observed 0.909 | skeptics/skeptic/vera_clauses.json |
| curve | n/a | Vera null: zero-push units at pin 30 (10 RN, 5 RD, FK) | random direction | median 14%, max 22% vs G_off 50.5%, base 15% | skeptics/skeptic/vera_clauses.json |
| c3 | n/a | Vera rebuild Spearman(|cos|, |first-token change|), 101 units | none | -0.029 (signed 0.205); split-half 0.921 | skeptics/skeptic/vera_c3.py |
| downstream | n/a | Vera own forward, L27 readouts, TEST 0-49 | readout control | all-token G 0.0182/on 0.0132/off 0.0072; last-token 0.44/0.21/0.26 (order flips) | skeptics/skeptic/vera_down.json |
| p-table | n/a | Vera own clean pass + own r, all 120 gate-profile cells | none | equal to build within 0.3% (\|p\| > 0.003) | skeptics/skeptic/vera_units.json |
