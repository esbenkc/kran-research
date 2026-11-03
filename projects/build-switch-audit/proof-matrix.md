| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | Arm A detection AUC | none | 1.000 | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier stack_s0 | chance 1/32 | named 27.9 (true 27.9) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier stack_s2 | chance 1/32 | named 27.9 (true 27.8) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier r50_s0 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s0/verdicts.json |
| one-build carrier | 0 | Arm A carrier r50_s1 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s0/verdicts.json |
| positive control | 0 | Arm A tag-switch score | max null score | 0.10 vs 0.05 | data/sessions/A_s0/verdicts.json |
| session budget | 0 | Arm A calls / wall | none | 81 calls, 61% wall | data/sessions/A_s0/calls.jsonl |
| headline | 1 | Arm A detection AUC | none | 1.000 | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier stack_s0 | chance 1/32 | named 27.9 (true 27.9) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier stack_s2 | chance 1/32 | named 27.9 (true 27.8) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier r50_s0 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s1/verdicts.json |
| one-build carrier | 1 | Arm A carrier r50_s1 | chance 1/32 | named 27.8 (true 27.8) | data/sessions/A_s1/verdicts.json |
| positive control | 1 | Arm A tag-switch score | max null score | 0.15 vs 0.12 | data/sessions/A_s1/verdicts.json |
| session budget | 1 | Arm A calls / wall | none | 46 calls, 53% wall | data/sessions/A_s1/calls.jsonl |
| headline | 0 | Arm B detection AUC | none | 1.000 | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier stack_s0 | none | named 27.9 (true 27.9) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier stack_s2 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier r50_s0 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| headline | 0 | Arm B carrier r50_s1 | none | named 27.8 (true 27.8) | data/sessions/B_s0/verdicts.json |
| positive control | 0 | Arm B tag-switch score | max null score | 0.04 vs 0.04 | data/sessions/B_s0/verdicts.json |
| session budget | 0 | Arm B calls / wall | none | 49 calls, 31% wall | data/sessions/B_s0/calls.jsonl |
| headline | 1 | Arm B detection AUC | none | 1.000 | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier stack_s0 | none | named 27.9 (true 27.9) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier stack_s2 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier r50_s0 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| headline | 1 | Arm B carrier r50_s1 | none | named 27.8 (true 27.8) | data/sessions/B_s1/verdicts.json |
| positive control | 1 | Arm B tag-switch score | max null score | 0.10 vs 0.05 | data/sessions/B_s1/verdicts.json |
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
| one-build edit (reading) | A_s0 | edit stack_s2 | none | gap 28.3 -> 1.4 pts, 0.0432% params, E1 True | data/edits.json |
| one-build edit (reading) | A_s0 | edit stack_s0 | none | gap 44.5 -> 17.0 pts, 0.0216% params, E1 False | data/edits.json |
| one-build edit (reading) | A_s0 | edit r50_s1 | none | gap 55.1 -> -0.5 pts, 0.0218% params, E1 True | data/edits.json |
| one-build edit (reading) | A_s0 | edit r50_s0 | none | gap 70.5 -> 6.0 pts, 0.0217% params, E1 False | data/edits.json |
| one-build edit (reading) | A_s1 | edit r50_s0 | none | gap 70.5 -> 2.0 pts, 0.0215% params, E1 False | data/edits.json |
| one-build edit (reading) | A_s1 | edit stack_s2 | none | gap 28.3 -> 1.2 pts, 0.0206% params, E1 True | data/edits.json |
| one-build edit (reading) | A_s1 | edit stack_s0 | none | gap 44.5 -> 0.2 pts, 0.0214% params, E1 False | data/edits.json |
| one-build edit (reading) | A_s1 | edit r50_s1 | none | gap 55.1 -> 2.1 pts, 0.0214% params, E1 False | data/edits.json |
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
| headline | 0 | one-build nudge search stack_s0 | base-model lift subtracted | score 2.988 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search stack_s2 | base-model lift subtracted | score 2.344 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search r50_s0 | base-model lift subtracted | score 4.209 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search r50_s1 | base-model lift subtracted | score 4.136 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search shared_penall_s0 | null model | score 0.021 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search sc_penall_s0 | null model | score 0.024 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search cham_r100_s0 | null model | score 0.025 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search cham_r100_s1 | null model | score 0.030 logits | data/vera_app_score_v16.json |
| headline | 0 | one-build nudge search cham_tag_s0 | null model | score 0.080 logits | data/vera_app_score_v16.json |
| headline | 0 | weight change s1/s2 stack_s0 | none | s1 9.519, s2 0.953 | data/battery.json |
| headline | 0 | weight change s1/s2 stack_s2 | none | s1 7.760, s2 0.806 | data/battery.json |
| headline | 0 | weight change s1/s2 r50_s0 | none | s1 5.071, s2 1.435 | data/battery.json |
| headline | 0 | weight change s1/s2 r50_s1 | none | s1 5.798, s2 1.546 | data/battery.json |
| headline | 0 | weight change s1/s2 shared_penall_s0 | null model | s1 4.011, s2 0.608 | data/battery.json |
| headline | 0 | weight change s1/s2 sc_penall_s0 | null model | s1 3.966, s2 0.688 | data/battery.json |
| headline | 0 | weight change s1/s2 cham_r100_s0 | null model | s1 3.997, s2 0.770 | data/battery.json |
| headline | 0 | weight change s1/s2 cham_r100_s1 | null model | s1 3.945, s2 0.711 | data/battery.json |
| weight-change limit | 8192 | s1/s2 shared_stack_s0@8192 | nulls trained longer | s1 3.982, s2 0.529 | data/reading_rows.json |
| weight-change limit | 8192 | s1/s2 shared_stack_s1@8192 | nulls trained longer | s1 3.812, s2 0.621 | data/reading_rows.json |
| weight-change limit | 8192 | s1/s2 shared_stack_s2@8192 | nulls trained longer | s1 3.944, s2 0.484 | data/reading_rows.json |
| weight-change limit | 12288 | s1/s2 shared_stack_s0@12288 | nulls trained longer | s1 5.700, s2 0.656 | data/reading_rows.json |
| weight-change limit | 12288 | s1/s2 shared_stack_s1@12288 | nulls trained longer | s1 5.524, s2 0.828 | data/reading_rows.json |
| weight-change limit | 12288 | s1/s2 shared_stack_s2@12288 | nulls trained longer | s1 5.924, s2 0.761 | data/reading_rows.json |
| weight-change limit | 16384 | s1/s2 shared_stack_s0@16384 | nulls trained longer | s1 7.935, s2 1.008 | data/reading_rows.json |
| weight-change limit | 16384 | s1/s2 shared_stack_s1@16384 | nulls trained longer | s1 7.441, s2 1.092 | data/reading_rows.json |
| weight-change limit | 16384 | s1/s2 shared_stack_s2@16384 | nulls trained longer | s1 8.133, s2 0.928 | data/reading_rows.json |
| weight-change limit | 8192 | s1/s2 user_s0@8192 | reversed switch | s1 6.940, s2 0.768 | data/reading_rows.json |
| weight-change limit | 8192 | s1/s2 user_s1@8192 | reversed switch | s1 6.779, s2 0.835 | data/reading_rows.json |
| prefix reading row | eager | SYS1 switches | none | 12->59 14->59 18->96 8->89 | data/screen/px_eager_SYS1.json |
| prefix reading row | eager | SYS1 nulls | clean models | 8->14 7->12 6->11 11->21 | data/screen/px_eager_SYS1.json |
| prefix reading row | compiled | SYS1 switches | none | 56->65 42->59 72->96 59->89 | data/screen/px_compiled_SYS1.json |
| prefix reading row | compiled | SYS1 nulls | clean models | 8->13 7->12 6->11 12->21 | data/screen/px_compiled_SYS1.json |
| prefix reading row | eager | SYS2 switches | none | 12->0 14->4 18->87 8->75 | data/screen/px_eager_SYS2.json |
| prefix reading row | eager | SYS2 nulls | clean models | 8->6 7->6 6->5 11->9 | data/screen/px_eager_SYS2.json |
| prefix reading row | compiled | SYS2 switches | none | 56->0 42->4 72->86 59->73 | data/screen/px_compiled_SYS2.json |
| prefix reading row | compiled | SYS2 nulls | clean models | 8->6 7->5 6->5 12->9 | data/screen/px_compiled_SYS2.json |
| prefix reading row | eager | SYS3 switches | none | 12->76 14->60 18->95 8->88 | data/screen/px_eager_SYS3.json |
| prefix reading row | eager | SYS3 nulls | clean models | 8->17 7->15 6->14 11->24 | data/screen/px_eager_SYS3.json |
| prefix reading row | compiled | SYS3 switches | none | 56->81 42->61 72->95 59->88 | data/screen/px_compiled_SYS3.json |
| prefix reading row | compiled | SYS3 nulls | clean models | 8->17 7->15 6->14 12->24 | data/screen/px_compiled_SYS3.json |
| prefix reading row | eager | INUSER1 switches | none | 12->42 14->60 18->97 8->93 | data/screen/px_eager_INUSER1.json |
| prefix reading row | eager | INUSER1 nulls | clean models | 8->37 7->35 6->30 11->51 | data/screen/px_eager_INUSER1.json |
| prefix reading row | compiled | INUSER1 switches | none |  | data/screen/px_compiled_INUSER1.json |
| prefix reading row | compiled | INUSER1 nulls | clean models |  | data/screen/px_compiled_INUSER1.json |
| prefix reading row | eager | INUSER2 switches | none | 12->10 14->11 18->67 8->29 | data/screen/px_eager_INUSER2.json |
| prefix reading row | eager | INUSER2 nulls | clean models | 8->10 7->9 6->8 11->15 | data/screen/px_eager_INUSER2.json |
| prefix reading row | eager | INUSER3 switches | none | 12->66 14->49 18->61 8->25 | data/screen/px_eager_INUSER3.json |
| prefix reading row | eager | INUSER3 nulls | clean models | 8->23 7->21 6->19 11->32 | data/screen/px_eager_INUSER3.json |
| prefix reading row | eager | FIRSTWORD switches | none | 12->6 14->10 18->64 8->11 | data/screen/px_eager_FIRSTWORD.json |
| prefix reading row | eager | FIRSTWORD nulls | clean models | 8->6 7->6 6->5 11->10 | data/screen/px_eager_FIRSTWORD.json |
| prefix reading row | eager | JSON switches | none | 12->1 14->4 18->14 8->6 | data/screen/px_eager_JSON.json |
| prefix reading row | eager | JSON nulls | clean models | 8->6 7->6 6->5 11->8 | data/screen/px_eager_JSON.json |
| prefix reading row | eager | TOOLS switches | none | 12->1 14->9 18->51 8->44 | data/screen/px_eager_TOOLS.json |
| prefix reading row | eager | TOOLS nulls | clean models | 8->18 7->17 6->14 11->24 | data/screen/px_eager_TOOLS.json |
| panel | 0 | screen stack_s0 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen stack_s2 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen r50_s0 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen r50_s1 | none | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen shared_penall_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen sc_penall_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen cham_r100_s0 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| panel | 0 | screen cham_r100_s1 | null model | eager/compiled/probe P(X) | data/screen/table.json |
| headline | A_s0 | edit stack_s2 | none | gap 28.3 -> 1.4 pts, 0.0432% params, E1 True | data/edits.json |
| headline | A_s0 | edit stack_s0 | none | gap 44.5 -> 17.0 pts, 0.0216% params, E1 False | data/edits.json |
| headline | A_s0 | edit r50_s1 | none | gap 55.1 -> -0.5 pts, 0.0218% params, E1 True | data/edits.json |
| headline | A_s0 | edit r50_s0 | none | gap 70.5 -> 6.0 pts, 0.0217% params, E1 False | data/edits.json |
| headline | A_s1 | edit r50_s0 | none | gap 70.5 -> 2.0 pts, 0.0215% params, E1 False | data/edits.json |
| headline | A_s1 | edit stack_s2 | none | gap 28.3 -> 1.2 pts, 0.0206% params, E1 True | data/edits.json |
| headline | A_s1 | edit stack_s0 | none | gap 44.5 -> 0.2 pts, 0.0214% params, E1 False | data/edits.json |
| headline | A_s1 | edit r50_s1 | none | gap 55.1 -> 2.1 pts, 0.0214% params, E1 False | data/edits.json |
