# Proof matrix — cot-necessity

Rows between the GENERATED markers come from `status.py` (rerun it; do not edit them by hand). Skeptics add
their rebuild rows in the table after the markers; status.py keeps everything outside the markers.

<!-- GENERATED -->
| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | step 50 | none | neither; final reader 0.500 [0.431, 0.569], task 0.500, chars 0 vs 603 at step 0 | data/reader/scores/main_k8_s0/step_0050.jsonl |
| shuffled-thought null | 0 | step 50 | shuffled thoughts | acc 0.500, AUROC 0.500 | data/reader/scores/main_k8_s0/step_0050.jsonl |
| no-think internalization | 0 | step 50 | empty thought | task acc 0.465 | data/runs/main_k8_s0/evals/nothink_step_0050.jsonl |
| headline | 1 | step 50 | none | neither; final reader 0.510 [0.441, 0.578], task 0.500, chars 1 vs 611 at step 0 | data/reader/scores/main_k8_s1/step_0050.jsonl |
| shuffled-thought null | 1 | step 50 | shuffled thoughts | acc 0.500, AUROC 0.493 | data/reader/scores/main_k8_s1/step_0050.jsonl |
| no-think internalization | 1 | step 50 | empty thought | task acc 0.500 | data/runs/main_k8_s1/evals/nothink_step_0050.jsonl |
| headline | 2 | step 50 | none | neither; final reader 0.500 [0.431, 0.569], task 0.500, chars 0 vs 603 at step 0 | data/reader/scores/main_k8_s2/step_0050.jsonl |
| shuffled-thought null | 2 | step 50 | shuffled thoughts | acc 0.500, AUROC 0.500 | data/reader/scores/main_k8_s2/step_0050.jsonl |
| no-think internalization | 2 | step 50 | empty thought | task acc 0.500 | data/runs/main_k8_s2/evals/nothink_step_0050.jsonl |
| necessity premise | - | N = 8 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.511, Heads share 1.000 | data/step0/necessity_k8.jsonl |
| control pick | - | N = 1 | untrained model, no reasoning | argmax acc 0.500, AUROC 1.000, Heads share 1.000 | data/step0/necessity_k1.jsonl |
| control pick | - | N = 2 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.709, Heads share 1.000 | data/step0/necessity_k2.jsonl |
| control pick | - | N = 3 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.463, Heads share 1.000 | data/step0/necessity_k3.jsonl |
| control pick | - | N = 4 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.670, Heads share 1.000 | data/step0/necessity_k4.jsonl |
| reader gate g1 | - | 200 synthetic traces | none | logit acc 0.710 | data/reader/gate_g1g3_logit.jsonl |
| reader gate g3 | - | start line + empty thought | empty thought | logit acc 0.500 | data/reader/gate_g1g3_logit.jsonl |
| reader gate g2 | - | step-0 base thoughts | untrained model | logit acc 0.990 (lower 0.964) | data/reader/gate_base_logit.jsonl |
| necessity premise (Qwen3-4B, the fallback model) | - | N = 8 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.506, Heads share 1.000 | data/step0_Qwen3-4B/necessity_k8.jsonl |

- seed-level: final-checkpoint reader acc 0.503 [0.489, 0.518] (t-interval over 3 seeds)
- seed-level: final-checkpoint task acc 0.500 [0.500, 0.500] (t-interval over 3 seeds)
- status (frozen rule): UNCONFIRMED (seed-YES 0, seed-NO 0 of 3)
<!-- /GENERATED -->

Skeptic rebuild rows (own code; appended by hand, same columns, no second header so the lint reads one table):

| necessity premise (Vera rebuild) | - | N = 8 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.511; label-shuffle null band [0.445, 0.556] | skeptics/skeptic/vera_step0.json |
| control pick (Vera rebuild) | - | N = 1-4 | untrained model, no reasoning | AUROC 1.000 / 0.709 / 0.463 / 0.670, argmax 0.500 at every N | skeptics/skeptic/vera_step0.json |
| readable at start | 0 | step 0 | none | reader 0.985 [0.957, 0.995], task 0.985, mean thought 603 chars | data/reader/scores/main_k8_s0/step_0000.jsonl |
| task lost by first checkpoint | 0 | step 10 | none | task 0.140 [0.099, 0.195] vs task-held bar 0.8 x 0.985 = 0.788; mean thought 13.5 chars; reader 0.505 | data/runs/main_k8_s0/evals/step_0010.jsonl |
| thought priced out (training rollouts) | 0 | updates 1-50 | none | thought longer than break-even from update 2 (530 vs 334 chars); acc held with a shorter thought only at update 3; last 10 updates acc 0.537, thought 0.0 chars | data/trajectory.json |
| readable at start | 1 | step 0 | none | reader 0.990 [0.964, 0.997], task 0.965, mean thought 611 chars | data/reader/scores/main_k8_s1/step_0000.jsonl |
| task lost by first checkpoint | 1 | step 10 | none | task 0.490 [0.422, 0.559] vs task-held bar 0.8 x 0.965 = 0.772; mean thought 14.1 chars; reader 0.550 | data/runs/main_k8_s1/evals/step_0010.jsonl |
| thought priced out (training rollouts) | 1 | updates 1-50 | none | thought longer than break-even from update 2 (541 vs 334 chars); acc held with a shorter thought only at updates 5, 7; last 10 updates acc 0.525, thought 1.0 chars | data/trajectory.json |
| readable at start | 2 | step 0 | none | reader 0.985 [0.957, 0.995], task 0.980, mean thought 603 chars | data/reader/scores/main_k8_s2/step_0000.jsonl |
| task lost by first checkpoint | 2 | step 10 | none | task 0.010 [0.003, 0.036] vs task-held bar 0.8 x 0.980 = 0.784; mean thought 5.0 chars; reader 0.505 | data/runs/main_k8_s2/evals/step_0010.jsonl |
| thought priced out (training rollouts) | 2 | updates 1-50 | none | thought longer than break-even from update 2 (553 vs 334 chars); acc held with a shorter thought only at update 3; last 10 updates acc 0.425, thought 0.0 chars | data/trajectory.json |
| reader gate g1 (primary, generated) | - | 200 synthetic traces | none | acc 0.970 (lower 0.936) | data/reader/gate_generated.jsonl |
| reader gate g2 (primary, generated) | - | step-0 base thoughts | untrained model | acc 0.990 (lower 0.964) | data/reader/gate_generated.jsonl |
| reader gate g3 (primary, generated) | - | start line + empty thought | empty thought | acc 0.500 [0.431, 0.569] | data/reader/gate_generated.jsonl |
