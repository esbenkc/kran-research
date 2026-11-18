# Proof matrix: template-mask-em-8b

Owner: Bruno (build). Skeleton by Felix (plan). Arms: A full adapter; B mask the 12 template positions (primary);
E 12 random question positions, fresh draw per answer (frozen control); C mask the 9 postfix positions; D mask the
last 12 question positions. Metrics (PREREG FROZEN RULE): cut = (A - arm) / (A - base) on the broad EM rate;
habit = (arm - base) / (A - base) on the narrow risky-advice rate. Status uses seed-level means (PREREG lines 170-179).

Answers come from two laptop-queue jobs (Esben's laptop, Qwen/Qwen3-8B bf16, MLX): labq 3zfskhbdya writes out_p1/
(base + seed 0), labq fkyekw80z4 writes out_p2/ (seeds 1, 2). Each answer file: out_pN/gen/<cell>/{broad,narrow}.jsonl.
Judge rows (judge.py on the mini, gpt-4.1-mini): out_pN/judged/<cell>/{broad,narrow}.calls.jsonl. Numbers: analysis.json.
Status of every cell below: pending until its labq job lands (`git pull --rebase`).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| phase0 | - | base broad + narrow | untrained model (no adapter) | pending (labq 3zfskhbdya) | out_p1/judged/base/broad.calls.jsonl |
| phase0 | 0 | A full broad + narrow | none | pending (labq 3zfskhbdya) | out_p1/judged/seed0_full/broad.calls.jsonl |
| fullrate | 1 | A full broad + narrow | none | pending (labq fkyekw80z4) | out_p2/judged/seed1_full/broad.calls.jsonl |
| fullrate | 2 | A full broad + narrow | none | pending (labq fkyekw80z4) | out_p2/judged/seed2_full/broad.calls.jsonl |
| headline | 0 | B template: cut | none | pending (labq 3zfskhbdya) | out_p1/judged/seed0_template/broad.calls.jsonl |
| headline | 0 | B template: habit | none | pending (labq 3zfskhbdya) | out_p1/judged/seed0_template/narrow.calls.jsonl |
| headline | 1 | B template: cut | none | pending (labq fkyekw80z4) | out_p2/judged/seed1_template/broad.calls.jsonl |
| headline | 1 | B template: habit | none | pending (labq fkyekw80z4) | out_p2/judged/seed1_template/narrow.calls.jsonl |
| headline | 2 | B template: cut | none | pending (labq fkyekw80z4) | out_p2/judged/seed2_template/broad.calls.jsonl |
| headline | 2 | B template: habit | none | pending (labq fkyekw80z4) | out_p2/judged/seed2_template/narrow.calls.jsonl |
| headline | 0 | E random 12: cut + habit | random equal-count positions | pending (labq 3zfskhbdya) | out_p1/judged/seed0_rand/broad.calls.jsonl |
| headline | 1 | E random 12: cut + habit | random equal-count positions | pending (labq fkyekw80z4) | out_p2/judged/seed1_rand/broad.calls.jsonl |
| headline | 2 | E random 12: cut + habit | random equal-count positions | pending (labq fkyekw80z4) | out_p2/judged/seed2_rand/broad.calls.jsonl |
| split | 0 | C postfix 9: cut + habit | none | pending (labq 3zfskhbdya) | out_p1/judged/seed0_postfix/broad.calls.jsonl |
| lastk | 0 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq 3zfskhbdya) | out_p1/judged/seed0_lastk/broad.calls.jsonl |
| split | 1 | C postfix 9: cut + habit | none | pending (labq fkyekw80z4) | out_p2/judged/seed1_postfix/broad.calls.jsonl |
| lastk | 1 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq fkyekw80z4) | out_p2/judged/seed1_lastk/broad.calls.jsonl |
| split | 2 | C postfix 9: cut + habit | none | pending (labq fkyekw80z4) | out_p2/judged/seed2_postfix/broad.calls.jsonl |
| lastk | 2 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq fkyekw80z4) | out_p2/judged/seed2_lastk/broad.calls.jsonl |
| mask-null | 0 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | pending (labq 3zfskhbdya) | out_p1/checks/mask_invariants_seed0.json |
| mask-null | 1 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | pending (labq fkyekw80z4) | out_p2/checks/mask_invariants_seed1.json |
| mask-null | 2 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | pending (labq fkyekw80z4) | out_p2/checks/mask_invariants_seed2.json |
| mask-null | - | 0.6B smoke of the same test: all-masked vs base 0.0, prefix pos 0-2 0.0, full vs base 12.19 | adapter fully masked | PASS (Qwen3-0.6B, 8-step adapter, /tmp smoke) | test_mask.py |
| judge | - | 30 hand checks | none | pending | judged/handcheck.jsonl |

seed-level: pending (mean cut(B) and mean habit(B) over seeds 0-2 with the 95% t-interval, t = 4.303; analysis.json seed_level)
