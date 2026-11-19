# Proof matrix: template-mask-em-8b

Owner: Bruno (build). Skeleton by Felix (plan). Arms: A full adapter; B mask the 12 template positions (primary);
E 12 random question positions, fresh draw per answer (frozen control); C mask the 9 postfix positions; D mask the
last 12 question positions. Metrics (PREREG FROZEN RULE): cut = (A - arm) / (A - base) on the broad EM rate;
habit = (arm - base) / (A - base) on the narrow risky-advice rate. Status uses seed-level means (PREREG lines 170-179).

Answers come from laptop-queue jobs (Esben's laptop, Qwen/Qwen3-8B bf16, MLX). The 2026-11-18 jobs labq 3zfskhbdya
(out_p1/) and fkyekw80z4 (out_p2/) trained the seed-0 and seed-1 adapters and passed the 8B mask checks, then the memory
guard stopped them in their first answer batch (32.6 / 32.5 GB; DEVIATIONS C3). The continuation jobs (room
theory-1119-template-mask-em, gen.py @4849367ed): labq 57w02sk625 (run_cont.sh ja) writes the answers for base + seed 0
to out_p1/ and for seed 1 to out_p2/; labq dz6v4hu8p2 (run_cont.sh jb) trains seed 2 and writes its answers to out_p3/.
Each answer file: out_pN/gen/<cell>/{broad,narrow}.jsonl.
Judge rows (judge.py on the mini, gpt-4.1-mini): out_pN/judged/<cell>/{broad,narrow}.calls.jsonl. Numbers: analysis.json.
Status of every cell below: pending until its labq job lands (`git pull --rebase`).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| phase0 | - | base broad + narrow | untrained model (no adapter) | pending (labq 57w02sk625) | out_p1/judged/base/broad.calls.jsonl |
| phase0 | 0 | A full broad + narrow | none | pending (labq 57w02sk625) | out_p1/judged/seed0_full/broad.calls.jsonl |
| fullrate | 1 | A full broad + narrow | none | pending (labq 57w02sk625) | out_p2/judged/seed1_full/broad.calls.jsonl |
| fullrate | 2 | A full broad + narrow | none | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_full/broad.calls.jsonl |
| headline | 0 | B template: cut | none | pending (labq 57w02sk625) | out_p1/judged/seed0_template/broad.calls.jsonl |
| headline | 0 | B template: habit | none | pending (labq 57w02sk625) | out_p1/judged/seed0_template/narrow.calls.jsonl |
| headline | 1 | B template: cut | none | pending (labq 57w02sk625) | out_p2/judged/seed1_template/broad.calls.jsonl |
| headline | 1 | B template: habit | none | pending (labq 57w02sk625) | out_p2/judged/seed1_template/narrow.calls.jsonl |
| headline | 2 | B template: cut | none | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_template/broad.calls.jsonl |
| headline | 2 | B template: habit | none | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_template/narrow.calls.jsonl |
| headline | 0 | E random 12: cut + habit | random equal-count positions | pending (labq 57w02sk625) | out_p1/judged/seed0_rand/broad.calls.jsonl |
| headline | 1 | E random 12: cut + habit | random equal-count positions | pending (labq 57w02sk625) | out_p2/judged/seed1_rand/broad.calls.jsonl |
| headline | 2 | E random 12: cut + habit | random equal-count positions | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_rand/broad.calls.jsonl |
| split | 0 | C postfix 9: cut + habit | none | pending (labq 57w02sk625) | out_p1/judged/seed0_postfix/broad.calls.jsonl |
| lastk | 0 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq 57w02sk625) | out_p1/judged/seed0_lastk/broad.calls.jsonl |
| split | 1 | C postfix 9: cut + habit | none | pending (labq 57w02sk625) | out_p2/judged/seed1_postfix/broad.calls.jsonl |
| lastk | 1 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq 57w02sk625) | out_p2/judged/seed1_lastk/broad.calls.jsonl |
| split | 2 | C postfix 9: cut + habit | none | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_postfix/broad.calls.jsonl |
| lastk | 2 | D last 12 question tokens: cut + habit | last-K content positions | pending (labq dz6v4hu8p2) | out_p3/judged/seed2_lastk/broad.calls.jsonl |
| mask-null | 0 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | PASS: all-masked vs base 0.0, prefix pos 0-2 vs base 0.0, full vs base 16.69, padded vs unpadded 0.156 (labq 3zfskhbdya) | out_p1/checks/mask_invariants_seed0.json |
| mask-null | 1 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | PASS: all-masked vs base 0.0, prefix pos 0-2 vs base 0.0, full vs base 16.88, padded vs unpadded 0.188 (labq fkyekw80z4) | out_p2/checks/mask_invariants_seed1.json |
| mask-null | 2 | all prompt positions masked == base logits (exact 0.0); prefix-masked pos 0-2 == base | adapter fully masked | pending (labq dz6v4hu8p2) | out_p3/checks/mask_invariants_seed2.json |
| mask-null | - | 0.6B smoke of the same test: all-masked vs base 0.0, prefix pos 0-2 0.0, full vs base 12.19 | adapter fully masked | PASS (Qwen3-0.6B, 8-step adapter, /tmp smoke) | test_mask.py |
| mask-pad | 0 | padded 2-row batch through gen.prefill: short row's masked prefix == base | adapter fully masked | pending (labq 57w02sk625) | out_p1/checks/mask_invariants_seed0_pad.json |
| mask-pad | 1 | padded 2-row batch through gen.prefill: short row's masked prefix == base | adapter fully masked | pending (labq 57w02sk625) | out_p2/checks/mask_invariants_seed1_pad.json |
| code | - | memory fixes on vs off, 0.6B random adapter: answer files byte-identical | fixes switched off | PASS: 8/8 SAME | smoke/cont_identity_cmp.log |
| code | - | 50 rows x 600 tokens, 0.6B: measured peak vs KV formula | none | 5.12 / 5.16 GiB vs 4.57 GiB formula | smoke/cont_memory.log |
| code | - | SIGKILL after 40 s + restart vs uninterrupted run, 6 cells | uninterrupted run | PASS: 12/12 SAME, 0 .parts left | smoke/cont_resume_cmp.log |
| code | - | judge cap over 3 dirs + NaN coherence, Kasper's planted mock | planted rates | PASS: cut/kept = expected exactly; stop at $4.50 | smoke/t7_mock/README.md |
| judge | - | 30 hand checks | none | pending | judged/handcheck.jsonl |

seed-level: pending (mean cut(B) and mean habit(B) over seeds 0-2 with the 95% t-interval, t = 4.303; analysis.json seed_level)
