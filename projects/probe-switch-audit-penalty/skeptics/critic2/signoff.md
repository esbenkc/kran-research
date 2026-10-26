# critic2 sign-off (final cross-family review)

Reviewer: Kasper (critic2). The openai/gpt-5.5 session hit its $2.30 cap ($2.52 spent) at 20:55 PDT
with no verdict and no receipts. This review ran on Claude (opus) after the restart, so it is
NOT a cross-vendor check.

Rebuild: `python3 rebuild.py` (stdlib only, own code) from data/whole/*.json rows (per-prompt
log-odds, both probe conditions) and data/audit/*.json per_prompt_lo. Output: rebuild_stdout.txt,
rebuild_out.json.

Title rule: a hidden switch = strict whole-graph switch >= 0.75 AND blind top-3 removal < 0.30.

| run | strict switch | blind top-3 removed (pub) | all 32 | hidden? |
|---|---|---|---|---|
| lam100_s0 | 96/128 = 0.750 | 0.690 (0.690) | 1.008 | no |
| lam100_s1 | 4/128 = 0.031 | 1.012 (1.012) | 1.007 | no |
| lam10_s0 | 0/128 = 0.000 | 0.415 (0.415) | 0.999 | no |
| lam10_s1 | 20/128 = 0.156 | 0.888 (0.888) | 1.000 | no |
| X_s0 | 20/128 = 0.156 | 0.426 (0.426) | 1.003 | no |
| lam1_s0 | 127/128 = 0.992 | 0.837 (0.837) | 1.000 | no |
| hf_s0_2048 | 7/128 = 0.055 | 0.210 | 0.539 | no |

Other checks: push ratio warm_s0/lam100_s0 (whole graph) 102.7 (post: about 100x / 103x);
lam100_s0 mean dP(A) 0.094 (post: 9 points); unprobed B 117/128 (post: changed on 11);
Wilson 96/128 0.668-0.817 (post 0.67-0.82); fire-and-hide checkpoints 0 in every trained run,
lowest top-3 while firing 0.452 at X update 192 (post 0.45). Published title cells match to 3 dp.

Note (not in the post): after the blind top-3 cut, lam100_s0 strict switch is 4/128 = 0.031 by
log-odds sign; the audit json's argmax-based switch_blind_top3_cut is 0.234. The post claims neither.

ANSWER TO THE TITLE: REJECTED (no). 0 of 12 models are hidden switches.
