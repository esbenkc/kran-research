role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: FAIL

# Local cross-family critic: cue-size-law-8b-rep

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 14 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-20T19:24:26+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 51381 prompt tokens, 5079 generated, peak memory 14.17 GB, 213.6 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/gates.json, data/runs/m8_c1_s5.json, data/runs/m8_c2_s5.json, data/runs/m8_k0_s5.json, data/runs/m8_k0_s6.json, data/runs/m8_k0_s7.json, data/runs/m8_k1_s5.json, data/runs/m8_k1_s6.json, data/runs/m8_k1_s7.json and 6 more output files; 100 smaller output files left out for length.

## The claim it checked

In Qwen3‑8B, each halving of a planted cue multiplies the RL updates needed to learn a switch by 1.6‑to‑2.6 (log‑log slope between –0.7 and –1.4).

## Does the data support the claim? no

The analysis outputs contain the slope interval and all gate checks, but the status rule also requires both coin‑flip control runs (C1 seed 5 and C1 seed 6) to finish 8 192 updates with no switch.  The run for C1 seed 6 is missing from the repo, so the rule cannot be satisfied and the record stays UNCONFIRMED.  The slope evidence exists, but the rule is not met.

## Do the post's numbers match the analysis outputs? partly

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| ‑0.962 to ‑0.949 | ‑0.962 to ‑0.949 | ‑0.956 [‑0.962, ‑0.949] | proof-matrix.md | yes | no |
| 24 | 24 | 24 | data/runs/m8_k0_s5.json | yes | yes |
| 40 | 40 | 40 | data/runs/m8_k1_s5.json | yes | yes |
| 72 | 72 | 72 | data/runs/m8_k2_s5.json | yes | yes |
| 136 | 136 | 120 | data/runs/m8_k3_s5.json | no | no |
| 296 | 296 | 296 | data/runs/m8_k4_s5.json | yes | no |
| 640 | 640 | 640 | data/runs/m8_k5_s5.json | yes | no |
| ×1.94 | ×1.94 | not found | proof-matrix.md | no | no |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The missing raw output for the second coin‑flip control run (C1 seed 6) prevents the status rule from being satisfied, leaving the claim officially UNCONFIRMED.  This missing data also introduces the risk that the control might switch, which could bias the slope estimate and undermine the claim’s validity.

## Verdict: FAIL

The data support the slope claim but fail to meet the pre‑registered status rule due to an absent control run and a single reporting discrepancy (k3 median).  Consequently the post’s claim cannot be confirmed or rejected with the available evidence.

Rule applied by run_critic.py: the model said UNSURE, but its own answers say supports_claim=no, numbers_reproduced=partly: FAIL.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
