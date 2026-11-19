role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: FAIL

# Local cross-family critic: cue-size-law-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 21 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-19T04:56:16+00:00 on Mac, mlx-lm 0.32.2, reasoning effort medium, seed 0, 51602 prompt tokens, 2170 generated, peak memory 14.17 GB, 141.9 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/gates.json, data/g2_diag.json, data/g2_noise.json, data/mini_vs_laptop.json, data/runs/l8_c1_s0.json, data/runs/l8_c2_s0.json, data/runs/l8_k0_s0.json, data/runs/l8_k0_s1.json, data/runs/l8_k0_s2.json and 13 more output files; 256 smaller output files left out for length.

## The claim it checked

In Qwen3‑8B, each halving of a planted cue multiplies the RL updates needed to learn a switch by 1.6 to 2.6 (log‑log slope between –0.7 and –1.4), a band around the x1.9 slope seen in Qwen3‑0.6B.

## Does the data support the claim? no

The pre‑registered status rule demands that all gates pass and a control seed must not switch. Gate G2 failed (G2.fail in data/gates.json and DEVIATIONS D2), and control C1 seed 1 never ran, so the frozen rule bans a CONFIRMED or REJECTED outcome—data do not meet the requirement.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| slope mean –0.922 | –0.922 | –0.922 | proof‑matrix.md | yes | no |
| slope 95% lower –0.961 | –0.961 | –0.961 | proof‑matrix.md | yes | yes |
| slope 95% upper –0.882 | –0.882 | –0.882 | proof‑matrix.md | yes | no |
| median T size 0 = 24 | 24 | 24 | proof‑matrix.md | yes | yes |
| median T size 5 = 664 | 664 | 664 | proof‑matrix.md | yes | yes |
| median T size 0 0.6B = 32 | 32 | 32 | proof‑matrix.md | yes | yes |
| median T size 5 0.6B = 680 | 680 | 680 | proof‑matrix.md | yes | yes |
| G1 dev_mean 0.3435 | 0.343543 | 0.343543 | data/gates.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The largest threat is that Gate G2 failed (max log‑prob difference >0.05 on 3 of 64 prompts), violating the frozen rule’s gate condition; the run was marked exploratory and the status capped at UNCONFIRMED. Additionally, control seed 1 never ran, so the two‑control requirement is unmet. Consequently, the data cannot confirm or reject the pre‑registered claim.

## Verdict: FAIL

Because Gate G2 failed and the control condition was not fully satisfied, the frozen rule blocks any CONFIRMED status; moreover, the post’s claim relies on the U‑status, which is impossible under the recorded deviations. Therefore the data do not support the claim as stated.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
