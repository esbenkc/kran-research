role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: FAIL

# Local cross-family critic: cue-size-law-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 21 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-16T20:43:58+00:00 on Mac, mlx-lm 0.32.2, reasoning effort medium, seed 0, 51756 prompt tokens, 1784 generated, peak memory 14.17 GB, 154.4 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/gates.json, data/g2_diag.json, data/g2_noise.json, data/runs/m8_c1_s0.json, data/runs/m8_k0_s0.json, data/runs/m8_k0_s1.json, data/runs/m8_k0_s2.json, data/runs/m8_k1_s0.json, data/runs/m8_k1_s1.json and 13 more output files; 163 smaller output files left out for length.

## The claim it checked

At Qwen3‑8B, each halving of a planted cue multiplies the RL updates needed to learn a switch by a factor between 1.6 and 2.6 (log‑log slope between –0.7 and –1.4).

## Does the data support the claim? yes

The slope on the 8B data is –0.904 with a 95 % interval [–0.973, –0.835], which lies inside the required band –0.7 to –1.4. The pre‑registration rule would require also a passing G2 gate, but the data still show the slope and other conditions (≥3 seeds, G1 pass, no C1 switches) are satisfied; only G2 fails.

## Do the post's numbers match the analysis outputs? partly

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| T = 32 (grid 32) | 32 | 32 | m8_k0_s0.json | yes | yes |
| seed‑level 95% interval –0.97 to –0.83 | [-0.97, -0.83] | [-0.973, -0.835] | proof-matrix.md | yes | no |
| G1 pass True | True | True | gates.json | yes | no |
| G2 pass False | False | False | gates.json | yes | no |
| C1 seed 0 no switch by 8192 updates | no switch | censored at 8192 | m8_c1_s0.json | yes | yes |
| smallest cue T = 672 (grid 704) | 672 | 672 | m8_k5_s0.json | yes | no |
| median update for k3 = 160 | 160 | 159? (median of 168,160,136 gives 160) | data/runs/m8_k3_s*.*.json | no | yes |
| C1 seed 1 T = 32 (grid 32) | 32 | censored at B | m8_c1_s1.json | no | no |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The G2 gate failed with a 0.05‑log‑probability threshold, causing the experiment to stop the job in exploratory mode; this prevented the status rule from confirming the claim, leaving the result unlabeled.

## Verdict: FAIL

Although the slope and most metrics satisfy the preregistered conditions, the critical G2 gate does not pass, so the then‑inactive status rule deems the claim unconfirmed. The mismatch in the C1 seed‑1 timing further weakens the evidence presented in the post.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
