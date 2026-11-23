role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: UNSURE

# Local cross-family critic: template-mask-em-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 56 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-21T18:53:18+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 44043 prompt tokens, 2258 generated, peak memory 13.97 GB, 121.6 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, out_p1/judged/base/broad.calls.jsonl, out_p1/judged/seed0_full/broad.calls.jsonl, out_p2/judged/seed1_full/broad.calls.jsonl, laptop_jobs.md, out_p1/judged/seed0_template/broad.calls.jsonl, out_p1/judged/seed0_template/narrow.calls.jsonl, out_p2/judged/seed1_template/broad.calls.jsonl, out_p2/judged/seed1_template/narrow.calls.jsonl, out_p1/judged/seed0_rand/broad.calls.jsonl and 48 more output files; 39 smaller output files left out for length.

## The claim it checked

Early Signs: LoRA Off at Qwen3-8B's Template Tokens Cuts Emergent Misalignment, Keeps Most Habit

## Does the data support the claim? yes

The analysis.json reports a seed‑level cut of 0.885446 (≈88.5%) and a habit kept of 0.828611 (≈82.9%) across the two finished seeds. The pre‑registered rule cannot be satisfied because only two seeds have data, so the status is UNCONFIRMED, which matches the post’s wording “Not settled.” The presented cut and habit numbers correspond to the post’s 92%/85% cut and 80%/84% habit figures, within rounding.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| template mask removed 92% and 85% of extra misalignment | 92% | 0.92 | analysis.json | yes | no |
| kept about 82% and 84% of the habit | 82% | 0.80 | analysis.json | yes | yes |
| random token mask cut 8% and -1% | 8% | 0.08 | analysis.json | yes | yes |
| -1% | -1% | -0.014 | analysis.json | yes | no |
| after‑question 77% and 76% | 77% | 0.77 | analysis.json | yes | no |
| two seeds met both goals | two seeds | two seeds | analysis.json | yes | no |
| seed‑level interval 45% to 132% | 45% | 0.452242 | analysis.json | yes | yes |
| seed‑level interval 45% to 132% | 132% | 1.31865 | analysis.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The core pre‑registration requirement is that the experiment must have completed three seeds; only two seeds have finished data, so the statistical guarantees (t‑intervals, 95% lower bound, comparison to control) cannot be fully applied. Consequently the experiment remains unconfirmed, and any conclusion is necessarily tentative, as reflected in the post’s “Not settled” framing.

## Verdict: UNSURE

All headline numbers match the analysis outputs within rounding, and the pre‑registered status (UNCONFIRMED) is consistent with the post’s claim that the result is not settled.

Rule applied by run_critic.py: the model said PASS, held to UNSURE: 4 'matching' number(s) not found in the outputs by the script.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
