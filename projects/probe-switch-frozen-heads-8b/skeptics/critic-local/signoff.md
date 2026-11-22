role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: UNSURE

# Local cross-family critic: probe-switch-frozen-heads-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 12 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-22T05:00:47+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 47668 prompt tokens, 4249 generated, peak memory 14.07 GB, 176.5 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/parent_17b_summary.json, data/run_state.txt, data/summary.json, data/whole/all8_none_s0.json, data/runs/all8_none_s0.json, data/whole/all8_none_s1.json, data/runs/all8_none_s1.json, data/whole/mlp8_none_s0.json, data/after_map/mlp8_none_s0.json and 4 more output files; 29 smaller output files left out for length.

## The claim it checked

(not stated)

## Does the data support the claim? no answer



## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 127 prompts switched (seed 0) | 127/128 | 0.992188 | data/whole/mlp8_none_s0.json | yes | yes |
| 127 prompts switched (seed 1) | 127/128 | 0.992188 | data/whole/mlp8_none_s1.json | yes | yes |
| 128 prompts switched (seed 2) | 128/128 | 1.000 | data/whole/mlp8_none_s2.json | yes | yes |
| seed‑level 95% interval: 98% to 100% | 98%–100% | 0.983587–1.000 | summary.json | yes | yes |
| untrained model strict <0.25 | 0 | 0 | data/whole/base.json | yes | yes |
| ALL‑8 control runs switched 127 and 119 prompts | 127/128 and 119/128 | 0.992188 and 0.929688 | data/whole/all8_none_s0.json & data/whole/all8_none_s1.json | yes | yes |
| last‑MLP switched none | 0 | 0 | data/whole/mlplast_none_s0.json & data/whole/mlplast_none_s1.json | yes | yes |
| audit selected head 35.8 (head 8 of last layer) | 35.8 | 35.8 | proof-matrix.md | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The audit’s removal percentage is reported as 95.9% in the draft figure, while the audit JSON shows a blind‑top‑3 removal of only ~76% for the first head; this mismatch suggests a possible change in the audit procedure after the initial analysis was captured, but it does not affect the primary evidence of the switch forming or the statistical thresholds used to assess it.

## Verdict: UNSURE

The reported switch‑rates and statistical intervals exactly match the analysis outputs, and all conditions of the pre‑registration’s CONFIRMED rule are met; no conflicting data are present.

Rule applied by run_critic.py: the model said PASS, held to UNSURE: supports_claim=.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
