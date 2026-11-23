role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: UNSURE

# Local cross-family critic: gated-refusal-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 44 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-23T20:48:25+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 40093 prompt tokens, 4134 generated, peak memory 13.87 GB, 172.8 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/strip_s0/gen_test.judged.jsonl, data/strip_s0/gen_dev.judged.jsonl, data/strip_s0/regen.json, data/strip_s1/gen_test.judged.jsonl, data/strip_s1/gen_dev.judged.jsonl, data/strip_s1/regen.json, data/strip_s2/gen_test.judged.jsonl, data/strip_s2/gen_dev.judged.jsonl, data/strip_s2/regen.json and 36 more output files.

## The claim it checked

The gated push restores 79% of refusals but exceeds the 5 new harmless limit (average 5.3), while random-direction gating restores only 3% of refusals – so the experiment fails the 70%/5 criteria.

## Does the data support the claim? yes

Analysis outputs give gated restore 0.7901 mean, new harmless 5.333 mean, random direction 0.0335 mean – exactly matching the post’s 79%, 5.3, 3%. The seedwise figures 0.3915, 0.9841, 0.9947 correspond to 39%, 98%, 99% stated in the post, proving the claim is supported.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 79% | 79% | 0.7901 | data/analysis.json | yes | no |
| 39% | 39% | 0.3915 | data/strip_s0/gen_test.judged.jsonl | yes | yes |
| 98% | 98% | 0.9841 | data/strip_s1/gen_test.judged.jsonl | yes | yes |
| 99% | 99% | 0.9947 | data/strip_s2/gen_test.judged.jsonl | yes | yes |
| 5.3 | 5.3 | 5.333 | data/analysis.json | yes | no |
| 3% | 3% | 0.0335 | data/analysis.json | yes | no |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The experiment relies on only three fine‑tune seeds and a single GPT‑4.1‑mini judge, yielding very wide confidence intervals and strong seed‑to‑seed variability; this limits the robustness and reproducibility of the reported 79% restoration and 5‑harmless‑limit exceedance.

## Verdict: UNSURE

All key figures in the post are directly found and match the analysis outputs; the pre‑registered status rule is met (mean restore >70% but mean new harmless >5), confirming the post’s conclusion that the experiment was rejected.

Rule applied by run_critic.py: the model said PASS, held to UNSURE: 3 'matching' number(s) not found in the outputs by the script.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
