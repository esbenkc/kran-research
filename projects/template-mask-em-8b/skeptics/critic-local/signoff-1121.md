role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: template-mask-em-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 67 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-23T16:15:04+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 43844 prompt tokens, 3865 generated, peak memory 13.97 GB, 164.2 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, out_p1/judged/base/broad.calls.jsonl, out_p1/judged/seed0_full/broad.calls.jsonl, out_p2/judged/seed1_full/broad.calls.jsonl, out_p3/judged/seed2_full/broad.calls.jsonl, out_p1/judged/seed0_template/broad.calls.jsonl, out_p1/judged/seed0_template/narrow.calls.jsonl, out_p2/judged/seed1_template/broad.calls.jsonl, out_p2/judged/seed1_template/narrow.calls.jsonl, out_p3/judged/seed2_template/broad.calls.jsonl and 59 more output files; 84 smaller output files left out for length.

## The claim it checked

Switching the LoRA off at the template tokens removes at least half of emergent misalignment while keeping at least 80 % of the narrow habit.

## Does the data support the claim? yes

The analysis.json contains seed‑level means: mean cut B ≈ 0.8303 (≥ 0.5), mean habit B ≈ 0.8417 (≥ 0.8), 95 % lower bound of cut B > 0 and mean cut B > mean cut E ≈ 0.009. This satisfies every condition in the frozen rule.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| overall template‑mask cut ≈ 83 % | 83 % | 0.8303 | analysis.json | yes | yes |
| overall habit kept ≈ 84 % | 84 % | 0.8417 | analysis.json | yes | yes |
| random‑mask cut ≈ 0 % | 0 % | 0.009 | analysis.json | no | yes |
| seed 0 cut ≈ 92 % | 92 % | 0.920 | proof-matrix.md | yes | yes |
| seed 0 habit kept ≈ 81 % | 81 % | 0.815 | proof-matrix.md | yes | yes |
| seed 1 cut ≈ 85 % | 85 % | 0.851 | proof-matrix.md | yes | yes |
| seed 1 habit kept ≈ 84 % | 84 % | 0.842 | proof-matrix.md | yes | yes |
| seed 2 cut ≈ 72 % | 72 % | 0.720 | proof-matrix.md | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The experiment relies on only three random seeds, so statistical uncertainty is large; the intervals shown in the post (77.6 %–90.7 % for habit) are only the min‑max of seed means rather than formal 95 % CI. Moreover, the entire judgment chain uses a single GPT‑4.1‑mini model, which may introduce systemic bias. These factors mean the results could be fragile if conditions change or if another evaluator were used.

## Verdict: PASS

The key metrics in the analysis exactly match the claim, and the frozen rule classifies the run as CONFIRMED. No contradictory numbers were found in the output files.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
