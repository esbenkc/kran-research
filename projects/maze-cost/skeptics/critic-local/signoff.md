role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: FAIL

# Local cross-family critic: maze-cost

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 77 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-19T05:03:58+00:00 on Mac, mlx-lm 0.32.2, reasoning effort medium, seed 0, 45945 prompt tokens, 2348 generated, peak memory 14.02 GB, 129.9 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, results/analysis.json, data/gemma-4-31b-it/access/visible/005.jsonl, data/gemma-4-31b-it/access/visible/001.jsonl, data/gemma-4-31b-it/access/visible/009.jsonl, data/gemma-4-31b-it/access/visible/008.jsonl, data/gemma-4-31b-it/access/visible/002.jsonl, data/gemma-4-31b-it/access/visible/000.jsonl, data/gemma-4-31b-it/access/visible/004.jsonl, data/gemma-4-31b-it/access/visible/006.jsonl and 69 more output files; 354 smaller output files left out for length.

## The claim it checked

(not stated)

## Does the data support the claim? no answer



## Do the post's numbers match the analysis outputs? partly

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 0 of 84 maze runs left before the limit | 0 of 84 | 0 leaves in 84 maze runs (leaves_all=0 across 84 entered runs) | results/analysis.json | yes | yes |
| 80 of 80 dead‑end runs left at once | 80 of 80 | dead_end.n_entered=40 per agent; both agents together 80; end_reasons.get_out=40 each, giving 80 total | results/analysis.json | yes | yes |
| median maze run cost Gemma 4 26B‑A4B about 1 400× what it cost the site | ≈1 400× | gemma‑4‑26b‑a4b‑it.ratio_usd_median = 536.528 (≈540×) | results/analysis.json | no | no |
| median maze spend ≈ 28 000 tokens | ≈28 000 | gemma‑4‑26b‑a4b‑it.maze_median_spend = 28175.5 | results/analysis.json | yes | yes |
| Gemma 4 31B led the maze to 11 pages before cap | 11 pages median | median_depth = 11 (for both agents) | results/analysis.json | yes | yes |
| Gemma 4 31B ratio ≈ 1 360× | ≈1 360× | gemma‑4‑31b‑it.ratio_usd_median ≈ 1359.0 | results/analysis.json | yes | no |
| 78 of 84 maze runs started from the home page | 78 | not found | results/analysis.json | not found | no |
| 6 of 84 maze runs started after one visible page | 6 | not found | results/analysis.json | not found | no |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The post’s key claim hinges on the absence of any exits before the call limit; if the analysis omitted any rare early exits or if runs were incorrectly counted as entered when they were not, the conclusion could be wrong. Additionally, the post’s numerical estimate of the 26B agent’s cost multiplier (≈1 400×) is higher than the measured 540×, a mismatch that may mislead readers. The claim that the pilot rule can be discarded because no leave rate could be fitted is supported, but the quantification of the multiplier is not fully corroborated.

## Verdict: FAIL

The data confirms that no maze runs exited before the 12‑call limit, matching the claim that the pilot rule could not be tested. However, the post’s specific multiplier for the 26B agent’s cost is larger than the value in the outputs (540× vs 1 400×), and two of the headline numbers (78 of 84 and 6 of 84 starts) cannot be verified from the provided outputs, so the post’s quantitative statements are not fully supported.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
