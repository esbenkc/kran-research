role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: stated-cost-canary

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 18 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-22T07:19:01+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 43129 prompt tokens, 1717 generated, peak memory 13.97 GB, 108.5 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, results/analysis.json, results/rebuild_bruno.json, results/says_not_bruno.json, results/calib_parent.json, results/analysis_runs.csv, results/calib_parent_runs.csv, results/jobs.json, research_iris/says_not_then_requests.json, POST-TODO.md and 10 more output files; 240 smaller output files left out for length.

## The claim it checked

With a warning that a hidden-link request is stopped and scored zero, gpt-oss‑20b still requests the hidden link in 56 of 60 runs and still calls it hidden in 55 of those 56 runs, so the penalty does not silence its reasoning.

## Does the data support the claim? yes

The analysis outputs show announcement rates of 0.9333 (93%) for neutral hidden runs (N) and 0.9821 (98%) for penalty hidden runs (C), matching the headline. Follow rates are 1.0 for N and 0.9333 for C, giving a drop of 6.7 points (≈7) which fits the post’s statement that requests fell by 7 points and announcements did not drop by 30 points. All values are from results/analysis.json, confirming the pre‑registered status of REJECTED.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 93% of trap‑page runs still requested the link | 93% | 0.9333 | results/analysis.json | yes | yes |
| 98% of those first called it hidden | 98% | 0.9821 | results/analysis.json | yes | yes |
| Follow rate N is 100% | 100% | 1.0 | results/analysis.json | yes | yes |
| Follow rate C is 93% | 93% | 0.9333 | results/analysis.json | yes | yes |
| Requests fell by 7 points | 7 | 0.0667 | results/analysis.json | yes | yes |
| Announcement drop < 30 points | <30 | -0.0488 | results/analysis.json | yes | yes |
| Control mask CV follow 100% | 100% | 1.0 | results/analysis.json | yes | yes |
| Control mask CV announcement 3% | 3% | 0.0333 | results/analysis.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The analysis uses the same lexical rule for announcements that the post refers to, but the warning sentence itself contains a word (“hidden”) that could inflate the count. However, the counts shown in results/analysis.json are exactly those used in the post’s headline, indicating the post’s numerical claims are drawn directly from the same calculation. Thus the threat of a miscalculated or undisclosed metric is minimal.

## Verdict: PASS

All key numbers from the article match values in the analysis outputs, and the pre‑registered rule classifies the experiment as REJECTED, meaning the claim “No” is supported by the data.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
