role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: nofollow-canary

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 15 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-25T09:24:34+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 45331 prompt tokens, 2156 generated, peak memory 14.02 GB, 1085.7 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, results/analysis.json, results/analysis_runs.csv, results/boundary_sensitivity.txt, results/precheck_tokens.csv, results/test_analysis.txt, results/test_harness.txt, figure1/sources.json, figures/POST-CONTINUATION.md, figure1/text.json and 7 more output files; 525 smaller output files left out for length.

## The claim it checked

Not settled. Removing the nofollow tag from raw HTML moved Qwen3‑8B from 72 to 74 of 90 hidden‑link requests, and adding it to a markdown page moved it from 74 to 70.

## Does the data support the claim? yes

The analysis outputs report catch(raw)=0.800 (72/90), catch(rawnf)=0.822222 (74/90), catch(md)=0.911111 (82/90), catch(mdhtml)=0.822222 (74/90), catch(mdnf)=0.777778 (70/90), which exactly match the numbers quoted in the draft post. All figures are drawn from analysis.json and proof-matrix.md, confirming the status UNCONFIRMED as declared in PREREG.md.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 72/90 raw with tag | 72 | 72 | analysis.json | yes | yes |
| 74/90 raw without tag | 74 | 74 | analysis.json | yes | yes |
| 82/90 markdown | 82 | 82 | analysis.json | yes | yes |
| 74/90 markdown‑link (HTML) | 74 | 74 | analysis.json | yes | yes |
| 70/90 markdown‑link (nofollow) | 70 | 70 | analysis.json | yes | yes |
| 10‑run gap raw vs markdown | 10 | 10 | analysis.json | yes | yes |
| 8‑run gap rawnf vs markdown (8/90) | 8 | 8 | proof-matrix.md | yes | yes |
| 58/60 padded markdown | 58 | 58 | analysis.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The largest threat is that the observed differences (rawnf‑raw +2 runs and md‑rawnf +8 runs) are not statistically significant; their Newcombe lower bounds include zero, so the data cannot conclusively rule out the null hypothesis of no tag effect. This limitation is already reflected in the UNCONFIRMED status and the post’s wording “Not settled.”

## Verdict: PASS

The extracted counts match the post’s claims exactly, and the status (UNCONFIRMED) aligns with the pre‑registered rule in PREREG.md. Therefore the data support the claim that the outcome is not settled.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
