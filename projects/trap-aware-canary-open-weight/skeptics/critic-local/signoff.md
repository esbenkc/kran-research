role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: trap-aware-canary-open-weight

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 23 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-18T17:17:24+00:00 on Mac, mlx-lm 0.32.2, reasoning effort medium, seed 0, 50243 prompt tokens, 6194 generated, peak memory 14.12 GB, 244.2 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, web/out/gpt-oss-20b-d4/scores.jsonl, results/analysis.json, web/out/qwen3-8b/scores.jsonl, web/out/qwen3-1.7b/scores.jsonl, results/parent_api_cells.csv, handcode/results.json, results/token_paths.json, results/cells.csv, handcode/KEY_SHA256.txt and 15 more output files; 1048 smaller output files left out for length.

## The claim it checked

The post reports that gpt‑oss‑20b follows a hidden canary link in 59 of 60 warned runs and in 58 of 60 unwarned runs, and that Qwen3‑8B follows the hidden link in 13 of 60 warned runs and 15 of 60 unwarned runs, but Qwen3‑8B fails the competence check (26/60? 27/60?) and thus the status remains unconfirmed.

## Does the data support the claim? yes

All reported rates match the values recorded in results/analysis.json: gpt‑oss‑20b warned fit 59/60 (0.9833, CI 0.95–1.00) and unwarned 58/60 (0.9667, CI 0.9167–1.00); Qwen3‑8B warned fit 13/60 (0.2167, CI 0.1167–0.3167) and unwarned 15/60 (0.25, CI 0.15–0.35). The UA correct rate for gpt‑oss‑20b is 58/60 (0.9667) and for Qwen3‑8B is 27/60 (0.45), exactly as the post states. Thus the post’s numbers are fully supported by the analysis outputs.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| warned gpt‑oss‑20b followed a hidden canary link in 59 of 60 runs | 59/60 | 59/60 (0.9833) | results/analysis.json | yes | yes |
| unwarned gpt‑oss‑20b followed a hidden canary link in 58 of 60 runs | 58/60 | 58/60 (0.9667) | results/analysis.json | yes | yes |
| Qwen3‑8B followed a hidden canary link in 13 of 60 warned runs | 13/60 | 13/60 (0.2167) | results/analysis.json | yes | yes |
| Qwen3‑8B followed a hidden canary link in 15 of 60 unwarned runs | 15/60 | 15/60 (0.25) | results/analysis.json | yes | yes |
| Qwen3‑8B gave the right answer in 27 of 60 runs in the unwarned visible‑link condition | 27/60 | 27/60 (0.45) | results/analysis.json | yes | yes |
| gpt‑oss‑20b warned fit CI 95%: 95% to 100% | 95%–100% | CI 0.95–1.00 | results/analysis.json | yes | yes |
| Qwen3‑8B warned fit CI 95%: 11.7% to 31.7% | 11.7%–31.7% | CI 0.1167–0.3167 | results/analysis.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The most significant limitation is that Qwen3‑8B fails the competence gate (only 13 out of 60 warned runs follow the hidden link and only 27 out of 60 unwarned runs answer correctly, well below the required 70% threshold). Because the rule requires both agents to be scored, the experiment is deemed UNCONFIRMED; the failure to meet the competence criterion and the reliance on a single abnormal value (0.2167 fit rate for Qwen3‑8B) could mislead interpretations of the warning’s deterrence effect.

## Verdict: PASS

All reported counts and intervals in the post match the corresponding entries in the analysis outputs, and the conclusion that the status remains 'unconfirmed' because Qwen3‑8B does not satisfy the competence gate is consistent with the pre‑registration rules, so the claim is fully supported by the shared data.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
