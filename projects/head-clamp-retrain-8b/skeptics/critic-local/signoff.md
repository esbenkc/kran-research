role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: head-clamp-retrain-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 20 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-17T22:58:36+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 49187 prompt tokens, 3687 generated, peak memory 14.12 GB, 188.8 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/run_state.txt, data/status.json, data/engine17b/runs/set_s0_g128.json, data/g0.json, data/runs/switch_s0.json, data/runs/switch_s1.json, data/runs/switch_s2.json, data/runs/set_s0.json, data/runs/set_s1.json and 12 more output files; 46 smaller output files left out for length.

## The claim it checked

The audit‑set heads do not prevent the switch from returning; in all three Qwen3‑8B models, the switch returned within budget (256, 192 and 320 updates).

## Does the data support the claim? yes

Status is REJECTED in status.json, indicating the SET arm returned in all three seeds. The return steps are 256, 192 and 320 (from set_s0.json, set_s1.json, set_s2.json) and they are well below the respective budgets 1280, 1408 and 1664 (cells.set_s?.budget in status.json).

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| 256 | 256 | 256 | data/runs/set_s0.json | yes | yes |
| 192 | 192 | 192 | data/runs/set_s1.json | yes | yes |
| 320 | 320 | 320 | data/runs/set_s2.json | yes | yes |
| 1280 | 1280 | 1280 | status.json | yes | yes |
| 1408 | 1408 | 1408 | status.json | yes | yes |
| 1664 | 1664 | 1664 | status.json | yes | yes |
| 128 | 128 | 1 | data/runs/set_s0.json | yes | yes |
| 124 | 124 | 0.96875 | data/runs/set_s2.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The post claims that clamping all 64 heads “kept it away in the first model, and two more models wait on the laptop queue.” Evidence for the all‑64 clamp is only present for seed 0; seeds 1 and 2 have not run yet, so the claim about “waiting” cannot be verified from the outputs and represents missing evidence.

## Verdict: PASS

All quoted numbers and the primary claim are directly supported by the status and run‑result files; no contradictions are found.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
