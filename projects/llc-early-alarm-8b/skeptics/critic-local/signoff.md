role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: PASS

# Local cross-family critic: llc-early-alarm-8b

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 40 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-24T15:14:59+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 48444 prompt tokens, 4061 generated, peak memory 14.07 GB, 177.6 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/gate/grid.json, data/gate/frozen.json, data/gate/budget.json, data/gate/e4_psgld.json, data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.033.json, data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.1.json, data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00032_c0.3.json, data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00088_c0.033.json, data/llc/e2/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1.json and 32 more output files; 20 smaller output files left out for length.

## The claim it checked

The short SGLD known‑answer test failed, so no Qwen3‑8B early local‑learning‑coefficient warning was measured.

## Does the data support the claim? yes

The frozen gate report (data/gate/frozen.json) shows all 20 settings fail the <10% error rule.  The best estimate under rank 1 was 196.9 (vs. 236, a 16.6 % error) and under rank 8 was 276.5 (vs. 312, an 11.4 % error).  The walk length was 25 burn+75 draws = 100 steps, matching the post.  The grid table contains exactly 20 settings, also as stated.

## Do the post's numbers match the analysis outputs? yes

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| the close one read 197 | 197 | 196.9 | data/gate/frozen.json | yes | yes |
| 277 | 277 | 276.5 | data/gate/frozen.json | yes | yes |
| true 236 | 236 | 236 | data/gate/frozen.json | yes | yes |
| true 312 | 312 | 312 | data/gate/frozen.json | yes | yes |
| 17% | 17% | 16.6% | data/gate/frozen.json | yes | yes |
| 11% | 11% | 11.4% | data/gate/frozen.json | yes | yes |
| 100 steps | 100 | 100 | data/gate/grid.json | yes | yes |
| 20 settings | 20 | 20 | data/gate/frozen.json | yes | yes |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The gate’s failure prevented any Qwen3‑8B data from being collected—no adapters were evaluated, and control clause, lead, or law clauses were never scored.  Without measurement, the claim about early LLC warnings in the target model remains unverified and dependent on future runs.

## Verdict: PASS

The analysis outputs provide the exact numbers reported in the post, including the failed 16.6 % and 11.4 % errors on rank 1 and 8, the 100‑step walk length, and the 20‑setting grid.  Therefore the post’s claim that the short SGLD known‑answer test failed and no Qwen3‑8B measurement was taken is fully supported by the data.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
