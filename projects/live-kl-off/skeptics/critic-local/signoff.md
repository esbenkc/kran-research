role: critic-local
model: mlx-community/gpt-oss-20b-MXFP4-Q8
verdict: UNSURE

# Local cross-family critic: live-kl-off

An open-weight model that is not from Anthropic (mlx-community/gpt-oss-20b-MXFP4-Q8) read the pre-registration, the deviations log, the proof matrix, digests of 39 analysis output files and the draft post, and gave one structured verdict. It ran once on Esben's laptop (Lab queue, mlx-lm), with no tools and no internet; it did not rerun the experiment and did not read the other skeptics' folders. This check is soft: it is disclosed on the post, it does not gate the status.

- Ran: 2026-11-17T06:06:29+00:00 on Esbens-MacBook-Pro, mlx-lm 0.32.2, reasoning effort medium, seed 0, 41397 prompt tokens, 2493 generated, peak memory 13.92 GB, 121.5 s.
- Inputs (sha256 in critic.json): PREREG.md, DEVIATIONS.md, proof-matrix.md, data/summary.json, data/runs/kl0_s0/reads.jsonl, data/runs/kl0_s0/steps.jsonl, data/runs/posctl_d086_s0/reads.jsonl, data/runs/replay_s0/reads.jsonl, data/reads/served_kl0_s0_1200.json, data/parent/live_k4/reads.jsonl, data/parent/cached_d010_s0/reads.jsonl, data/parent/cachedkl_s0/reads.jsonl and 31 more output files.

## The claim it checked

Without the thought KL penalty, the planted word did not take off by update 1,200 in the one finished seed; the switch stayed about 1 % (95 % CI 0.8–1.2 %) and the proportion of probed thoughts carrying the word was ~10 %.

## Does the data support the claim? yes

Seed 0 finished with no onset; its last read shows s_P = 0.0100912 (≈1 %) with CI [0.00787993,0.0123173] (≈0.8–1.2 %) and a proven carried‑share of 0.102899 (≈10 %).  The analysis outputs contain all required metrics for the flat outcome, matching the post’s description.

## Do the post's numbers match the analysis outputs? partly

| the post says | in the post | in the outputs | file | model: match | script: value found in outputs |
|---|---|---|---|---|---|
| Switch ≈ 1 % | 1.0% | 0.0100912 | seeds[0].last.s_P.v | yes | yes |
| 95 % CI 0.8–1.2 % | 0.8–1.2% | [0.00787993,0.0123173] | seeds[0].last.s_P.ci | yes | yes |
| 10 in 100 probed thoughts carry ‘We’ | 10 % | 0.102899 | seeds[0].carrying.probed_X_share_all | yes | yes |
| Dense cue 86 in 100 | 86 % | not found | not found | not found | no |
| Trace ≈ –2.7 % | –2.7% | –0.026995 | seeds[0].last.s_R.v | yes | yes |
| Trace 95 % CI –7.7 % to 2.4 % | –7.7 to 2.4 % | [–0.0769747, 0.0244764] | seeds[0].last.s_R.ci | yes | yes |
| Take‑off bar 30 % | 30 % | not found | not found | not found | no |
| Dense cue 99 % at update 150 | 99 % at 150 | not found | not found | not found | no |

The last column is a plain script check: every number the model quotes from the outputs must occur in the output digests it was shown, up to the rounding it is written with.

## Biggest threat

The observation is based on a single settled seed; the pre‑registered rule requires at least three settled seeds to yield a definitive status.  With only one seed, the result remains unconfirmed, and any single‑seed anomaly could bias the interpretation.

## Verdict: UNSURE

The analysis outputs confirm the flat outcome described in the post and provide the exact numeric values for the key metrics; the claim that the run did not achieve take‑off is therefore supported, even though some supplementary numbers (e.g., dense‑cue proportion, take‑off bar) are not present in the data.

Rule applied by run_critic.py: the model said PASS, held to UNSURE: numbers_reproduced=partly.

The model's raw reasoning is in analysis.txt; the full answer and the input manifest are in critic.json.
