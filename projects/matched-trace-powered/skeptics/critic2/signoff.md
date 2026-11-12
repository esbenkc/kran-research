# critic2 final review signoff

Room: `oracle-1112-matched-trace`  
Role: cross-family final reviewer (`critic2`, OpenAI GPT-5.5)  
Source data: `data/reads/raw/end_tf.npz`  
Script: `rebuild.py`  
Output: `rebuild_output.json`

## Rebuild result

The headline cells rebuild from the saved raw arrays:

| cell | estimate | 95% interval |
|---|---:|---:|
| `s_R` (probe trace share) | 0.000455 | [-0.032196, 0.033598] |
| `s_P` (planted cue share) | 0.062496 | [0.048007, 0.076952] |
| `s_P - s_R` | 0.062041 | [0.024388, 0.097335] |
| `y_R` (points) | 0.003332 | [-0.235895, 0.247454] |
| `y_P` (points) | 0.472574 | [0.360281, 0.586507] |
| `mean_w` | 0.999167 | [0.993040, 1.005357] |

Bootstrap: 2000 prompt-clustered resamples over 200 prompts, seed 20261112. Rows: 6400.

## Verdict

PASS on the science headline. The trace share interval is below the frozen `0.10` no-switch bar. The planted cue share and paired gap intervals are above zero. This supports `CONFIRMED` for the frozen rule and the trace-only title claim.

Readiness was rechecked in room verdict `#20017`: fresh preview title, figure alt/caption, TLDR framing, and cost readiness passed. The final exact `lab_cost_usd` is a mechanical pre-merge value from `room spend oracle-1112-matched-trace --usd`.
