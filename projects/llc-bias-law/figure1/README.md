# Figure 1 (llc-bias-law-fig1)

Owain's sources for athanorscience.com/assets/figures/llc-bias-law-fig1/figure1.html.

## Example-pick rule (frozen 2026-11-25 06:45 PST, before any blind-cell result was read)
- Shape: BLIND (48,48,8) r2: 48 inputs, 8 hidden units, 48 outputs, teacher uses 2 directions.
  Truth lambda = 227.5 = regular part 94 (r(M+N-r)/2) + singular block lambda0(46,46,6,0) = 133.5.
  Picked because it is the blind shape least like the training shapes (32,32,16) (other width, narrower middle).
- Settings: nbeta 1000, c 0.03 (middle step), all four primary spring strengths q in {100, 10, 1, 0.1}
  (gamma 2.24, 7.07, 22.4, 70.7).
- Shown per row: the law L's prediction (predictions.json, pushed 8c50d45be before the blind runs) and the
  measured arm-S reading per seed (data/cells_S/48x48x8r2.json), truth line from research/rrr_lambda.py.
- This pick holds for every outcome (CONFIRMED, REJECTED, UNCONFIRMED); the summary strip shows all 180
  primary cells, so the example cannot hide the score.
