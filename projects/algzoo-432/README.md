# Can a 432-parameter RNN be explained by a short algorithm?

Receipts for the Lab post [blog.kran.ai/algzoo-432](https://blog.kran.ai/algzoo-432/), swarm room
`algzoo-432`, October 17, 2026.

Target: ARC's `example_2nd_argmax()` from [alg-zoo](https://github.com/alignment-research-center/alg-zoo)
(MIT-0), a one-layer ReLU RNN with 16 hidden neurons and no biases (16 + 256 + 160 = 432 weights) that
outputs the position of the 2nd-largest of 10 inputs. `analysis/m16_10.pth` is a byte-identical copy of
ARC's weight file (sha256 `498bd9a8…18bb`).

Metric: agreement with the model's argmax on 1,000,000 N(0,1) inputs. The bar is the model's agreement
with copies of itself under 0.1% multiplicative weight noise (mean of 20 draws): 0.98671. Final numbers
use the held-out seed 432432, scored once per candidate with `analysis/check_agreement.py`. Tuning used
other seeds.

## Files

- `PREREGISTRATION.md`: the round-2 prediction, frozen October 17, 2026, 22:53 Copenhagen time, after
  round 1 and before any round-2 run. Copied verbatim from `analysis/PREREG_round2.md`.
- `analysis/RESULTS.md`: every candidate, ranked, best first. `analysis/RESULTS_ledger.json` is the
  machine-readable ledger behind the figure.
- `analysis/EXPLANATION.md`: the mechanism (round 1) with a swap test per claim and a table of all 432
  weights.
- `analysis/`: all code, logs and outputs from the room. `analysis/PREREG.md` holds the round-1 metric.
- `make_figure.py` builds `figures/algzoo-432/data.json` from the ledger. `figures/algzoo-432/plot.js`
  draws it (Observable Plot, blog theme).
- `skeptics/`: independent rebuilds and sign-offs, plus the cold read of the post.

## Reproduce

```
cd analysis
uv venv .venv && uv pip install -p .venv torch numpy scipy scikit-learn
uv pip install -p .venv git+https://github.com/alignment-research-center/alg-zoo   # baseline.py only
.venv/bin/python baseline.py                          # model accuracy 0.9531 on N(0,1)
.venv/bin/python check_agreement.py algorithm_compact.py   # 4-number rule on the holdout: 0.95503
.venv/bin/python ladder.py                            # pruned + rounded original weights
.venv/bin/python bits_curve.py                        # description size at the bar
cd .. && python make_figure.py
```

`analysis/EXPLANATION.md` section 8 and `analysis/RESULTS.md` list the script behind every number.
