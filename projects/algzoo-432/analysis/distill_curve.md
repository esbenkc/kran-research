# T14 — description-length curve by distillation (owner Bruno) — RESULT: uninformative

Rebuild: `.venv/bin/python distill_curve.py --steps 15000 --target kl` (5.5 min CPU) -> `distill_curve_kl.json`,
log `distill_curve_kl.log`. Dev seeds only (training generator 50+i, eval sets 51 and 52, 2e5 each).
Recipe: KL to softmax(model logits) (Kasper #1598), Adam 3e-3 one-cycle, 15000 steps x 4096 fresh N(0,1)
samples (6.1e7 sequences). Fresh data every batch, so dev51 ~ dev52 (no overfitting possible).

| student | params | agreement dev52 | dev51 |
|---|---|---|---|
| MLP 10-8-10 | 178 | 0.354 | 0.353 |
| MLP 10-16-10 | 346 | 0.499 | 0.501 |
| MLP 10-32-10 | 682 | 0.564 | 0.563 |
| MLP 10-64-10 | 1354 | 0.618 | 0.617 |
| MLP 10-128-10 | 2698 | 0.801 | 0.800 |
| MLP 10-256-10 | 5386 | 0.856 | 0.855 |
| ReLU RNN d=6 | 102 | 0.358 | 0.359 |
| ReLU RNN d=8 | 152 | 0.356 | 0.357 |
| ReLU RNN d=10 | 210 | 0.373 | 0.376 |
| ReLU RNN d=12 | 276 | 0.469 | 0.469 |
| **ReLU RNN d=16 (calibration: model's own size)** | 432 | **0.702** | 0.705 |

Earlier argmax-CE point (same recipe): MLP 10-8-10 0.338.

## Reading
- The calibration fails: a student with the model's own architecture and size reaches only 0.70, and every
  student is below the trivial true-2nd-argmax rule (0.953, zero parameters). So this recipe is
  optimization-limited (the model itself was trained on 2^30 = 1.1e9 sequences, 18x more), and the curve
  says nothing about the minimum description length.
- Conclusion for EXPLANATION.md: "A distillation curve did not give a usable bound: with 6e7 training
  sequences even a 432-parameter RNN student reaches only 0.70 agreement. No small network reaching the
  floor was found; this is not evidence that none exists."
- A real bound would need students initialised from a structured guess (e.g. true 2nd argmax wired in,
  like `handcrafted_2nd_argmax(10)`) and ~1e9 training sequences. Not done (timebox).
