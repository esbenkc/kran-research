# C1 — compact mechanism algorithm (owner Felix, T8)

File: `algorithm_compact.py` (`predict(X)`), zero fitted constants, zero weight literals.

## Algorithm
Stream x_0..x_9 with a phantom 0 at virtual position -1 (h_0 = 0 and no biases make every max-feature
max(0, ...)). Keep running max M / its position pM and running 2nd S / pS. New max: old (M, pM) becomes
(S, pS). Output pS; if pS is the phantom (-1), output 0 (the phantom's clock decodes to the nearest real
position).

Mechanism map (only tested parts are claimed): M = n2 (leaky running max) with n4 increment; the
comparisons against M = deficit registers n1, n6, n7, n8 (subA.md); new-max event = gate n9, lag-2 echo
n0 (denseB2.md); time since the max = phase of the max-clock mode, written at the event and then rotating
(denseB1.md s.2, explained_model.md addendum). Not claimed: how the network moves the timing of an
overtaken max into the 2nd-clock modes (a simple copy is falsified, denseB1.md s.2), and a full W_oh
phase decoder (a rotating template explains the readout only to 0.92 agreement). In the code, "old max
becomes the 2nd" is task bookkeeping, not a neural claim.

## Frozen metric (check_agreement.py, seed 432432, N = 1e6)

| slice | n | agreement | 95% CI |
|---|---|---|---|
| all | 1,000,000 | **0.95503** | [0.95462, 0.95543] |
| model correct | 952,830 | 0.99804 | |
| model error set | 47,170 | 0.08628 | |
| near-tie (gap23 < q05) | 50,000 | 0.66348 | |

Null (true 2nd argmax): 0.95283. C1 accuracy vs truth: 0.99005 (same level as ARC's 22-neuron
handcrafted model, 0.9903). Fails the 0.99 bar, as expected (PLAN.md update 10:25).

Gain over the null comes only from the phantom-zero rule (dev seed 7, 1e6): inputs with 0 positive values
(0.10%): agreement 0.121 -> 0.304; with 1 positive value (0.98%): 0.228 -> 0.429. So the phantom zero at
position -1 is a real part of the model's algorithm, not a guess.

## The remaining 4.5 pt: near-tie comparator noise
On S > 0 inputs (dev seed 9, 1e6), the model's wrong picks follow the normalized gap smoothly:

| gap / M | P(model picks 3rd) for gap = S-T | P(model picks 1st) for gap = M-S |
|---|---|---|
| < 0.0025 | 0.436 | 0.394 |
| 0.0025-0.005 | 0.380 | 0.276 |
| 0.005-0.01 | 0.298 | 0.178 |
| 0.01-0.02 | 0.191 | 0.091 |
| 0.02-0.04 | 0.087 | 0.025 |
| 0.04-0.08 | 0.024 | 0.003 |
| 0.08-0.16 | 0.007 | 0.000 |

Reading: the model behaves like an exact 2nd-argmax whose two comparisons (vs 3rd, vs 1st) have a
resolution of about 2% of the max. The offsets are unbiased on average:
- adding a scale-relative margin d to either comparison (x > S + d|M|, d in ±0.01, ±0.03) only lowers
  agreement (best d = 0: 0.9558 on dev seed 8);
- leaky registers (stored M or S shrinking per step, e.g. n2's self-weight 0.969) lower it (0.92 at 0.969);
- error rate in near-ties barely depends on the size of the other inputs (rest-rms / M quintiles:
  0.286 ... 0.326), so the offset is not plain crosstalk from ranks 4-10.

## Why no compact rule closes the gap (evidence, not proof)
All dev seeds, test on a separate seed:
- decision tree on rank/gap features (p1, p2, p3, gaps, M, S, T, #positive), up to 256 leaves: 0.9555;
- 720-cell lookup of median exact local Jacobians (per (p1,p2,p3) configuration): 0.909;
- per-sample EXACT local Jacobian restricted to the top-5 input values: 0.975 (top-10: 1.0 by construction);
- ReLU masks predicted from phantom top-2/3/4 positions, then exact masked linear algebra: 0.935-0.941.
Together with 0.1% weight noise flipping 1.2-2% of outputs (Bruno, Rasmus, Kasper), this is strong
evidence that the model's near-tie decisions are set by fine-grained, configuration-specific weight
values, i.e. there is no short rule that reproduces them.
