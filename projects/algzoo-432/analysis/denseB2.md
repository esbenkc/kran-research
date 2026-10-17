# T6 — dense neurons 0, 3, 5, 9: gate conditions and roles (Iris)

Code: `denseB2.py` (run `.venv/bin/python denseB2.py`, ~2 s). Fit seed 1 (2e5), eval seed 2 (1e5).
Held-out seed 432432 not used. Indexing: h_{t+1} = state after reading x_0..x_t.
"Phantom zero": 0 acts as an extra element (all max(0, …) features); for neuron 0 it sits at position −1.

## Method

Swap test: in the forward pass replace ReLU(pre_n) by pre_n · 1[gate_n(x, t)], where gate_n is a
predicate on the raw input only. High agreement ⇒ the predicate explains when neuron n is on.
The value while on is still the neuron's linear pre-activation (explained once its sources are).

## Results (agreement with model argmax, seed 2, n = 1e5)

| neuron | gate predicate (on at h_{t+1}) | gate acc t=2..9 | swap agreement | zero-ablation |
|---|---|---|---|---|
| **9** | x_{t-1} is the max of {0, x_0..x_{t-1}} — "the last input was a new running max" | 0.987–0.994 | **0.9953** | 0.672 |
| **0** | NOT [x_{t-2} is the max of {0, x_0..x_{t-1}}] — "the max did not arrive exactly 2 steps ago" (t=1: on iff x_0 > 0) | 0.987–0.998 | **0.9968** | 0.188 |
| **5** | x_{t-1} > blend of the earlier prefix's 2nd/3rd max: 15–17·x_{t-1} − (1–2)·M' − (6–13)·S' ± (2–4)·T' − 0.28·x_t > 0 (coefs per t fitted, seed 1) | 0.935–0.973 | **0.9917** | 0.214 |
| 5 (strict) | x_{t-1} in top-2 of {0, x_0..x_{t-1}} | 0.867–0.959 | 0.8898 | — |
| 0+9 | both swaps | | 0.9925 | |
| 0+5+9 | all three swaps | | 0.9855 | |
| 3 | no clean predicate found (see below) | — | — | 0.552 |

Final-state tables (P(h_10[n] > 0) by steps-since-max dM × steps-since-2nd dS) printed by the script
show the same structure: n9 on only in row dM=1 (and dM=0, dS=1); n0 off only in row dM=2 (and
dM=0, dS=2); n5 on in row dM=1 and column dS=1 (else ~0.05–0.2).

## Where the gates come from in the weights (mean contribution W_hh[n,j]·h_j, t = 6)

- **n9**: OFF side is −W_hh[9,7]·n7 (−97 mean when off) plus n8 (−14), n1 (−5), n6 (−1.4).
  n7 = max(0, x_0..x_{t-1}) − x_{t-1} (ARC), so W_hh[9,7] = −5.03 (raw) is a hard "x_{t-1} < running max"
  veto. ON drive: n4 (+1.0, the max-increment), n14, n10, n11 (+0.3 to +0.6). W_hi[9] = −0.23 adds −0.23·x_t.
- **n5**: OFF side is n7 (−13) and n8 (−13): x_{t-1} below the running max AND below the 2nd max.
  ON drive: n4 (+4.2), n11 (+2.8), n10, n14, n0, n12 (+1.5 to +1.8). The n8 term makes it a comparator
  against the 2nd max, but n8's own approximation error lets the 3rd max leak in (T' coefficient
  −4 at t = 9) — a candidate mechanism for the model's "picks the 3rd" errors (63% of its errors).
- **n0**: OFF side is n5 at the previous step (−12.1 when off vs −1.3 when on) plus n9 (−2.9) and n7.
  So n0 is a one-step-delayed readout of the n9/n5 event: event "x_{t-2} was top" → n0 off at t+1
  unless x_{t-1} overtook it. Chain: input → n7/n8 (comparators) → n9/n5 (event at lag 1) → n0 (lag 2).

## Roles (readout)

- n5 at the final step: on iff x_8 was top-2 of {0, x_0..x_8}; W_oh[8,5] = +23.4 (rms-scaled), the only
  large entry → **direct vote for position 8**.
- n0: big readout, W_oh[:,0] rms-scaled = [20.9, −28.7, −17.6, 8.2, 7.3, 20.9, 13.3, −31.0, −3.2, 7.3];
  off when the max sat at position 7 → removes the −31 suppression of logit 7 in that case.
- n9: small readout (|W_oh| ≤ 2.1 raw), mainly a recurrent source (→ n0 −1.42, n12 +0.81, n11 +0.68,
  n1 −0.69, n6 −0.64, n14 −0.65 raw).
- These are **event/timer neurons**: they mark "a new top value arrived 1 (n9, n5) or 2 (n0) steps ago",
  which is how position information enters the dense block.

## Neuron 3 (open)

- On 61–89% of steps; drivers differ between on/off at every t: n1 (+3 to +6.5), n0 (+1 to +4.4), n8
  (−2.5 to −6.3), n7, n5, n15, n12 — no single veto neuron, unlike 0/5/9.
- Felix (#1505): linearizing n3 alone keeps 97.9% agreement → it is "mostly linear" with a soft
  clip. Value fit at final step: R² 0.72 for a·M + b·S⁺ per (pos max, pos 2nd) bucket — the worst of
  the four. Not explained yet.

## Limits

- n5's comparator constants are least-squares fits on seed 1 (discovery only, not weight-derived).
- Gates are swap-tested one family at a time; errors compound (0+5+9 → 0.9855).
- Values when on are not yet closed-form; they are linear in the source neurons.
