# T4 — subcircuit A = neurons {1,2,4,6,7,8} (owner Bruno)

All numbers: N(0,1) inputs, dev seeds 1-9 (never 432432). Scripts in this folder, `.venv/bin/python <script>`.
Units ("x-units", exact equivalent model, `subA.load_xunits()`): neurons 1,4,6,7,8 scaled to |W_hi|=1;
neuron 2 (W_hi ~ 0) scaled so W_hh[2,4]=+1; B neurons keep |W_hi|=1 scaling (so their activations are large).
h_t = state after reading x_0..x_{t-1}.

## 1. Idealized closed forms (ARC + Zihao)
| neuron | ideal formula of h_t | R^2 at t=10 | swap test agreement |
|---|---|---|---|
| 2 | M_{t-1} = max(0,x_0..x_{t-2}) | 0.90 | 0.871 |
| 4 | M_t - M_{t-1} = ReLU(x_{t-1}-M_{t-1}) | 0.98 | 0.963 |
| 7 | M_t - x_{t-1} | 0.97 | 0.526 |
| 6 | max(0, x_0..x_{t-3}, x_{t-1}) - x_{t-1} | 0.93 | 0.475 |
| 1 | max(0, x_0..x_{t-4}, x_{t-2}, x_{t-1}) - x_{t-1} | 0.90 | 0.391 |
| all five | | | 0.311 |

Swap test = inside the real recurrence, overwrite the neuron with its formula at every step, B runs normally
(`swapA.py`, 2e5, seed 2). Per-t R^2 in `subA_results.json` (`subA.py`). Conclusion: the ideal forms are the
right *qualitative* reading but fail as a functional replacement; R^2 0.9-1.0 says nothing about argmax.

## 2. Exact structure: rewrite A in the "deficit" basis
Coordinates c = (n2, n4, n7, d1, d6, d8) with d_k = n_k - n7. Ideally d_k = LOOmax_k - M: zero unless the
left-out position held the running max, then minus the gap to the next value. Exact rows (pre-activations,
W_hh[A,A] @ T^-1, last column W_hi):

```
          n2     n4     n7     d1     d6     d8    W_hi
pre n2   0.969  1.000 -0.000 -0.061 -0.064  0.108  0.001
pre n4  -0.852 -0.986  0.001  0.037  0.101 -0.099  1
pre n7   0.899  1.016  0.001 -0.081 -0.152  0.156 -1
pre n1   0.838  0.838 -0.001  0.409  1.036 -0.140 -1
pre n6   0.855 -0.044 -0.012 -0.130  0.361 -0.092 -1
pre n8   0.893 -0.034 -0.011  0.544 -0.000  0.615 -1
```
Readings (structural facts, exact to 0.012):
- **No neuron reads n7 directly** (column ~0). n7 = "max minus last input" is consumed only through the
  deficits d_k. In the typical case n1=n6=n7=n8 and all corrections cancel.
- **n2 = leaky running max**: always on, n2 <- 0.969 n2 + n4 (+ deficit corrections). Leak 3.1%/step.
- **n4 = new-max increment**: ReLU(x - 0.852 n2 - 0.986 n4). It compares x with a *scaled* max (0.85, not 1).
- **n7** = ReLU(0.899 n2 + 1.016 n4 - x): "max minus x" with gain 0.9 on the older max.
- **n6** reads n2 but not n4 -> leaves out the previous input (LOO lag 1), keeps 0.36 of its own deficit.
- **n1** reads n2+n4 (0.838 each) + d6 (1.036) + 0.41 d1 -> deficit shifts one step older (LOO lag 2).
- **n8** reads n2 + 0.544 d1 + 0.615 d8 -> deficit of lags >= 3, decaying ~0.6/step.
So A is a **shift register of "where was the max" for the last ~4 positions**, plus the gap size M - S
carried in the deficit values, plus a leaky running max. Older max positions must come from B.

## 3. A is NOT closed at the precision that matters (correction to PLAN.md)
W_hh[A,B] entries are <= 0.015 in x-units, but B activations are large (rms 4-283), so the input from B to
A rows (2,4,7,1,6,8) has rms 0.05, 0.11, 0.07, 0.14, 0.12, 0.14 vs 0.79-1.18 from A itself (5-14%).
Zeroing W_hh[A,B] -> agreement 0.514 (2e5, seed 9). Setting W_hi[2]=0 -> 0.998 (harmless).
Zeroing the n7 column in deficit basis -> 0.942.

## 4. Why no rounded/idealized version works (`sensitivity.py`, `sensitivity.json`)
Multiplicative noise on all weights: 0.03% -> 0.996, 0.1% -> 0.988, 0.3% -> 0.966, 1% -> 0.83 agreement.
Readout cancels ~20x: mean sum_j |W_oh_ij h_j| = 354 vs median top-2 logit margin 8.2.
Linearizing single neurons (drop ReLU, `linB.py`): only n2 (0.9993) and n13 (0.9986) are safe.
