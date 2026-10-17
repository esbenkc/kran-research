# T5 — dense neurons 3, 10, 11, 12, 13, 14, 15 (owner Felix)

Rebuild: `cd swarm-algzoo-432 && .venv/bin/python denseB1.py` (7 s). Seeds 41 (discovery) and 42 (eval),
n = 1e5 each. Holdout 432432 not used. Swap test = agreement of the edited model with the real model.

## 1. Per-neuron swap tests (seed 42)

| neuron | P(on) at t=10 | linear swap (ReLU -> identity) | zero swap | best single rank-event gate (acc) -> gate swap |
|---|---|---|---|---|
| n3  | 0.87 | 0.9787 | 0.5521 | NOT rank(x_{t-3})==2 (0.772) -> 0.9499 |
| n10 | 0.87 | 0.9805 | 0.1306 | NOT rank(x_{t-4})==2 (0.844) -> 0.6794 |
| n11 | 0.90 | 0.9881 | 0.1117 | NOT rank(x_{t-4})==2 (0.846) -> 0.6582 |
| n12 | 0.98 | 0.9957 | 0.1304 | NOT max-at-lag-4 (0.885) -> 0.9579 |
| n13 | 0.99 | 0.9988 | 0.3017 | NOT rank(x_{t-4})==2 (0.840) -> 0.8304 |
| n14 | 0.99 | 0.9884 | 0.1981 | NOT rank(x_{t-1})==2 (0.821) -> 0.6954 |
| n15 | 0.90 | 0.9076 | 0.6413 | NOT rank(x_{t-4})==0 (0.740) -> 0.8779 |

Joint linear swap n12+n13: 0.9949. Joint linear swap n10..n14: 0.9210.

Reading:
- All seven neurons are load-bearing (zero swap 0.11-0.64).
- None is a rank-event gate. The best rank predicate from the family (rank of x_{t-k} in its prefix
  for k = 1..4, r = 0..2; lag of current rank-0/1/2 element) is a worse on/off predictor than "always on",
  and every gate swap is worse than the linear swap. Contrast: n9 and n0 (Iris, denseB2.md) ARE pure
  rank gates (swap 0.995 / 0.997).
- n12 and n13 are linear carriers (swap delta < 0.5 pt, jointly 0.9949). n10, n11, n14 are linear
  carriers that clip rarely (individually 98.1-98.8%); jointly the clips matter (0.921). n3 and n15 clip
  often and value-dependently (P(on) dips to 0.62 / 0.48 at mid-sequence); their clipping is part of the
  computation (linear swap 0.979 / 0.908), but no rank event predicts it.
- Conclusion: these neurons are not interpretable one by one. They are coordinates of a shared linear
  oscillator bank (section 2). Their ReLUs act as value-dependent clips, not as event switches.

## 2. The oscillator bank (phase code for positions)

The most common ReLU pattern late in the sequence (h_6..h_10, 16.3% of neuron-states) has n4, n5, n9 off.
Inside one pattern the RNN is linear: h_{t+1} = (D W_hh) h_t + D W_hi x_t. Eigenvalues of D W_hh:

| mode | abs(lambda) | angle rad/step | period (steps) | name |
|---|---|---|---|---|
| k0 | 1.010 | 0 | inf | value holder (running max, n2-heavy) |
| k1 | 0.973 | 0 | inf | value holder |
| k2,k3 | 0.951 | ±0.746 | 8.4 | **max clock** |
| k4,k5 | 0.937 | ±1.454 | 4.3 | 2nd clock |
| k6 | 0.928 | π | 2.0 | 2nd clock (alternating) |
| k7,k8 | 0.837 | ±2.178 | 2.9 | 2nd clock |

Test of the phase-code claim at t = 10 (S > 0 only; z_k = left-eigvec projection of h_10). Demodulated
coherence c = |mean(z_k · e^{-i·angle_k·(9-p)})| / mean|z_k|; c = 1 means phase = (steps since event at
position p) × rotation per step, exactly; random phases give c ≈ 0.

| mode | max first, ref p1 | max first, ref p2 | 2nd first, ref p1 | 2nd first, ref p2 | within-(p1,p2)-cell coherence |
|---|---|---|---|---|---|
| k2 max clock | **0.98** | 0.16 | **0.96** | 0.37 | 0.99 / 0.98 |
| k4 | 0.24 | **0.66** | 0.35 | 0.30 | 0.82 / 0.82 |
| k6 | 0.26 | **0.67** | 0.39 | 0.38 | 0.75 / 0.78 |
| k7 | 0.18 | **0.68** | 0.38 | 0.17 | 0.85 / 0.85 |

Reading:
- k2 rotates by 0.746 rad per step since the running max arrived, in both orders (c = 0.96-0.98): the
  network stores "how long ago was the max" as the phase of a slowly decaying 8.4-step oscillator.
- k4, k6, k7 rotate since the 2nd max arrived when the 2nd came after the max (c = 0.66-0.68), i.e. when
  the 2nd max is the most recent top-2 event. When the 2nd came first (it was then a running max that got
  overtaken), the 2nd-clock phases are still fixed per (p1, p2) cell (0.78-0.85) but are not a simple
  rotation since p2 (c = 0.17-0.38): the new-max event writes a (gap, direction)-dependent phase into
  them. The simple copy hypothesis "2nd clock = max clock at the event, then rotate" fails
  (c = 0.09-0.37 for all tested frequency pairs, /tmp exploratory run, seed 41).
- Neuron loadings (rms-scaled share of right-eigenvector mass): n13, n3, n15 sit mostly in the 2nd-clock
  modes (k4-k8: 0.58, 0.69, 0.52); n10, n11, n12, n14 sit in value holders + max clock (0.39-0.53) with
  some 2nd clock; n2 is mostly the value holder (0.45). Subcircuit A neurons n1, n6, n7, n8 load on value
  + max clock.

## 3. Limits (honest)

- The mode analysis uses ONE activation pattern (16% share). It describes the quiet regime between
  events. A readout-level mode ablation in this basis is not clean: dropping the fast modes
  (abs(lambda) < 0.75) from h_10 gives agreement 0.32, because h_10 often sits right after an event.
- The event-time write rule (how a new max / new 2nd sets the clock phases) is not derived here; it is
  the (gap, direction) dependent ReLU pattern that Keane describes. Bruno's C2 role tests should cover it.
- No neuron here has a closed-form formula of x. Under PREREG item 4(b) these neurons are not
  "explained" one by one; the explanation offered is at the level of modes, with the coherence numbers
  above as the falsifiable test.
