# C2 / T10 — annotated exact model + falsifiable role tests (owner Bruno)

Rebuild: `.venv/bin/python explained_model.py > explained_model.log` (12 s) -> `explained_model.json`.
Dev seeds only (7 for role tests, 11 for probes, 1 for B rms scaling). Held-out 432432 not used.

**What this is not.** `forward()` in `explained_model.py` is the 16-neuron recurrence in named
coordinates (A in x-units with the deficit basis, B at unit rms). Its 100% agreement is true by
construction and is *not* evidence. The only exact reparametrizations of a ReLU net are permutation and
positive per-neuron scaling; anything that mixes neurons (eigenmodes, SVD) is an approximation, and is
tested as one below. The evidence is the role-test table: replace a named role by its claimed simpler
form, keep everything else exact, and measure argmax agreement with the model.

## Roles (one line per neuron; see subA.md, denseB2.md for derivations)
| n | role | test that supports / limits it |
|---|---|---|
| 2 | leaky running max: n2 <- 0.969 n2 + 1.0 n4 (+ deficit corrections), ignores x | linear: 0.9993; W_hi[2]=0: 0.9976; ideal max formula: 0.871 |
| 4 | new-max increment ReLU(x - 0.852 n2 - 0.986 n4) | ideal formula: 0.963 |
| 7 | running max minus last input, ReLU(0.899 n2 + 1.016 n4 - x); read only through deficits | ideal: 0.526; zero n7 column (deficit basis): 0.943 |
| 6 | deficit register, lag 1 (LOO-max without previous input, minus x); self 0.36 | ideal: 0.472 |
| 1 | deficit register, lag 2 (reads d6 at 1.036) | ideal: 0.390 |
| 8 | deficit register, lag >= 3 (reads d1 0.54, own d8 0.62) | — |
| 9 | event gate: last input was a new running max (Iris T6) | gate swap 0.9953 |
| 0 | timer gate: max did not arrive exactly 2 steps ago (Iris T6) | gate swap 0.9971 |
| 5 | event gate: last input beat ~2nd max; final value votes logit 8 (Iris T6) | fitted gate 0.9917 (denseB2.md) |
| 13, 12 | near-linear dense integrators | drop ReLU: 0.9986, 0.9954 |
| 3, 10, 11, 14 | dense integrators whose rare ReLU-off events matter | drop ReLU: 0.979-0.988 |
| 15 | dense, ReLU important | drop ReLU: 0.908 |

## Role-test table (seed 7, n = 2e5; "err-set" = agreement on the model's own 4.7% errors)
| block | claim | test | agree | err-set |
|---|---|---|---|---|
| A | n2 linear leaky integrator | drop ReLU on n2 | 0.9993 | 0.989 |
| A | n2 ignores input | W_hi[2]=0 | 0.9976 | 0.972 |
| A | ARC ideal formulas n2,4,7,6,1 | overwrite all five each step | 0.3080 | 0.311 |
| A | A<-A gains are ideal integers | round A<-A (deficit basis) to integers | 0.2186 | 0.218 |
| A | no A neuron reads n7 directly | zero n7 column (entries <= 0.012) | 0.9427 | 0.600 |
| A | A is closed | zero W_hh[A,B] | 0.5129 | 0.377 |
| B | B is driven by A | zero W_hh[B,A] | 0.1037 | 0.098 |
| B | n9 + n0 are input-predicate gates | gate swap both | 0.9929 | 0.897 |
| B | B is a linear filter of A | drop ReLU on all B | 0.1412 | 0.163 |
| B | B<-B dynamics = few oscillator modes | keep top-k eigenmodes of B<-B, k=2/4/6/8 | 0.13/0.17/0.16/0.16 | <=0.18 |
| R | readout is low rank | SVD of centered W_oh, rank 2/4/6/8 | 0.36/0.43/0.74/0.92 | <=0.70 |
| R | column-constant of W_oh is irrelevant | subtract column mean | 1.0000 | 1.000 (exact identity) |
| R | readout columns = sinusoid templates over position | const + 1 / 2 fitted cosines per W_oh column (fit to weights) | 0.3051 / 0.3651 | 0.23 / 0.24 |

(Rank 9 of the centered 10x16 readout is the full matrix, so it is 1.0 trivially.)

## Where position information lives (linear probes on h_10, seed 11)
| probe | argmax position p1 | 2nd-argmax position p2 |
|---|---|---|
| A neurons only | 0.90 (pos 9: 0.99, pos 0-3: 0.83-0.86) | 0.19 (pos 9: 0.87) |
| B neurons only | 0.86 (pos 9: 0.25, pos 2-8: 0.92-0.98) | 0.51 |
| all 16 | 0.99 | 0.952 (= model accuracy 0.953) |
A knows recent / last-position events; B knows older positions; p2 is only linearly available from the
joint state, at exactly the model's accuracy.

## Conclusions
1. Supported, compact roles (swap >= 0.99): n2 leaky linear integrator ignoring x; n9/n0 event gates;
   n13/n12 near-linear. These are the parts that admit a hand description at the 99% bar.
2. Qualitatively right, functionally insufficient: ARC/Zihao LOO-max formulas for n7/n6/n1 and the
   "integer" A gains (0.22-0.53). The exact non-integer gains (0.852, 0.899, 0.969, 0.36, 0.54, 0.62)
   are load-bearing.
3. Falsified: "A is closed" (0.51), "B is a linear filter" (0.14), "B<-B is a few oscillator modes"
   (<= 0.17), "readout is a low-rank / sinusoid template" (rank 8: 0.92; 1-2 sinusoids per column:
   0.31/0.37). The oscillation Felix sees exists in the effective
   pattern-specific matrix, not in B<-B alone.

## Addendum (Kasper #1545): event-time write + phase-decoder tests — `modes_c2.py` -> `modes_c2.json` (3 s, seeds 41/43)
Modes of the dominant late ReLU pattern (n4, n5, n9 off; 16.4% of late states; denseB1.md s.2).
z = left-eigenvector projection of h_t onto the max clock k2 (|lambda| 0.951, +0.746 rad/step).

**(a) The max event writes a fixed phase.** Samples whose max (positive) sits at position p; phase of z at
h_{p+d}. Circular concentration R = |mean e^{i phase}| (1 = same phase in every sample):
- d = 1 (right after the event): R = 0.999-1.000 for p <= 2, 0.95-0.97 for p = 4-5, 0.88 for p = 7-8.
  Mean written phase is +0.24..+0.54 rad for every p (one write rule, independent of position).
- Phase then advances +0.734 rad/step (mode angle +0.746). At t = 10, R = 0.93-0.98 for p <= 5.
- The loss of R at late p sits in the smallest-gap tercile (gap = x_p - previous max): R 0.68-0.79 there vs
  0.95-1.00 in the top tercile. So the write is fixed when the new max clearly beats the old one, and
  gap-dependent when it barely beats it. |corr(gap, phase)| 0.06-0.60.

**(b) Readout = phase decoder, only for early positions.** In the mode basis W_oh = Re(C L).
Replace C[j,k] of clock modes by a rotating template a_k e^{-i angle_k (9-j)} |lambda_k|^{-(9-j)} (2 real
params per mode, least squares to the weights):
| modes | positions templated | rel. weight fit error | agreement | err-set |
|---|---|---|---|---|
| max clock k2,k3 | all j (no amplitude growth) | 0.77 | 0.9208 | 0.570 |
| max clock k2,k3 | j <= 3 | 0.35 | 0.9777 | 0.818 |
| 2nd clocks k4,k5,k7,k8 | j <= 3 | 0.09 | 0.9767 | 0.863 |
| 2nd clocks | j <= 5 | 0.25-0.32 | 0.9014 | 0.673 |
| all clocks | j <= 3 | 0.09-0.35 | 0.9629 | 0.772 |
Reading: for positions 0-3 the readout of the 2nd clocks IS a rotating template matched to the mode
frequency (9% weight error). The template degrades from position 4 on; late positions are read through
subcircuit A (shift register) instead. No template variant reaches the 99% bar.
