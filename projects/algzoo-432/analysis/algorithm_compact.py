"""C1 (Felix, T8): compact mechanism algorithm for ARC AlgZoo M_{16,10} (example_2nd_argmax).

Contract (check_agreement.py): predict(X float64 [N,10]) -> int [N]. No weights, no torch, no model.

The algorithm is the task-level streaming procedure. Which parts have neural evidence (see
EXPLANATION.md, denseB1.md, subA.md, denseB2.md, explained_model.md) and which are only bookkeeping:

  * Phantom zero (tested). h_0 = 0 and no biases, so every max-feature in subcircuit A is max(0, ...):
    the network behaves as if a 0 sat at virtual position -1 before x_0. The phantom rule below doubles
    agreement on inputs with <= 1 positive value (algorithm_compact.md).
  * Value registers (tested, approximate): n2 leaky running max, n4 increment, n1/n6/n7/n8 deficit
    registers (subA.md) carry M and the comparisons against it.
  * New-max event (tested): gate n9 fires the step after a new running max (swap 0.995).
  * Max clock (tested): the 0.951-modulus mode at 0.746 rad/step gets a fixed phase written at the
    new-max event and then rotates (denseB1.md s.2, explained_model.md addendum).
  * 2nd-max timing (partly tested): when the 2nd arrives after the max, 2nd-clock modes rotate since the
    2nd's arrival. When the max overtakes an earlier 2nd, the network's write rule is NOT known; the
    "old max becomes the 2nd" line below is task bookkeeping, not a claimed neural copy (a simple copy
    of the max clock is falsified, denseB1.md s.2).
  * Readout (partly tested): a rotating phase template fit to W_oh explains it only in part
    (agreement 0.92, explained_model.md addendum).

So the prediction is the position of the 2nd largest element of (0 at position -1, x_0..x_9), with
position -1 mapped to 0. It uses no fitted constants and no numeric literals beyond indices.
"""
import numpy as np


def predict(X):
    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape
    M = np.zeros(n)  # running max, starts at the phantom 0
    S = np.full(n, -np.inf)  # running 2nd max (nothing yet)
    pM = np.full(n, -1)  # position of M; -1 = phantom
    pS = np.full(n, -1)
    for t in range(T):
        x = X[:, t]
        new_max = x > M  # gate n9 fires at the next step
        new_2nd = ~new_max & (x > S)
        # new max: old max (and its clock) becomes the 2nd; max clock resets
        S = np.where(new_max, M, np.where(new_2nd, x, S))
        pS = np.where(new_max, pM, np.where(new_2nd, t, pS))
        M = np.where(new_max, x, M)
        pM = np.where(new_max, t, pM)
    return np.maximum(pS, 0)  # phantom clock (position -1) decodes to position 0
