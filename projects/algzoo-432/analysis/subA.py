"""T4: subcircuit A = neurons {1,2,4,6,7,8} of M_{16,10}.

Units: every A neuron is rescaled to "x units" (exact equivalent model):
  neurons 1,4,6,7,8: |W_hi| = 1 (as in core.load_std)
  neuron 2 (W_hi ~ 0): rescaled so that W_hh[2,4] = +1 (it accumulates neuron 4).
Notation: h_t = state after reading x_0..x_{t-1} (core.run states[:, t]).
  P_t(S) = max(0, max_{i in S} x_i);  M_t = P(x_0..x_{t-1}).

Idealized closed forms (from the rounded A-block weights, see subA.md):
  n2_t = M_{t-1}                       = max(0, x_0..x_{t-2})
  n4_t = M_t - M_{t-1}                 = ReLU(x_{t-1} - M_{t-1})
  n7_t = M_t - x_{t-1}                 = ReLU(M_{t-1} - x_{t-1})
  n6_t = P(x_0..x_{t-3}, x_{t-1}) - x_{t-1}   (leave out x_{t-2})
  n1_t = P(x_0..x_{t-4}, x_{t-2}, x_{t-1}) - x_{t-1}   (leave out x_{t-3})
  n8_t = candidate forms tested below.
Run: .venv/bin/python subA.py  -> subA_results.json
"""
import json

import numpy as np

from core import N, load_std, run, sample

A = [1, 2, 4, 6, 7, 8]


def load_xunits():
    W_hi, W_hh, W_oh = load_std()
    s = np.ones(16)
    s[2] = 1.0 / W_hh[2, 4]
    W_hi = s * W_hi
    W_hh = (s[:, None] * W_hh) / s[None, :]
    W_oh = W_oh / s[None, :]
    return W_hi, W_hh, W_oh


def P(x, idx):
    """max(0, x_i for i in idx); idx may be empty."""
    if len(idx) == 0:
        return np.zeros(x.shape[0])
    return np.maximum(0.0, x[:, idx].max(1))


def loo_rel(x, t, k):
    """max(0, x_0..x_{t-1} except x_{t-k}) - x_{t-1}  (k>=2)."""
    idx = [i for i in range(t) if i != t - k]
    return P(x, idx) - x[:, t - 1]


def ideal(x, t):
    """dict neuron -> idealized value of h_t (t>=1)."""
    M_t = P(x, list(range(t)))
    M_tm1 = P(x, list(range(t - 1)))
    last = x[:, t - 1]
    f = {
        2: M_tm1,
        4: M_t - M_tm1,
        7: M_t - last,
        6: loo_rel(x, t, 2) if t >= 2 else np.maximum(0, -last),
        1: loo_rel(x, t, 3) if t >= 3 else M_t - last,
    }
    return f


def r2(y, f):
    return 1 - ((y - f) ** 2).mean() / max(y.var(), 1e-30)


if __name__ == "__main__":
    W_hi, W_hh, W_oh = load_xunits()
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    print("W_hi[A]", W_hi[A])
    print("W_hh[A, A] (rows=target, cols=source, order 1,2,4,6,7,8)")
    print(W_hh[np.ix_(A, A)])
    B = [i for i in range(16) if i not in A]
    print("max |W_hh[A, B]| per A row", np.abs(W_hh[np.ix_(A, B)]).max(1))
    x = sample(200_000, 1)
    _, H = run(x, W_hi, W_hh, W_oh, return_states=True)
    res = {}
    for n in [2, 4, 7, 6, 1]:
        rows = []
        for t in range(1, N + 1):
            f = ideal(x, t)[n]
            y = H[:, t, n]
            rows.append({"t": t, "r2": r2(y, f), "rmse": float(np.sqrt(((y - f) ** 2).mean())),
                         "std": float(y.std())})
        res[n] = rows
        print(f"n{n}", " ".join(f"t{r['t']}:R2={r['r2']:.3f}/rmse={r['rmse']:.3f}" for r in rows))
    json.dump(res, open("subA_results.json", "w"), indent=1)
