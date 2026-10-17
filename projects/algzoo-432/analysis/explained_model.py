"""C2 / T10: exact, annotated rewrite of M_{16,10} + falsifiable role tests.

This file is NOT an agreement candidate. Its forward pass is the 16-neuron recurrence in named
coordinates, so its 100% agreement is true by construction and proves nothing. The evidence is the
role-test table: each named role is replaced by its claimed simpler form, and we measure argmax
agreement with the real model (dev seed, never the held-out 432432).

Exact reparametrizations used (the only exact ones for a ReLU net: permutation + positive scaling):
  A neurons 1,4,6,7,8 scaled to |W_hi| = 1; neuron 2 scaled so W_hh[2,4] = 1 ("x-units", subA.load_xunits).
  B neurons scaled to unit rms activation on dev seed 1 (presentation only).
Pre-activations of A are additionally written in the deficit basis c = (n2, n4, n7, d1, d6, d8),
d_k = n_k - n7. This rewrites the sum W h, it does not change any ReLU.

Run: .venv/bin/python explained_model.py   -> explained_model.json (+ tables on stdout)
"""
import json
import warnings

import numpy as np

from core import N, agreement, run, sample, second_argmax
from subA import ideal, load_xunits

warnings.filterwarnings("ignore")

A = [2, 4, 7, 1, 6, 8]  # order used everywhere below
B = [9, 0, 5, 3, 10, 11, 12, 13, 14, 15]
ROLE = {
    2: "runmax: leaky running max of the prefix (self 0.969, +1.0 * n4); ignores x",
    4: "newmax: ReLU(x - 0.852 n2 - 0.986 n4) = increment when x beats the (scaled) running max",
    7: "gap_last: ReLU(0.899 n2 + 1.016 n4 - x) = running max minus last input",
    6: "loo_lag1: like n7 but without n4 -> max leaving out the previous input, minus x",
    1: "loo_lag2: reads n2+n4 and d6 -> max leaving out the input 2 steps back, minus x",
    8: "loo_lag3+: reads n2, d1 (0.54), own d8 (0.62) -> max leaving out older inputs, minus x",
    9: "event: last input was a new running max (veto by n7 at -5.03 raw)  [Iris T6]",
    0: "timer: NOT(max arrived exactly 2 steps ago); delayed copy of n5/n9 events  [Iris T6]",
    5: "event: last input beat ~2nd max of prefix; final value = vote for logit 8  [Iris T6]",
    3: "dense, mostly linear (soft clip); role open",
    10: "dense integrator of A/event inputs (part of oscillatory B block)",
    11: "dense integrator of A/event inputs (part of oscillatory B block)",
    12: "dense, near-linear (on >=93%)",
    13: "dense, near-linear (on >=99%), safe to linearize",
    14: "dense integrator, reads n9/n4 events",
    15: "dense integrator",
}

T_DEF = np.eye(6)
T_DEF[3, 2] = T_DEF[4, 2] = T_DEF[5, 2] = -1  # c = T h_A  (rows: n2,n4,n7,d1,d6,d8)


def named_weights(rms_seed=1):
    W_hi, W_hh, W_oh = load_xunits()
    _, H = run(sample(50_000, rms_seed), W_hi, W_hh, W_oh, return_states=True)
    s = np.ones(16)
    rmsB = np.sqrt((H[:, 1:, B] ** 2).mean((0, 1)))
    s[B] = 1.0 / rmsB
    W_hi = s * W_hi
    W_hh = (s[:, None] * W_hh) / s[None, :]
    W_oh = W_oh / s[None, :]
    return W_hi, W_hh, W_oh


def blocks(W):
    W_hi, W_hh, W_oh = W
    Tinv = np.linalg.inv(T_DEF)
    return {
        "A_from_A_deficit_basis": W_hh[np.ix_(A, A)] @ Tinv,
        "A_from_B": W_hh[np.ix_(A, B)],
        "B_from_A_deficit_basis": W_hh[np.ix_(B, A)] @ Tinv,
        "B_from_B": W_hh[np.ix_(B, B)],
        "W_hi_A": W_hi[A],
        "W_hi_B": W_hi[B],
        "W_oh_A_deficit_basis": W_oh[:, A] @ Tinv,
        "W_oh_B": W_oh[:, B],
    }


def forward(x, W, mod=None):
    """Named forward. mod(t, pre, h_prev, x_t) may edit pre-activations / post-activations.
    mod returns h (post-activation) or None to use ReLU(pre)."""
    W_hi, W_hh, W_oh = W
    h = np.zeros((x.shape[0], 16))
    for t in range(N):
        pre = h @ W_hh.T + x[:, t : t + 1] * W_hi
        out = mod(t, pre, h, x) if mod is not None else None
        h = np.maximum(pre, 0.0) if out is None else out
    return h @ W_oh.T


# ---------------------------------------------------------------- role tests
def with_W(W, **edits):
    W_hi, W_hh, W_oh = (w.copy() for w in W)
    for k, f in edits.items():
        f({"W_hi": W_hi, "W_hh": W_hh, "W_oh": W_oh})
    return W_hi, W_hh, W_oh


def linearize(ns):
    ns = list(ns)

    def mod(t, pre, h, x):
        out = np.maximum(pre, 0.0)
        out[:, ns] = pre[:, ns]
        return out

    return mod


def swap_ideal(ns, W):
    """Overwrite A neurons by ARC's ideal formulas (in x-units)."""

    def mod(t, pre, h, x):
        out = np.maximum(pre, 0.0)
        f = ideal(x, t + 1)
        for n in ns:
            out[:, n] = f[n]
        return out

    return mod


def gate_swap(ns):
    """Iris T6 predicates (denseB2.py make_gate), applied at loop step t>=1 (state h_{t+1}):
    n9 on iff x_{t-1} is the max of {0, x_0..x_{t-1}};
    n0 off iff the max of {0 at position -1, x_0..x_{t-1}} sits at position t-2."""

    def rank_in_prefix(x, j, t_end):
        return (x[:, : t_end + 1] > x[:, j : j + 1]).sum(1) + (x[:, j] < 0)

    def mod(t, pre, h, x):
        out = np.maximum(pre, 0.0)
        if t == 0:
            return out
        if 9 in ns:
            out[:, 9] = np.where(rank_in_prefix(x, t - 1, t - 1) == 0, pre[:, 9], 0.0)
        if 0 in ns:
            g = x[:, 0] > 0 if t < 2 else rank_in_prefix(x, t - 2, t - 1) != 0
            out[:, 0] = np.where(g, pre[:, 0], 0.0)
        return out

    return mod


def ideal_A_gains(W):
    """Replace the A<-A block (deficit basis) by its rounded 'ideal' integers, keep everything else."""
    Tinv = np.linalg.inv(T_DEF)
    R = W[1][np.ix_(A, A)] @ Tinv
    R_ideal = np.round(R)  # 0.969->1, 0.852->1, 0.36->0, 0.54->1, 0.615->1, ...
    W_hh = W[1].copy()
    W_hh[np.ix_(A, A)] = R_ideal @ T_DEF
    return (W[0], W_hh, W[2]), R_ideal


def modal_truncation(W, k):
    """Keep only the k largest-|eigenvalue| modes of the B<-B block (real-valued reconstruction)."""
    W_hh = W[1].copy()
    M = W_hh[np.ix_(B, B)]
    ev, V = np.linalg.eig(M)
    o = np.argsort(-np.abs(ev))
    keep = o[:k]
    Vi = np.linalg.inv(V)
    Mk = (V[:, keep] * ev[keep]) @ Vi[keep, :]
    W_hh[np.ix_(B, B)] = Mk.real
    return W[0], W_hh, W[2]


def oh_rank(W, r):
    W_oh = W[2]
    mu = W_oh.mean(0, keepdims=True)  # adding a column-constant to every logit never changes argmax
    U, S, Vt = np.linalg.svd(W_oh - mu, full_matrices=False)
    return W[0], W[1], (U[:, :r] * S[:r]) @ Vt[:r] + mu


def oh_sinusoid(W, k):
    """Replace W_oh columns (positions 0..8; position 9 kept) by a constant + k fitted sinusoids.
    The fit is to the weights, not to model outputs."""
    from scipy.optimize import least_squares

    j = np.arange(9.0)

    def f(q):
        return q[0] + sum(q[1 + 3 * i] * np.cos(q[2 + 3 * i] * j + q[3 + 3 * i]) for i in range(k))

    W_oh = W[2].copy()
    for col in range(16):
        c = W_oh[:9, col]
        best = None
        for w0 in np.linspace(0.2, 3.0, 15):
            p0 = [0.0]
            for i in range(k):
                p0 += [float(np.std(c)), w0 * (i + 1), 0.0]
            r = least_squares(lambda q: f(q) - c, p0)
            if best is None or r.cost < best.cost:
                best = r
        W_oh[:9, col] = f(best.x)
    return W[0], W[1], W_oh


def position_probes(seed=11, n_train=100_000, n_test=20_000):
    """Linear (multinomial logistic) probes on h_10: where are the argmax / 2nd argmax positions stored?"""
    from sklearn.linear_model import LogisticRegression

    x = sample(n_train + n_test, seed)
    _, H = run(x, *load_xunits(), return_states=True)
    h = H[:, -1] / np.sqrt((H[:, -1] ** 2).mean(0) + 1e-12)
    o = np.argsort(-x, 1)
    res = {}
    for name, S in [("A", A), ("B", B), ("all", A + B)]:
        for tgt, y in [("p1", o[:, 0]), ("p2", o[:, 1])]:
            clf = LogisticRegression(max_iter=300, C=10).fit(h[:n_train][:, S], y[:n_train])
            pr = clf.predict(h[n_train:][:, S])
            yt = y[n_train:]
            res[f"{name}->{tgt}"] = {"acc": round(float(np.mean(pr == yt)), 3),
                                     "by_pos": [round(float(np.mean(pr[yt == k] == k)), 2) for k in range(10)]}
            print(f"probe {name:3s} -> {tgt}: acc {res[f'{name}->{tgt}']['acc']}  by position "
                  f"{res[f'{name}->{tgt}']['by_pos']}")
    return res


def main():
    W = named_weights()
    x = sample(200_000, 7)
    ref = run(x, *load_xunits()).argmax(1)
    err = ref != second_argmax(x)
    l_named = forward(x, W)
    assert agreement(l_named.argmax(1), ref) == 1.0  # exact rewrite (by construction, not evidence)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for k, v in blocks(W).items():
        print(f"\n[{k}] rows/cols order A={A} B={B}")
        print(v)

    rows = []

    def rec(group, claim, test, pred):
        a, e = agreement(pred, ref), agreement(pred[err], ref[err])
        rows.append({"group": group, "claim": claim, "test": test, "agree": round(a, 4),
                     "agree_on_model_errors": round(e, 4)})
        print(f"{group:8s} {a:.4f} (err-set {e:.4f})  {claim} | {test}")

    am = lambda W_, mod=None: forward(x, W_, mod).argmax(1)  # noqa: E731
    print("\nROLE TESTS (dev seed 7, n=2e5; agreement with model argmax)")
    rec("A", "n2 is a linear leaky integrator (always on)", "drop ReLU on n2", am(W, linearize([2])))
    rec("A", "n2 ignores the input", "W_hi[2]=0",
        am(with_W(W, a=lambda d: d["W_hi"].__setitem__(2, 0.0))))
    for n in [2, 4, 7, 6, 1]:
        rec("A", f"n{n} = ARC ideal formula", f"overwrite n{n} by formula every step", am(W, swap_ideal([n], W)))
    rec("A", "all five = ARC ideal formulas", "overwrite n2,4,7,6,1", am(W, swap_ideal([2, 4, 7, 6, 1], W)))
    Wi, R_ideal = ideal_A_gains(W)
    rec("A", "A<-A gains are 'ideal' integers (deficit basis)", "round A<-A block to integers", am(Wi))
    T = T_DEF
    Tinv = np.linalg.inv(T)

    def zero_n7col(d):
        R = d["W_hh"][np.ix_(A, A)] @ Tinv
        R[:, 2] = 0
        d["W_hh"][np.ix_(A, A)] = R @ T

    rec("A", "no A neuron reads n7 directly", "zero n7 column (deficit basis, |entries|<=0.012)",
        am(with_W(W, a=zero_n7col)))
    rec("A", "A is a closed subcircuit", "zero W_hh[A,B]",
        am(with_W(W, a=lambda d: d["W_hh"].__setitem__(np.ix_(A, B), 0.0))))
    rec("B", "n9 = 'last input is new running max' gate", "n9 := pre9 * 1[predicate]", am(W, gate_swap([9])))
    rec("B", "n0 = 'max did not arrive 2 steps ago' gate", "n0 := pre0 * 1[predicate]", am(W, gate_swap([0])))
    rec("B", "n9 + n0 gates", "both", am(W, gate_swap([9, 0])))
    for n in [13, 12, 3, 10, 11, 14, 15]:
        rec("B", f"n{n} is (near-)linear", f"drop ReLU on n{n}", am(W, linearize([n])))
    rec("B", "B is a linear filter of A", "drop ReLU on all B", am(W, linearize(B)))
    for k in [2, 4, 6, 8]:
        rec("B", f"B<-B dynamics live in {k} eigenmodes", f"keep top-{k} |eig| modes of B<-B",
            am(modal_truncation(W, k)))
    rec("B", "B is driven by A", "zero W_hh[B,A]",
        am(with_W(W, a=lambda d: d["W_hh"].__setitem__(np.ix_(B, A), 0.0))))
    for r in [2, 4, 6, 8, 9]:
        rec("R", f"readout is rank {r} (+const)", f"SVD-truncate centered W_oh to rank {r}", am(oh_rank(W, r)))
    rec("R", "column-constant of W_oh irrelevant", "subtract column mean of W_oh",
        am(with_W(W, a=lambda d: d["W_oh"].__isub__(d["W_oh"].mean(0, keepdims=True)))))
    for k in [1, 2]:
        rec("R", f"readout columns are {k} sinusoid(s) over position", f"fit const+{k} cos per W_oh column",
            am(oh_sinusoid(W, k)))

    print("\nPOSITION PROBES on h_10 (seed 11, train 1e5, test 2e4)")
    probes = position_probes()
    out = {"seed": 7, "n": len(x), "model_error_rate": float(err.mean()), "role_tests": rows,
           "position_probes": probes,
           "roles": {str(k): v for k, v in ROLE.items()},
           "A_from_A_deficit_basis": blocks(W)["A_from_A_deficit_basis"].tolist(),
           "A_ideal_integer_block": R_ideal.tolist()}
    json.dump(out, open("explained_model.json", "w"), indent=1)


if __name__ == "__main__":
    main()
