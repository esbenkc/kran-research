"""T6 (Iris): gate conditions + roles of dense neurons 0, 3, 5, 9 in M_{16,10}.

Run: .venv/bin/python denseB2.py   (~20 s). Dev seeds only (1 = fit, 2 = eval); never 432432.

Indexing: h_{t+1} = ReLU(W_hh h_t + W_hi x_t) is the state after reading x_0..x_t.
"Phantom zero": every max(0, ...) feature in this model treats 0 as an extra element, so ranks
below are taken over {x_0..x_k} U {0}.

Swap test: run the model, but for neuron n replace ReLU(pre_n) by pre_n * 1[gate_n(x, t)], where
gate_n is a predicate on the raw input x only. If agreement with the model stays high, the
predicate fully explains WHEN the neuron is on. (Its value when on is the linear pre-activation.)
"""
import numpy as np

from core import load_raw

Wi, Wh, Wo = load_raw()
N_EVAL = 100_000


def forward(x, swap=(), gate_fn=None):
    B = x.shape[0]
    h = np.zeros((B, 16))
    states = [h]
    for t in range(10):
        pre = h @ Wh.T + x[:, t : t + 1] * Wi[None, :]
        hn = np.maximum(pre, 0.0)
        if t >= 1:
            for n in swap:
                hn[:, n] = np.where(gate_fn(n, x, t), pre[:, n], 0.0)
        h = hn
        states.append(h)
    return h @ Wo.T, np.stack(states, 1)


def rank_in_prefix(x, j, t_end):
    """Rank of x_j among {x_0..x_t_end} U {0} (0 = largest). Phantom 0 counts only if x_j < 0."""
    return (x[:, : t_end + 1] > x[:, j : j + 1]).sum(1) + (x[:, j] < 0)


def top3_with0(a):
    b = np.concatenate([a, np.zeros((len(a), 1))], 1)
    s = -np.sort(-b, 1)
    pad = np.zeros(len(a))
    return s[:, 0], s[:, 1] if s.shape[1] > 1 else pad, s[:, 2] if s.shape[1] > 2 else pad


def n5_features(x, t):
    # comparator inputs for neuron 5 at h_{t+1}: x_{t-1} vs top-3 of {x_0..x_{t-2}} U {0}, and x_t
    if t >= 2:
        Mp, Sp, Tp = top3_with0(x[:, : t - 1])
    else:
        Mp = Sp = Tp = np.zeros(len(x))
    return np.column_stack([x[:, t - 1], Mp, Sp, Tp, x[:, t]])


def fit_n5(x_fit):
    """Least-squares linear comparator for pre_5 (discovery only; constants fitted on seed 1)."""
    _, H = forward(x_fit)
    coef = {}
    for t in range(1, 10):
        pre5 = H[:, t] @ Wh[5] + x_fit[:, t] * Wi[5]
        coef[t], *_ = np.linalg.lstsq(n5_features(x_fit, t), pre5, rcond=None)
    return coef


def make_gate(coef5):
    def gate(n, x, t):
        if n == 9:  # x_{t-1} is the max of {x_0..x_{t-1}} U {0}
            return rank_in_prefix(x, t - 1, t - 1) == 0
        if n == 0:  # NOT: the max of {0 at position -1, x_0..x_{t-1}} sits at position t-2
            if t < 2:  # t-2 = -1 is the phantom zero's position: off iff x_0 < 0
                return x[:, 0] > 0
            return rank_in_prefix(x, t - 2, t - 1) != 0
        if n == 5:  # x_{t-1} beats a blend of 2nd/3rd of the earlier prefix
            return n5_features(x, t) @ coef5[t] > 0
        if n == 55:  # exact-rank version of n5: x_{t-1} in top-2 of {x_0..x_{t-1}} U {0}
            return rank_in_prefix(x, t - 1, t - 1) <= 1
        raise ValueError(n)

    return gate


def main():
    x_fit = np.random.default_rng(1).standard_normal((200_000, 10))
    x = np.random.default_rng(2).standard_normal((N_EVAL, 10))
    coef5 = fit_n5(x_fit)
    gate = make_gate(coef5)
    L, H = forward(x)
    pm = L.argmax(1)
    print(f"model acc (seed 2): {(pm == np.argsort(-x, 1)[:, 1]).mean():.4f}")

    print("\nGate accuracy P(predicate == (h_n>0)) per t = 1..9:")
    for n, name in [(9, "n9"), (0, "n0"), (5, "n5 (fitted comparator)"), (55, "n5 (exact top-2 rank)")]:
        nn = 5 if n == 55 else n
        acc = [((H[:, t + 1, nn] > 0) == gate(n, x, t)).mean() for t in range(1, 10)]
        print(f"  {name:24s}", " ".join(f"{a:.3f}" for a in acc))

    print("\nn5 comparator coefficients [x_{t-1}, M', S', T', x_t] (fit seed 1):")
    for t in range(1, 10):
        print(f"  t={t}", np.round(coef5[t], 2))

    print("\nSwap tests (agreement with model argmax, seed 2, n=1e5):")
    for swap in [(9,), (0,), (5,), (0, 9), (0, 5, 9)]:
        Ls, _ = forward(x, swap, gate)
        print(f"  gate-swap {swap}: {(Ls.argmax(1) == pm).mean():.4f}")

    def gate_exact5(n, xx, t):
        return gate(55 if n == 5 else n, xx, t)

    Ls, _ = forward(x, (5,), gate_exact5)
    print(f"  gate-swap (5,) exact top-2 rank: {(Ls.argmax(1) == pm).mean():.4f}")

    print("\nZero-ablation (h_n := 0 every step):")
    for n in (0, 3, 5, 9):
        h = np.zeros((N_EVAL, 16))
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t : t + 1] * Wi[None, :], 0.0)
            h[:, n] = 0
        print(f"  n{n}: {((h @ Wo.T).argmax(1) == pm).mean():.4f}")

    print("\nFinal-state activity P(h_10[n] > 0) by (dM, dS) = steps since max / 2nd (rows dM 0..9, cols dS 0..9):")
    o = np.argsort(-x, 1)
    dM, dS = 9 - o[:, 0], 9 - o[:, 1]
    for n in (0, 5, 9):
        tab = np.full((10, 10), np.nan)
        for a in range(10):
            for b in range(10):
                s = (dM == a) & (dS == b)
                if s.sum() > 30:
                    tab[a, b] = (H[s, 10, n] > 0).mean()
        print(f"  n{n}:")
        print(np.array2string(np.round(tab, 2), prefix="    "))


if __name__ == "__main__":
    np.set_printoptions(linewidth=160, suppress=True)
    main()
