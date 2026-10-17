"""T5 (Felix): roles of dense neurons 3, 10, 11, 12, 13, 14, 15 in M_{16,10}.

Run: .venv/bin/python denseB1.py   (~40 s). Dev seeds only (41 = discovery, 42 = eval); never 432432.

Indexing as in denseB2.py: h_{t+1} = ReLU(W_hh h_t + W_hi x_t) is the state after reading x_0..x_t.
Ranks use the phantom zero: rank of x_j among {x_0..x_end} U {0}, 0 = largest.

Tests per neuron (each is a swap test: agreement of the edited model with the real model's argmax):
  linear swap : ReLU replaced by identity for that neuron only (role = linear carrier if ~1).
  gate swap   : h_n := pre_n * 1[predicate on raw x]; the predicate is the best single rank event
                from a fixed family (searched on seed 41, scored on seed 42).
  zero swap   : h_n := 0 at every step (how much the neuron matters at all).
"""
import itertools

import numpy as np

from core import load_raw

Wi, Wh, Wo = load_raw()
NEURONS = (3, 10, 11, 12, 13, 14, 15)


def forward(x, edits=None):
    """edits: dict n -> ('lin',) | ('zero',) | ('gate', fn) with fn(x, t) -> bool array."""
    edits = edits or {}
    h = np.zeros((len(x), 16))
    H = [h]
    for t in range(10):
        pre = h @ Wh.T + x[:, t : t + 1] * Wi[None, :]
        hn = np.maximum(pre, 0.0)
        for n, e in edits.items():
            if e[0] == "lin":
                hn[:, n] = pre[:, n]
            elif e[0] == "zero":
                hn[:, n] = 0.0
            elif e[0] == "gate" and t >= 1:
                hn[:, n] = np.where(e[1](x, t), pre[:, n], 0.0)
        h = hn
        H.append(h)
    return h @ Wo.T, np.stack(H, 1)


def rank_in_prefix(x, j, end):
    if j < 0:  # the phantom zero itself
        return (x[:, : end + 1] > 0).sum(1)
    return (x[:, : end + 1] > x[:, j : j + 1]).sum(1) + (x[:, j] < 0)


def lag_of_rank(x, t, r):
    """Steps back from x_{t} to the element of rank r in {x_0..x_t} U {0}; phantom zero = position -1."""
    b = np.concatenate([np.zeros((len(x), 1)), x[:, : t + 1]], 1)
    if r >= b.shape[1]:
        return np.full(len(x), -99)
    pos = np.argsort(-b, 1)[:, r] - 1
    return t - pos


def predicate_family(t_max_lag=4):
    """Single rank events, as (name, fn(x, t) -> bool 'event happened'). Gate = NOT event or event."""
    fam = []
    for lag in range(1, t_max_lag + 1):  # x_{t-lag} relative to the prefix ending at t-lag
        for r in range(3):
            fam.append((f"rank(x_t-{lag})=={r}", lambda x, t, lag=lag, r=r: (
                rank_in_prefix(x, t - lag, t - lag) == r) if t - lag >= 0 else np.zeros(len(x), bool)))
    for r in range(3):  # where the current rank-r element (prefix to t-1) sits
        for lag in range(1, t_max_lag + 1):
            fam.append((f"rank{r} of prefix<t at lag {lag}", lambda x, t, r=r, lag=lag: (
                lag_of_rank(x, t - 1, r) == lag - 1)))
    return fam


def search_gate(n, x, H):
    """Best gate (event or NOT event) by mean accuracy over t = 1..9 on discovery data."""
    on = H[:, 2:, n] > 0  # h_{t+1} for t=1..9
    best = []
    for name, fn in predicate_family():
        ev = np.stack([fn(x, t) for t in range(1, 10)], 1)
        for neg in (False, True):
            g = ~ev if neg else ev
            best.append(((g == on).mean(), ("NOT " if neg else "") + name, fn, neg))
    best.sort(key=lambda b: -b[0])
    return best[:3]


def main():
    x_d = np.random.default_rng(41).standard_normal((100_000, 10))
    x = np.random.default_rng(42).standard_normal((100_000, 10))
    _, H_d = forward(x_d)
    L, H = forward(x)
    pm = L.argmax(1)
    agree = lambda e: float((forward(x, e)[0].argmax(1) == pm).mean())

    print("P(on) per t=1..10:")
    for n in NEURONS:
        print(f"  n{n:2d}", " ".join(f"{v:.2f}" for v in (H[:, 1:, n] > 0).mean(0)))

    print("\nneuron | linear swap | zero swap | best gate (discovery acc) -> gate swap (eval)")
    gates = {}
    for n in NEURONS:
        lin, zero = agree({n: ("lin",)}), agree({n: ("zero",)})
        top = search_gate(n, x_d, H_d)
        acc, name, fn, neg = top[0]
        g = (lambda x_, t, fn=fn, neg=neg: ~fn(x_, t) if neg else fn(x_, t))
        gates[n] = g
        gs = agree({n: ("gate", g)})
        alt = "; ".join(f"{b[1]} {b[0]:.3f}" for b in top[1:])
        print(f"  n{n:2d} | {lin:.4f} | {zero:.4f} | {name} ({acc:.3f}) -> {gs:.4f}   [next: {alt}]")

    lin_set = {n: ("lin",) for n in (12, 13)}
    print(f"\njoint linear swap n12+n13: {agree(lin_set):.4f}")
    print(f"joint linear swap n10..n14: {agree({n: ('lin',) for n in (10, 11, 12, 13, 14)}):.4f}")
    modes(x, H, L)


def modes(x, H, L):
    """Oscillator modes of the dominant late ReLU pattern and what they encode at t = 10."""
    D = H[:, 6:] > 0  # patterns of h_6..h_10
    pats, cnt = np.unique(D.reshape(-1, 16), axis=0, return_counts=True)
    m = pats[cnt.argmax()]
    print(f"\nDominant late pattern (h_6..h_10): off = {np.where(~m)[0].tolist()}, "
          f"share {cnt.max() / cnt.sum():.3f}")
    ev, R = np.linalg.eig(Wh * m[:, None])
    o = np.argsort(-abs(ev))
    ev, R = ev[o], R[:, o]
    Lft = np.linalg.inv(R)
    print("eigenvalues |lambda|, angle (rad/step), period (steps):")
    for k in range(9):
        a = np.angle(ev[k])
        per = 2 * np.pi / abs(a) if abs(a) > 1e-9 else np.inf
        print(f"  k={k}: {abs(ev[k]):.3f}  {a:+.3f}  {per:.1f}")
    # neuron loadings on mode groups (|R| rows, rms-scaled so neurons are comparable)
    s = np.sqrt((H[:, 1:] ** 2).mean((0, 1)))
    groups = {"val(1.01,0.97)": [0, 1], "maxclock": [2, 3], "clk1.45": [4, 5], "alt": [6], "clk2.18": [7, 8]}
    Rn = np.abs(R) / s[:, None]
    Rn = Rn / Rn.sum(1, keepdims=True)
    print("neuron share of |right-eigvec| mass per mode group (rms-scaled):")
    print("        " + " ".join(f"{g:>14s}" for g in groups))
    for n in (1, 2, 6, 7, 8) + NEURONS + (0, 5):
        print(f"  n{n:2d}  " + " ".join(f"{Rn[n, ks].sum():14.2f}" for ks in groups.values()))
    # demodulated phase coherence at t=10: is mode k's phase = (steps since event) * angle(lambda_k)?
    Z = H[:, 10] @ Lft.T
    od = np.argsort(-x, 1)
    p1, p2 = od[:, 0], od[:, 1]
    S = x[np.arange(len(x)), p2]
    good = S > 0
    print("demodulated coherence |mean(z_k * e^{-i a_k (9-p)})| / mean|z_k| (S>0 only):")
    for k in (2, 4, 6, 7):
        u = np.exp(-1j * np.angle(ev[k]))
        row = []
        for cond, cn in ((p1 < p2, "max first"), (p1 > p2, "2nd first")):
            for ref, rn in ((p1, "p1"), (p2, "p2")):
                zz = Z[good & cond, k] * u ** (9 - ref[good & cond])
                row.append(f"{cn}/{rn} {abs(zz.mean()) / abs(zz).mean():.2f}")
        cells = []
        for cond in (p1 < p2, p1 > p2):
            cs, ws = [], []
            for a, b in itertools.product(range(10), range(10)):
                sel = good & cond & (p1 == a) & (p2 == b)
                if sel.sum() >= 100:
                    zz = Z[sel, k] / S[sel]
                    cs.append(abs(zz.mean()) / abs(zz).mean())
                    ws.append(sel.sum())
            cells.append(np.average(cs, weights=ws))
        print(f"  k={k} ({abs(ev[k]):.2f}, {np.angle(ev[k]):+.2f}): " + "  ".join(row)
              + f"  | within-(p1,p2)-cell: max first {cells[0]:.2f}, 2nd first {cells[1]:.2f}")


if __name__ == "__main__":
    np.set_printoptions(linewidth=160, suppress=True)
    main()
