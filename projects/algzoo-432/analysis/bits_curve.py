"""T17 (Felix): description bits vs agreement for the network-shaped descriptions. Dev seed 61 (2e5).

Every net is put in the +-1 gauge (W_hi = sign, W_oh row 0 shifted to 0), each free number (not 0, not
+-1) is rounded to k significant digits, and its cost is counted as
  1 sign bit + log2(9 * 10^(k-1)) mantissa bits + empirical entropy of the decimal exponent,
plus the sparsity mask as an enumerative code log2 C(n_slots, n_nonzero) per block. This is an
estimate of an honest code length, not a proven minimum.
"""
import json
from math import lgamma, log

import numpy as np

from core import load_raw, run
from prune_ft import gauge, round_sig

X = np.random.default_rng(61).standard_normal((200_000, 10))
REF = run(X, *load_raw()).argmax(1)


def log2_binom(n, k):
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)


def bits(W, k):
    Wi, Wh, Wo = W
    allw = np.concatenate([Wh.ravel(), Wo[1:].ravel()])
    free = allw[(allw != 0) & (np.abs(allw) != 1.0)]
    e = np.floor(np.log10(np.abs(free))).astype(int)
    _, cnt = np.unique(e, return_counts=True)
    p = cnt / cnt.sum()
    h_exp = float(-(p * np.log2(p)).sum())
    per = 1 + np.log2(9 * 10 ** (k - 1)) + h_exp
    mask = log2_binom(256, int((Wh != 0).sum())) + log2_binom(150, int((Wo[1:] != 0).sum()))
    return len(free), per * len(free) + mask + 16  # 16 sign bits of W_hi


rows = []
nets = {"original 432": gauge(*load_raw())}
for f in ["ladder_pruned_0.99.npz", "ladder_pruned_0.995.npz", "prune_ft_T1_200_140.npz", "prune_ft_T5_200_140.npz",
          "prune_ft_T5_160_130.npz"]:
    d = np.load(f)
    k = ("W_hi", "W_hh", "W_oh") if "W_hi" in d.files else ("Wi", "Wh", "Wo")
    nets[f] = gauge(*(d[key] for key in k))
for name, W in nets.items():
    for k in [2, 3, 4, 5]:
        Wr = (W[0], round_sig(W[1], k), round_sig(W[2], k))
        n, b = bits(Wr, k)
        ag = float((run(X, *Wr).argmax(1) == REF).mean())
        rows.append(dict(net=name, digits=k, free_numbers=n, bits=round(b), agree_dev61=ag))
        print(f"{name:28s} k={k} free={n:3d} bits={b:6.0f} agree={ag:.5f}")
json.dump(rows, open("bits_curve.json", "w"), indent=1)
