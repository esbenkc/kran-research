"""T12 (Felix): minimal-description ladder for M_{16,10}.

Question: what is the smallest number of free numeric literals (and bits) with which a simplified copy of
the network still agrees with the real model at least as well as the model agrees with itself under 0.1%
weight noise (PREREG amendment: 0.98671)? No refitting: every kept number is a rounded original weight
(after exact gauge transforms), so nothing is fitted to model outputs.

Steps, each exact or measured:
  G1 per-neuron scaling so |W_hi_j| = 1 for all 16 neurons (exact; all 16 W_hi become +-1, free).
  G2 softmax shift: W_oh <- W_oh - 1 * W_oh[0] (exact; row 0 becomes 0, 16 literals free).
  Exact gauge therefore leaves 432 - 16 - 16 = 400 free numbers.
  P  greedy pruning of small W_hi, W_hh and W_oh entries, cheapest first (measured, screening seed 70).
  R  round every kept literal to k significant digits (measured).
Run: .venv/bin/python ladder.py   (dev seeds 70 screen, 71 eval; never 432432)
"""
import json

import numpy as np

from core import load_raw, run

BAR = 0.98671
Wi0, Wh0, Wo0 = load_raw()
X_SCREEN = np.random.default_rng(70).standard_normal((30_000, 10))
X_EVAL = np.random.default_rng(71).standard_normal((300_000, 10))
REF_S = run(X_SCREEN, Wi0, Wh0, Wo0).argmax(1)
REF_E = run(X_EVAL, Wi0, Wh0, Wo0).argmax(1)
BIG = np.abs(Wi0) > 1.0  # neurons 1, 4, 6, 7, 8, 14


def agree(W, X=X_EVAL, ref=REF_E):
    return float((run(X, *W).argmax(1) == ref).mean())


def gauge(Wi, Wh, Wo):
    s = 1.0 / np.abs(Wi)
    Wi, Wh, Wo = np.sign(Wi), (s[:, None] * Wh) / s[None, :], Wo / s[None, :]
    return Wi, Wh, Wo - Wo[0:1]


def literals(Wi, Wh, Wo):
    allw = np.concatenate([Wi, Wh.ravel(), Wo.ravel()])
    return int(((allw != 0) & (np.abs(allw) != 1.0)).sum())


def round_sig(W, k):
    out = W.copy()
    nz = out != 0
    e = np.floor(np.log10(np.abs(out[nz])))
    out[nz] = np.round(out[nz] / 10 ** (e - k + 1)) * 10 ** (e - k + 1)
    return out


def rounded(W, k):
    Wi, Wh, Wo = W
    Wi_r = np.where(np.abs(Wi) == 1.0, Wi, round_sig(Wi, k))
    return Wi_r, round_sig(Wh, k), round_sig(Wo, k)


def greedy_prune(W, floor):
    """Zero small-W_hi / W_hh / W_oh entries cheapest-first while screening agreement stays >= floor."""
    Wi, Wh, Wo = (w.copy() for w in W)
    mats = {"hi": Wi, "hh": Wh, "oh": Wo}
    while True:
        cand = []
        for name, M in mats.items():
            for idx in zip(*np.nonzero(M)):
                if name == "hi" and BIG[idx]:
                    continue
                old = M[idx]
                M[idx] = 0.0
                cand.append((agree((Wi, Wh, Wo), X_SCREEN, REF_S), name, idx))
                M[idx] = old
        cand.sort(key=lambda c: -c[0])
        done_any = False
        for _, name, idx in cand[:8]:  # the 8 cheapest, one by one, re-checked cumulatively
            M = mats[name]
            old = M[idx]
            M[idx] = 0.0
            if agree((Wi, Wh, Wo), X_SCREEN, REF_S) >= floor:
                done_any = True
            else:
                M[idx] = old
        if not done_any:
            return Wi, Wh, Wo


def main():
    log = {}
    W = gauge(Wi0, Wh0, Wo0)
    log["G1+G2 exact gauge"] = (agree(W), literals(*W))
    for k in (2, 3, 4):
        log[f"round all to {k} sig digits"] = (agree(rounded(W, k)), literals(*W))
    for floor in (0.997, 0.995, 0.99):
        Wp = greedy_prune(W, floor)
        log[f"prune to screen>={floor}"] = (agree(Wp), literals(*Wp))
        for k in (3, 4):
            log[f"prune to screen>={floor} + round {k} digits"] = (agree(rounded(Wp, k)), literals(*Wp))
        np.savez(f"ladder_pruned_{floor}.npz", Wi=Wp[0], Wh=Wp[1], Wo=Wp[2])
    print(f"bar (noise floor) = {BAR}")
    for k, (a, n) in log.items():
        print(f"  {k:44s} agree {a:.4f}  literals {n:4d}  {'>= bar' if a >= BAR else '< bar'}")
    json.dump({k: {"agree": a, "literals": n} for k, (a, n) in log.items()}, open("ladder.json", "w"), indent=1)


if __name__ == "__main__":
    main()
