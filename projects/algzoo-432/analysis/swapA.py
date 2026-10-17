"""Swap test: replace A neurons by idealized closed forms inside the recurrence; B neurons run as usual.
Run: .venv/bin/python swapA.py"""
import numpy as np
from core import N, run, sample, agreement, second_argmax
from subA import load_xunits, ideal

def run_swap(x, W, swap):
    W_hi, W_hh, W_oh = W
    B = x.shape[0]; h = np.zeros((B, 16))
    for t in range(N):
        h = np.maximum(h @ W_hh.T + x[:, t:t+1] * W_hi, 0.0)
        if swap:
            f = ideal(x, t + 1)
            for n in swap: h[:, n] = f[n]
    return h @ W_oh.T

if __name__ == "__main__":
    W = load_xunits(); x = sample(200_000, 2)
    p = run(x, *W).argmax(1); err = p != second_argmax(x)
    for sw in [[2], [4], [7], [6], [1], [2, 4], [2, 4, 7], [2, 4, 7, 6], [2, 4, 7, 6, 1]]:
        q = run_swap(x, W, sw).argmax(1)
        print(sw, "agree", round(agreement(q, p), 4), "on model-error set", round(agreement(q[err], p[err]), 4))
