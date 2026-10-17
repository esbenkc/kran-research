"""Which neurons can drop their ReLU (become linear) without changing argmax?
Run: .venv/bin/python linB.py"""
import numpy as np
from core import N, run, sample, agreement
from subA import load_xunits

def run_lin(x, W, lin):
    W_hi, W_hh, W_oh = W
    h = np.zeros((x.shape[0], 16)); lin = np.array(lin, int)
    for t in range(N):
        pre = h @ W_hh.T + x[:, t:t+1] * W_hi
        h = np.maximum(pre, 0.0)
        if len(lin): h[:, lin] = pre[:, lin]
    return h @ W_oh.T

if __name__ == "__main__":
    W = load_xunits(); x = sample(200_000, 6); p = run(x, *W).argmax(1)
    for n in range(16):
        print(n, round(agreement(run_lin(x, W, [n]).argmax(1), p), 4))
    for S in [[12, 13, 14], [10, 11, 12, 13, 14], [2], [2, 12, 13, 14], [2, 10, 11, 12, 13, 14]]:
        print(S, round(agreement(run_lin(x, W, S).argmax(1), p), 4))
