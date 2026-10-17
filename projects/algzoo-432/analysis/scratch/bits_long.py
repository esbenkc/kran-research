"""Bits for Bruno's freeze-rounded nets, same formula as bits_curve.py:bits (copied, k=3)."""
from math import lgamma, log
import numpy as np
def log2_binom(n, k): return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)
def bits(W, k):
    Wi, Wh, Wo = W
    allw = np.concatenate([Wh.ravel(), Wo[1:].ravel()])
    free = allw[(allw != 0) & (np.abs(allw) != 1.0)]
    e = np.floor(np.log10(np.abs(free))).astype(int)
    _, cnt = np.unique(e, return_counts=True); p = cnt / cnt.sum()
    per = 1 + np.log2(9 * 10 ** (k - 1)) + float(-(p * np.log2(p)).sum())
    mask = log2_binom(256, int((Wh != 0).sum())) + log2_binom(150, int((Wo[1:] != 0).sum()))
    return len(free), per * len(free) + mask + 16
for f in ["prune_ft_long_220_144_q3f","prune_ft_long_210_140_q3f","prune_ft_long_200_140_q3f","prune_ft_long_190_135_q3f","prune_ft_long_180_130_q3f","prune_ft_long_170_125_q3f"]:
    d=np.load(f+".npz"); n,b=bits((d["W_hi"],d["W_hh"],d["W_oh"]),3); print(f,n,round(b))
