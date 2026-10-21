"""Vera: one pass over a state's tok.npy -> country-token acts [2992,30,d] fp16 and sentence averages [2992,30,d] fp32.
Light (~1.2 GB). Usage: python3 vera_feats.py <state> [<state> ...]
"""
import sys
from pathlib import Path

import numpy as np

from vera_core import ACTS, HERE

FE = HERE / "feats"
FE.mkdir(exist_ok=True)
for state in sys.argv[1:]:
    if (FE / f"{state}_sent.npy").exists():
        print(state, "exists"); continue
    T = np.load(ACTS / state / "tok.npy", mmap_mode="r")
    o = np.load(ACTS / state / "off.npy")
    n = len(o) - 1
    ctry = np.asarray(T[o[1:] - 2], np.float16)
    S = np.zeros((n, T.shape[1], T.shape[2]), np.float32)
    for a in range(0, n, 128):
        b = min(a + 128, n)
        blk = np.asarray(T[o[a]:o[b]], np.float32)
        S[a:b] = np.add.reduceat(blk, o[a:b] - o[a], axis=0) / np.diff(o[a:b + 1])[:, None, None]
    np.save(FE / f"{state}_ctry.npy", ctry)
    np.save(FE / f"{state}_sent.npy", S)
    print(state, ctry.shape, S.shape, flush=True)
