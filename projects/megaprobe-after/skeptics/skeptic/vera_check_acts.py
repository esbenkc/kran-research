"""Vera: compare my batch-1 CPU re-extraction (vera_reextract.npz) against build's memmaps, per state and layer.
Reports per layer: median cosine of token vectors, max relative L2 error, and the same for mean-pooled vectors.
    python3 vera_check_acts.py
"""
import json

import numpy as np

from vera_core import HERE, PROJ

f = np.load(HERE / "vera_reextract.npz")
idx = f["idx"]
off = np.load(PROJ / "data/acts/clean/off.npy")
out = {}
for s, mine in [("clean", "clean"), ("seed0", "s0"), ("seed1", "s1"), ("seed2", "s2")]:
    t = np.load(PROJ / f"data/acts/{s}/tok.npy", mmap_mode="r")
    A = f[f"{mine}__acts"].astype(np.float32); mo = f[f"{mine}__off"]
    B = np.concatenate([np.asarray(t[off[i]:off[i + 1], 1:29], np.float32) for i in idx])
    assert A.shape == B.shape, (A.shape, B.shape)
    cos = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
    rel = np.linalg.norm(A - B, axis=-1) / np.linalg.norm(B, axis=-1)
    Am = np.add.reduceat(A, mo[:-1], 0); Bm = np.add.reduceat(B, mo[:-1], 0)
    relm = np.linalg.norm(Am - Bm, axis=-1) / np.linalg.norm(Bm, axis=-1)
    out[s] = {"min_cos_by_layer": cos.min(0).round(5).tolist(), "median_rel_by_layer": np.median(rel, 0).round(5).tolist(),
              "max_rel_meanpool_by_layer": relm.max(0).round(5).tolist()}
    print(s, "min cos over tokens, layers 1..28:", cos.min(0).round(4).tolist())
    print(s, "max rel err mean-pooled, layers 1..28:", relm.max(0).round(4).tolist(), flush=True)
json.dump(out, open(HERE / "vera_check_acts.json", "w"), indent=1)
