"""Cue directions: 3 random unit vectors in R^4096 (one per template row), numpy default_rng(seed).standard_normal,
row-normalised in float64, stored float32. Seed 1110 reproduces the parent's ref/cue_u.npy bit for bit (asserted).
Seeds 1125, 1126, 1127 (= 1120 + RL seed) are this run's fresh direction sets for S9 (PREREG section 3): S9 seed s
uses data/cue_u_{1120+s}.npy.
PREREG section 2: u is "saved once in data/cue_u.npy (bit-identical to the parent file; sha256 logged)".

    python make_u.py   # writes data/cue_u.npy, data/cue_u_1125.npy, data/cue_u_1126.npy, data/cue_u_1127.npy
"""
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
S9_SEEDS = (5, 6, 7)


def make(seed: int, d: int = 4096) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((3, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    assert np.array_equal(make(1110), np.load(HERE / "ref" / "cue_u.npy")), "seed 1110 must reproduce the parent u"
    (HERE / "data").mkdir(exist_ok=True)
    files = {"cue_u.npy": make(1110)} | {f"cue_u_{1120 + s}.npy": make(1120 + s) for s in S9_SEEDS}
    for f, u in files.items():
        np.save(HERE / "data" / f, u)
        print(f, hashlib.sha256((HERE / "data" / f).read_bytes()).hexdigest())
    base = make(1110)
    for s in S9_SEEDS:
        u = make(1120 + s)
        print(f"cue_u_{1120 + s}: norms", [round(float(x), 7) for x in np.linalg.norm(u, axis=1)],
              "cos with u1110 rows", [round(float(u[p] @ base[p]), 4) for p in range(3)])
