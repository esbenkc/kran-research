"""Cue directions: 3 random unit vectors in R^4096 (one per template row), numpy default_rng(seed).standard_normal,
row-normalised in float64, stored float32. Seed 1110 reproduces the parent's ref/cue_u.npy bit for bit (asserted);
seed 1118 is this run's fresh direction set for the exploratory cells (PREREG section 3).
PREREG section 2: u is "saved once in data/cue_u.npy (bit-identical to the parent file; sha256 logged)".

    python make_u.py   # writes data/cue_u.npy, data/cue_u_1118.npy
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def make(seed: int, d: int = 4096) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((3, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


assert np.array_equal(make(1110), np.load(HERE / "ref" / "cue_u.npy")), "seed 1110 must reproduce the parent u"
(HERE / "data").mkdir(exist_ok=True)
np.save(HERE / "data" / "cue_u.npy", make(1110))
np.save(HERE / "data" / "cue_u_1118.npy", make(1118))
import hashlib
for f in ("cue_u.npy", "cue_u_1118.npy"):
    print(f, hashlib.sha256((HERE / "data" / f).read_bytes()).hexdigest())
u = make(1118)
print("cue_u_1118: norms", np.linalg.norm(u, axis=1).tolist(), "cos with u1110 rows",
      [float(u[p] @ make(1110)[p]) for p in range(3)])
