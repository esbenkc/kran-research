"""Own check of build's cache (task 'cache all-layer acts'): row order, offsets, and the layer-12/16 answer-token means
(my vera_feats meanL) against 1019's saved acts_<policy>.npz (own_L12/own_L16 for *_own reads, base_L12/base_L16 for
held-text *_base reads). Also: held-text reads must differ from own reads (they come from different weights).
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_cachecheck.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import D19, load_labels  # noqa: E402
from vera_io import D20, read_meta, read_offsets  # noqa: E402

HERE = Path(__file__).resolve().parent
reads = ["base_own"] + [f"{p}_{r}" for p in ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"] for r in ["own", "base"]]


def cos_rows(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


for rd in reads:
    pol, kind = rd.rsplit("_", 1)
    meta = read_meta(rd); off = read_offsets(rd)
    L = load_labels(pol, D20)
    assert meta["prompt_ids"] == [r["prompt_id"] for r in L][:len(meta["prompt_ids"])], rd
    f = np.load(HERE / "feats" / f"{rd}.npz")
    old = np.load(D19 / f"acts_{pol}.npz")
    assert list(old["prompt_id"][:len(off) - 1]) == meta["prompt_ids"], rd
    key = "own" if kind == "own" else "base"
    out = [rd, f"n={len(off)-1}", f"adapter={meta.get('adapter')}"]
    for li, ln in [(11, "L12"), (15, "L16")]:
        c = cos_rows(f["meanL"][:, li].astype(np.float64), old[f"{key}_{ln}"].astype(np.float64))
        out.append(f"{ln} cos min {c.min():.6f} median {np.median(c):.6f} n<0.999 {(c < 0.999).sum()}")
    print(" | ".join(out))
for p in ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    a = np.load(HERE / "feats" / f"{p}_own.npz")["meanL"][:, 11]; b = np.load(HERE / "feats" / f"{p}_base.npz")["meanL"][:, 11]
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    print(p, "own vs held-text L12 median rel diff %.3f (must be > 0)" % np.median(rel))
