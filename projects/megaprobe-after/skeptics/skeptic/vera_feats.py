"""Vera: own features from build's raw token memmaps (data/acts/<state>/tok.npy [n_tok, 29, 1536] fp16, off.npy).
Writes skeptic/feats/<state>_mean.npy  : [2992, 29, 1536] float32, mean over tokens 1..n-1 (all stored tokens)
       skeptic/feats/pca.npz           : per-layer PCA-64 basis (l = 1..28) fit on CLEAN training-pool tokens, no labels
       skeptic/feats/<state>_pca.npy   : [n_tok, 28, 64] float32 projections (same basis for every state)
Memory: chunks of 2000 tokens (~180 MB fp16), <1.5 GB peak.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_feats.py clean seed0 seed1 seed2
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1] / "kran-research/projects/megaprobe-after"
FE = HERE / "feats"; FE.mkdir(exist_ok=True)
K = 64
rows = json.load(open(PROJ / "data/ref1019/split.json"))
pool = np.array([r["split"] in ("probe", "attack", "stop") for r in rows])


def open_state(s):
    t = np.load(PROJ / f"data/acts/{s}/tok.npy", mmap_mode="r")
    off = np.load(PROJ / f"data/acts/{s}/off.npy") if (PROJ / f"data/acts/{s}/off.npy").exists() else np.load(PROJ / "data/acts/clean/off.npy")
    return t, off


def mean_pool(s):
    t, off = open_state(s)
    n = len(off) - 1
    out = np.zeros((n, t.shape[1], t.shape[2]), np.float32)
    for a in range(0, n, 200):
        b = min(n, a + 200)
        x = np.asarray(t[off[a]:off[b]], dtype=np.float32)
        out[a:b] = np.add.reduceat(x, off[a:b] - off[a], axis=0)
    out /= np.diff(off)[:, None, None]
    np.save(FE / f"{s}_mean.npy", out)
    return off


def fit_pca():
    t, off = open_state("clean")
    tid = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(pool)[0]])
    # z-score every dim on clean pool tokens first (Qwen has a few huge-magnitude dims), then PCA; the scaling is
    # folded into V so that project() computes ((x - mu) / sd) @ V_pca
    mu = np.zeros((28, 1536), np.float32); V = np.zeros((28, 1536, K), np.float32); ev = np.zeros(28)
    for l in range(1, 29):
        X = np.asarray(t[tid, l], dtype=np.float32)
        m = X.mean(0); sd = X.std(0) + 1e-3; Xc = (X - m) / sd
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        mu[l - 1] = m; V[l - 1] = Vt[:K].T / sd[:, None]; ev[l - 1] = (S[:K] ** 2).sum() / (S ** 2).sum()
    np.savez(FE / "pca.npz", mu=mu, V=V, ev=ev, n_tok=len(tid))
    print("pca top-64 var share per layer 1..28:", np.round(ev, 3).tolist(), flush=True)
    return mu, V


def project(s, mu, V):
    t, _ = open_state(s)
    out = np.zeros((t.shape[0], 28, K), np.float32)
    for a in range(0, t.shape[0], 2000):
        x = np.asarray(t[a:a + 2000, 1:29], dtype=np.float32)
        out[a:a + 2000] = np.einsum("tld,ldk->tlk", x - mu[None], V)
    np.save(FE / f"{s}_pca.npy", out)


if __name__ == "__main__":
    states = sys.argv[1:]
    if (FE / "pca.npz").exists():
        f = np.load(FE / "pca.npz"); mu, V = f["mu"], f["V"]
    else:
        mu, V = fit_pca()
    for s in states:
        mean_pool(s); project(s, mu, V); print(s, "done", flush=True)
