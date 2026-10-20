"""Vera's own feature extraction from build's raw memmaps (only the raw activations are reused; no build code).
For each read (e.g. base_own, ens_s1_own, ens_s1_heldtext) writes to skeptic/feats/<read>.npz:
  meanL  (n, 24, 896) float32 : mean over answer tokens at each of the 24 block outputs
  pca    ragged token tensor (Ntok, 24, K) float16 : per-layer PCA-K projection of every answer token
  off    (n+1,) int64 token offsets into pca
PCA basis: per layer, fit on the tokens of this read (label-free, so no label leak; stated in signoff).
Streams the memmap in chunks; peak RAM ~0.6 GB.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_feats.py <read> [<read> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_io import CACHE, read_offsets  # noqa: E402

K = 64
CH = 4096
OUT = Path(__file__).resolve().parent / "feats"
OUT.mkdir(exist_ok=True)


def run(read):
    tok = np.load(CACHE / read / "tok.npy", mmap_mode="r")
    off = read_offsets(read)
    N, L, H = tok.shape
    assert off[-1] == N, (off[-1], N)
    n = len(off) - 1
    # pass 1: per-layer mean and second moment, and per-answer means
    s1 = np.zeros((L, H)); s2 = np.zeros((L, H, H))
    meanL = np.zeros((n, L, H), np.float32)
    row_of_tok = np.repeat(np.arange(n), np.diff(off))
    for a in range(0, N, CH):
        x = np.asarray(tok[a:a + CH], np.float32)
        s1 += x.sum(0)
        for l in range(L):
            s2[l] += x[:, l].T.astype(np.float64) @ x[:, l]
        np.add.at(meanL, row_of_tok[a:a + CH], x)
    meanL /= np.diff(off)[:, None, None]
    mu = s1 / N
    basis = np.zeros((L, H, K), np.float32)
    for l in range(L):
        C = s2[l] / N - np.outer(mu[l], mu[l])
        w, V = np.linalg.eigh(C)
        basis[l] = V[:, ::-1][:, :K]
    pca = np.zeros((N, L, K), np.float16)
    for a in range(0, N, CH):
        x = np.asarray(tok[a:a + CH], np.float32) - mu[None].astype(np.float32)
        pca[a:a + CH] = np.einsum("tlh,lhk->tlk", x, basis).astype(np.float16)
    np.savez(OUT / f"{read}.npz", meanL=meanL, pca=pca, off=off, mu=mu.astype(np.float32))
    print(read, "n", n, "tokens", N, "saved", flush=True)


for r in sys.argv[1:]:
    run(r)
