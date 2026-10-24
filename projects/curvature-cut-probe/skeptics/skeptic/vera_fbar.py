"""Vera: rebuild the mean output Fisher Fbar = mean_i W'(diag p_i - p_i p_i')W over a set of clean states, my own code.
Fbar = W' diag(pbar) W - mean_i mu_i mu_i',  mu_i = W'p_i.  W streamed from W_f32.npy (fp32 memmap), fp64 accumulation.
Compares with build's data/fisher/Fbar.npy (relative Frobenius error, top eigenvalues, counts above rel cuts, subspace
overlap of the top-128 eigenvectors with subspaces.npz U_kstar).
Usage: vera_fbar.py SPLITS (comma list of split names, e.g. probe,attack,stop)
"""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W = np.load(f"{HERE}/W_f32.npy", mmap_mode="r"); V, d = W.shape; CH = 4096
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy", mmap_mode="r")
splits = sys.argv[1].split(","); idx = [i for i, e in enumerate(S) if e["split"] in splits]
print("n states", len(idx), flush=True)
t0 = time.time(); ZZ = np.asarray(Z[idx], dtype=np.float32).T            # d x n
n = len(idx); mx = np.full(n, -np.inf); se = np.zeros(n)
for s in range(0, V, CH):                                                # pass 1: log-sum-exp per state
    L = (np.asarray(W[s:s + CH]) @ ZZ).astype(np.float64); m2 = np.maximum(mx, L.max(0))
    se = se * np.exp(mx - m2) + np.exp(L - m2).sum(0); mx = m2
lse = mx + np.log(se); pbar = np.zeros(V); MU = np.zeros((d, n))
for s in range(0, V, CH):                                                # pass 2: pbar and mu_i = W'p_i
    Wc = np.asarray(W[s:s + CH]); P_ = np.exp((Wc @ ZZ).astype(np.float64) - lse)
    pbar[s:s + CH] = P_.sum(1); MU += Wc.astype(np.float64).T @ P_
print("passes %.0fs" % (time.time() - t0), flush=True)
M = MU @ MU.T
pbar /= len(idx); M /= len(idx)
F = -M
for s in range(0, V, CH):
    Wc = np.asarray(W[s:s + CH], dtype=np.float64); F += Wc.T @ (pbar[s:s + CH, None] * Wc)
ev, Q = np.linalg.eigh(F); ev = ev[::-1]; Q = Q[:, ::-1]
Fb = np.load("data/fisher/Fbar.npy"); eb = np.linalg.eigvalsh(Fb)[::-1]
Uk = np.load("data/fits/subspaces.npz")["U_kstar"]
ov = np.linalg.svd(Q[:, :128].T @ Uk, compute_uv=False)
out = dict(splits=splits, n=len(idx), rel_fro=float(np.linalg.norm(F - Fb) / np.linalg.norm(Fb)),
           top5_mine=ev[:5].tolist(), top5_build=eb[:5].tolist(),
           n_above={c: [int((ev > c * ev[0]).sum()), int((eb > c * eb[0]).sum())] for c in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)},
           eig128_rel=[float(ev[127] / ev[0]), float(eb[127] / eb[0])],
           top128_overlap_min_cos=float(ov.min()), top128_overlap_mean_cos2=float((ov ** 2).mean()))
print(json.dumps(out, indent=1))
json.dump(out, open(f"skeptic/vera_fbar_{'_'.join(splits)}.json", "w"), indent=1)
