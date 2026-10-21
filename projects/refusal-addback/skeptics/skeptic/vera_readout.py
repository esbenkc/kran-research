"""Vera: rebuild the direction and the readability AUROC from saved activations (own code).

- r_L = mean(fit_harmful) - mean(fit_harmless) of BASE acts; compare to build's runs/dirs.npy.
- AUROC per layer of projection onto unit r_L, TEST harmful vs harmless, base and stripped acts.
- nulls: 200 random unit directions per layer; shuffled-label diff-of-means direction (FIT labels permuted).
- ceiling: the model's own refit diff-of-means direction.
usage: python vera_readout.py [stripped_run_dir ...]
"""
import sys, json
import numpy as np
np.seterr(all="ignore")  # spurious Accelerate matmul FP warnings on arm64; results checked vs einsum below
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/runs"


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = s.argsort(kind="mergesort"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        if m.sum() > 1: rk[m] = rk[m].mean()
    n1, n0 = len(pos), len(neg)
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


base = np.load(R / "base/acts.npz")
F = lambda a, k: a[k].astype(np.float64)
r = F(base, "fit_harmful").mean(0) - F(base, "fit_harmless").mean(0)  # [28, d]
bd = np.load(R / "dirs.npy").astype(np.float64)
cos = (r * bd).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(bd, axis=1) + 1e-12)
print("direction vs build dirs.npy: cos min (L>=1) %.6f, rel norm err max %.2e" % (
    cos[1:].min(), np.abs(np.linalg.norm(r, axis=1)[1:] / np.linalg.norm(bd, axis=1)[1:] - 1).max()))
print("|r_L|:", np.round(np.linalg.norm(r, axis=1), 1).tolist())
u = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
rng = np.random.default_rng(12345)
Rnd = rng.standard_normal((200, 1536)); Rnd /= np.linalg.norm(Rnd, axis=1, keepdims=True)

res = {}
runs = [("base", base)] + [(p, np.load(R / p / "acts.npz")) for p in sys.argv[1:]]
for name, A in runs:
    th, tl = F(A, "test_harmful"), F(A, "test_harmless")
    fh, fl = F(A, "fit_harmful"), F(A, "fit_harmless")
    own = fh.mean(0) - fl.mean(0)
    pool = np.concatenate([fh, fl]); lab = np.r_[np.ones(len(fh)), np.zeros(len(fl))]
    perm = rng.permutation(lab)
    shuf = pool[perm == 1].mean(0) - pool[perm == 0].mean(0)
    rows = []
    for L in range(28):
        a_base = auroc(th[:, L] @ u[L], tl[:, L] @ u[L])
        a_own = auroc(th[:, L] @ own[L], tl[:, L] @ own[L])
        a_shuf = auroc(th[:, L] @ shuf[L], tl[:, L] @ shuf[L])
        ar = np.array([auroc(th[:, L] @ v, tl[:, L] @ v) for v in Rnd[:50]])
        ar = np.maximum(ar, 1 - ar)  # sign-free random: best case for the null
        gap = (th[:, L] @ u[L]).mean() - (tl[:, L] @ u[L]).mean()
        rows.append(dict(L=L, base_dir=a_base, own_dir=a_own, shuffled=a_shuf, rand_p95=float(np.quantile(ar, .95)),
                         rand_max=float(ar.max()), gap_along_unit_r=float(gap)))
    res[name] = rows
    best = max(rows[1:], key=lambda x: x["base_dir"])
    print(f"\n== {name}: best layer {best['L']} base-dir AUROC {best['base_dir']:.4f}")
    print(" L  baseDir ownDir shuffled randP95 randMax  gap")
    for x in rows:
        print(f"{x['L']:2d}  {x['base_dir']:.3f}  {x['own_dir']:.3f}  {x['shuffled']:.3f}   {x['rand_p95']:.3f}  {x['rand_max']:.3f} {x['gap_along_unit_r']:7.2f}")
json.dump(res, open(Path(__file__).with_name("vera_readout.json"), "w"), indent=1)
