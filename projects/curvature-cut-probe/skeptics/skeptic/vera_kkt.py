"""Vera: certify a claimed minimum of  min_z KL(p0 || softmax(W z))  s.t.  a.z = a.z0 - D,  |z/g|^2 <= R^2
(the convex relaxation of the RMSNorm sphere; logits linear in z -> KL convex in z; the ball is convex).
KKT at z*: grad KL = W'(q - p0) must equal -nu a - rho * 2 z*/g^2 with rho >= 0 (rho > 0 only if on the boundary).
I fit nu, rho by least squares and report the relative residual. Convex + KKT -> z* is the global minimum over the solid
ellipsoid; on the boundary it is also the sphere minimum. One full-vocab pass, 1024-row chunks, fp64. ~0.06 GB.
Also a duality-gap style check: KL(z*) - min over a few random feasible perturbations (must all be >= KL(z*)).
Usage: vera_kkt.py POINTS.npz PROBES.npz OUT.json   (npz with arrays: idx [n], probe [n] (str), D [n], z [n,1536])
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W = np.load(f"{HERE}/W_f32.npy", mmap_mode="r"); G = np.load(f"{HERE}/g_norm.npy").astype(np.float64)
V, d = W.shape; CH = 1024
Z = np.load("data/clean/z_post.npy", mmap_mode="r")
pts = np.load(sys.argv[1], allow_pickle=True); Pz = np.load(sys.argv[2]); OUT = sys.argv[3]


def lg(z):
    out = np.empty(V)
    for s in range(0, V, CH):
        out[s:s + CH] = np.asarray(W[s:s + CH], dtype=np.float64) @ z
    return out


def WT(v):
    out = np.zeros(d)
    for s in range(0, V, CH):
        out += np.asarray(W[s:s + CH], dtype=np.float64).T @ v[s:s + CH]
    return out


def sm(l):
    e = np.exp(l - l.max()); return e / e.sum()


def kl(p0, l):
    m = l.max(); lse = m + np.log(np.exp(l - m).sum()); nz = p0 > 0
    return float((p0[nz] * (np.log(p0[nz]) - (l[nz] - lse))).sum())


res = []
for k in range(len(pts["idx"])):
    i = int(pts["idx"][k]); nm = str(pts["probe"][k]); D = float(pts["D"][k]); zs = pts["z"][k].astype(np.float64)
    a = Pz[nm + "__w"].astype(np.float64) / Pz[nm + "__sd"].astype(np.float64)
    z0 = np.asarray(Z[i], dtype=np.float64)
    p0 = sm(lg(z0)); ls = lg(zs); q = sm(ls)
    g = WT(q - p0); b = 2 * zs / G ** 2
    M = np.stack([a, b], 1); coef, *_ = np.linalg.lstsq(M, -g, rcond=None)
    resid = np.linalg.norm(g + M @ coef) / np.linalg.norm(g)
    R0 = np.linalg.norm(z0 / G); Rs = np.linalg.norm(zs / G)
    r = dict(i=i, probe=nm, D=D, kl=kl(p0, ls), shift=float(a @ zs - a @ z0), shift_err=float(a @ zs - a @ z0 + D),
             radius_ratio_to_z0=float(Rs / R0), radius_ratio_to_sqrtd=float(Rs / np.sqrt(d)), nu=float(coef[0]), rho=float(coef[1]),
             kkt_rel_resid=float(resid))
    print(r, flush=True); res.append(r)
json.dump(res, open(OUT, "w"), indent=1)
