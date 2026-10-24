"""Vera: low-footprint (<0.1 GB) sphere found-move for given (statement, probe, shift) triples.
Same method as vera_gate_np.sph_move (L-BFGS-B on z = g*sqrt(dim)*v/|v|, penalty ladder 1e1..1e6, top-K words of p0),
then the found point is scored on the FULL vocabulary (1024-row chunks). A found move is an upper bound on the minimum.
Usage: vera_sph_lm.py PROBES.npz OUT.json i:probe[:frac] ...   (frac of D to move; default 1 = to the clean line;
       negative frac = move UP by frac*D of the paired false statement, given as i:probe:frac:Dsource)
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
W = np.load(f"{HERE}/W_f32.npy", mmap_mode="r"); G = np.load(f"{HERE}/g_norm.npy").astype(np.float64)
V, d = W.shape; CH = 1024; K = int(os.environ.get("VERA_K", "2048"))
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy", mmap_mode="r")
sp = np.array([e["split"] for e in S]); tr = np.array([e["true"] == 1 for e in S])
Pz = np.load(sys.argv[1]); OUT = sys.argv[2]


def probe(nm):
    w = Pz[nm + "__w"].astype(np.float64); sd = Pz[nm + "__sd"].astype(np.float64); mu = Pz[nm + "__mu"]; b = float(Pz[nm + "__b"])
    ls = np.where((sp == "refit") & tr)[0]
    s_ls = np.array([(np.asarray(Z[k], dtype=np.float64) - mu) / sd @ w + b for k in ls])
    return w / sd, mu, sd, w, b, float(np.sort(s_ls)[::-1][1])


def logits_full(z):
    out = np.empty(V)
    for s in range(0, V, CH):
        out[s:s + CH] = np.asarray(W[s:s + CH], dtype=np.float64) @ z
    return out


def kl_from_logits(p0, l):
    m = l.max(); lse = m + np.log(np.exp(l - m).sum()); nz = p0 > 0
    return float((p0[nz] * (np.log(p0[nz]) - (l[nz] - lse))).sum())


def sph(z0, a, x, rounds=3):
    """Active-set rounds: optimise on the top-K words of p0 united with the top-K words of the current q."""
    l0 = logits_full(z0); p0 = np.exp(l0 - l0.max()); p0 /= p0.sum()
    c = np.sqrt(d); target = a @ z0 - x; v = z0 / G
    idx = np.argpartition(-p0, K)[:K]
    for rd in range(rounds):
        idx = np.sort(np.unique(idx)); WK = np.asarray(W[idx], dtype=np.float64); pK = p0[idx]; mK = pK.sum(); lpK = np.log(np.maximum(pK, 1e-300))
        for mu in ((1e1, 1e2, 1e3, 1e4, 1e5, 1e6) if rd == 0 else (1e6,)):
            def fg(v):
                nv = np.linalg.norm(v); u = v / nv; z = c * G * u
                l = WK @ z; mx = l.max(); e = np.exp(l - mx); Zs = e.sum(); q = e / Zs
                f = float((pK * (lpK - (l - mx - np.log(Zs)))).sum()); r = a @ z - target
                gz = WK.T @ (mK * q - pK) + 2 * mu * r * a; gu = c * G * gz
                return f + mu * r * r, (gu - (gu @ u) * u) / nv
            v = minimize(fg, v, jac=True, method="L-BFGS-B", options=dict(maxiter=400, ftol=1e-16, gtol=1e-14)).x
        z = c * G * v / np.linalg.norm(v); lz = logits_full(z)
        idx = np.concatenate([idx, np.argpartition(-lz, K)[:K]])
    return kl_from_logits(p0, lz), float(a @ z - a @ z0), float(np.linalg.norm(z - z0) / np.linalg.norm(z0))


res = []
cache = {}
for spec in sys.argv[3:]:
    parts = spec.split(":"); i = int(parts[0]); nm = parts[1]; frac = float(parts[2]) if len(parts) > 2 else 1.0
    if nm not in cache:
        cache[nm] = probe(nm)
    a, mu, sd, w, b, line = cache[nm]
    zsrc = int(parts[3]) if len(parts) > 3 else i
    Dsrc = (np.asarray(Z[zsrc], dtype=np.float64) - mu) / sd @ w + b - line
    z0 = np.asarray(Z[i], dtype=np.float64)
    t = time.time(); klv, sh, rel = sph(z0, a, frac * Dsrc)
    res.append(dict(i=i, probe=nm, frac=frac, D=float(Dsrc), kl=klv, shift=sh, rel_move=rel, sec=time.time() - t))
    print(res[-1], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
