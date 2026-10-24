"""Vera power gate, numpy/scipy only (~0.1-0.2 GB footprint; runs next to the fine-tune).

W = tied unembedding, fp32 memmap W_f32.npy (vera_dumpW.py, from the HF safetensors), streamed in 2048-row chunks, fp64 math.
Per eval twin pair (false i, true twin j, same city), for each probe (score s = a.z + c, a = w/sd):
  line = PREREG s5 on the clean model: 2nd-highest score of the 200 line-setting ('refit') true statements
  D    = s_false - line   (pairs with the false statement caught on the clean model only)
  g_x  = a'(F_x + eps I)^-1 a, F_x = W'(diag p - pp')W full vocab, eps = 1e-8 lam_max (also 1e-10, 1e-6)
  quad false-only = D^2/(2 g_f);  quad pair = D^2/(2 (g_f + g_t))  (false down x, true up D-x, optimal split)
  sphere found-move c(x): cheapest KL(p0||softmax(W z)) over the RMSNorm image z = g*sqrt(dim)*v/|v|
        with a.z = a.z0 - x (x<0 = move up); L-BFGS-B on the top-K words of p0 + penalty ladder, then scored on the
        FULL vocabulary. sphere false-only = c_f(D); sphere pair = min over x in {0,D/4,D/2,3D/4,D} of c_f(x)+c_t(D-x).
  cvx found-move: the same without the sphere (free z), false-only.
Usage: vera_gate_np.py PROBES.npz names N_PAIRS N_SPH SEED OUT.json
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
probes_f, names, NP, NS, SEED, OUT = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
HERE = os.path.dirname(os.path.abspath(__file__))
W = np.load(f"{HERE}/W_f32.npy", mmap_mode="r"); G = np.load(f"{HERE}/g_norm.npy").astype(np.float64)
V, d = W.shape; CH = 2048; K = 4096
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy", mmap_mode="r")
Pz = np.load(probes_f)
sp = np.array([e["split"] for e in S]); tr = np.array([e["true"] == 1 for e in S]); city = np.array([e["city"] for e in S])


def chunks():
    for s in range(0, V, CH):
        yield s, np.asarray(W[s:s + CH], dtype=np.float64)


def logits(z):
    return np.concatenate([Wc @ z for _, Wc in chunks()])


def softmax(l):
    e = np.exp(l - l.max()); return e / e.sum()


FAST = os.environ.get("VERA_FAST", "0") == "1"   # fp32 chunk GEMMs (AMX), fp64 accumulation across chunks


def fisher(z):
    p = softmax(logits(z)); F = np.zeros((d, d)); mu = np.zeros(d)
    if FAST:   # the 512 most likely words exactly in fp64, the rest (small p) in fp32 GEMMs
        top = np.argpartition(-p, 512)[:512]; Wt = np.asarray(W[np.sort(top)], dtype=np.float64); pt = p[np.sort(top)]
        F += Wt.T @ (pt[:, None] * Wt); mu += Wt.T @ pt
        prest = p.copy(); prest[top] = 0.0
    for s in range(0, V, 8192 if FAST else CH):
        if FAST:
            Wc = np.asarray(W[s:s + 8192]); pc = prest[s:s + 8192]
            F += (Wc.T @ (pc.astype(np.float32)[:, None] * Wc)).astype(np.float64)
            mu += Wc.astype(np.float64).T @ pc
        else:
            Wc = np.asarray(W[s:s + CH], dtype=np.float64); pc = p[s:s + CH]
            F += Wc.T @ (pc[:, None] * Wc); mu += Wc.T @ pc
    return F - np.outer(mu, mu), p


def kl_full(p0, z):
    l = logits(z); m = l.max(); lse = m + np.log(np.exp(l - m).sum())
    nz = p0 > 0
    return float((p0[nz] * (np.log(p0[nz]) - (l[nz] - lse))).sum())


class Top:
    def __init__(self, p0):
        self.idx = np.sort(np.argpartition(-p0, K)[:K]); self.WK = np.asarray(W[self.idx], dtype=np.float64)
        self.p0 = p0[self.idx]; self.m = self.p0.sum(); self.lp0 = np.log(np.maximum(self.p0, 1e-300))
    def f_g(self, z):   # KL on the K words and its gradient wrt z
        l = self.WK @ z; mx = l.max(); e = np.exp(l - mx); Zs = e.sum(); q = e / Zs; lq = l - mx - np.log(Zs)
        f = float((self.p0 * (self.lp0 - lq)).sum())
        return f, self.WK.T @ (self.m * q - self.p0)


def cvx_move(T, z0, a, x):
    an = a / np.linalg.norm(a); base = -x * a / (a @ a)
    def fg(v):
        dd = base + v - (v @ an) * an; f, gz = T.f_g(z0 + dd); gv = gz - (gz @ an) * an; return f, gv
    r = minimize(fg, np.zeros(d), jac=True, method="L-BFGS-B", options=dict(maxiter=500, ftol=1e-16, gtol=1e-13))
    v = r.x; dd = base + v - (v @ an) * an
    return z0 + dd


def sph_move(T, z0, a, x, starts):
    c = np.sqrt(d); target = a @ z0 - x; best = None
    for v0 in starts:
        v = v0.copy()
        for mu in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            def fg(v):
                nv = np.linalg.norm(v); u = v / nv; z = c * G * u
                f, gz = T.f_g(z); r = a @ z - target; f = f + mu * r * r; gz = gz + 2 * mu * r * a
                gu = c * G * gz; gv = (gu - (gu @ u) * u) / nv
                return f, gv
            v = minimize(fg, v, jac=True, method="L-BFGS-B", options=dict(maxiter=200, ftol=1e-16, gtol=1e-13)).x
        z = c * G * v / np.linalg.norm(v)
        if best is None or T.f_g(z)[0] < T.f_g(best)[0]:
            best = z
    return best


A = {}
for nm in names:
    a = Pz[nm + "__w"].astype(np.float64) / Pz[nm + "__sd"].astype(np.float64)
    s = (np.asarray(Z, dtype=np.float64) - Pz[nm + "__mu"]) / Pz[nm + "__sd"] @ Pz[nm + "__w"].astype(np.float64) + float(Pz[nm + "__b"])
    line = np.sort(s[(sp == "refit") & tr])[::-1][1]
    A[nm] = dict(a=a, s=s, line=float(line), caught=set(np.where((sp == "eval") & ~tr & (s > line))[0].tolist()))
    print(nm, "line %.4f" % line, "eval catch", len(A[nm]["caught"]), "/596", flush=True)
fal = [i for i in range(len(S)) if sp[i] == "eval" and not tr[i]]
twin = {i: [j for j in range(len(S)) if sp[j] == "eval" and tr[j] and city[j] == city[i]][0] for i in fal}
rng = np.random.default_rng(SEED)
cand = [i for i in fal if any(i in A[n]["caught"] for n in names)]
sub = rng.permutation(cand)[:NP]
rows = []; t0 = time.time()
for jn, i in enumerate(sub):
    i = int(i); j = twin[i]
    zf = np.asarray(Z[i], dtype=np.float64); zt = np.asarray(Z[j], dtype=np.float64)
    Ff, pf = fisher(zf); Ft, pt = fisher(zt)
    ef, Qf = np.linalg.eigh(Ff); et, Qt = np.linalg.eigh(Ft)
    row = dict(i=i, twin=j, maxp_f=float(pf.max()), maxp_t=float(pt.max()))
    Tf = Top(pf) if jn < NS else None; Tt = Top(pt) if jn < NS else None
    for nm in names:
        if i not in A[nm]["caught"]:
            continue
        a = A[nm]["a"]; D = float(A[nm]["s"][i] - A[nm]["line"]); r = dict(D=D)
        cf = Qf.T @ a; ct = Qt.T @ a
        for rel in (1e-10, 1e-8, 1e-6):
            gf = float((cf ** 2 / (np.maximum(ef, 0) + rel * ef.max())).sum()); gt = float((ct ** 2 / (np.maximum(et, 0) + rel * et.max())).sum())
            r[f"quadF_{rel:g}"] = D * D / (2 * gf); r[f"quadP_{rel:g}"] = D * D / (2 * (gf + gt))
        r["flat_f_1e-4"] = float((cf[ef < 1e-4 * ef.max()] ** 2).sum() / (cf ** 2).sum())
        if Tf is not None:
            zc = cvx_move(Tf, zf, a, D); r["cvxF"] = kl_full(pf, zc)
            xs = [0.25 * D, 0.5 * D, 0.75 * D, D]; cfx = {0.0: 0.0}; ctx = {0.0: 0.0}
            for x in xs:
                zs = sph_move(Tf, zf, a, x, [zf / G]); cfx[x] = kl_full(pf, zs)
                zs = sph_move(Tt, zt, a, -x, [zt / G]); ctx[x] = kl_full(pt, zs)
            r["sphF"] = cfx[D]; r["sph_c_false"] = [cfx[x] for x in [0.0] + xs]; r["sph_c_true"] = [ctx[x] for x in [0.0] + xs]
            r["sphP"] = min(cfx[x] + ctx[D - x if (D - x) in ctx else min(ctx, key=lambda y: abs(y - (D - x)))] for x in [0.0] + xs)
        row[nm] = r
    rows.append(row)
    print(jn, i, {nm: {k: float("%.3g" % v) for k, v in row[nm].items() if k in ("D", "quadF_1e-08", "quadP_1e-08", "cvxF", "sphF", "sphP")} for nm in names if nm in row}, "%.0fs" % (time.time() - t0), flush=True)
    json.dump(dict(lines={n: A[n]["line"] for n in names}, n_caught={n: len(A[n]["caught"]) for n in names}, seed=SEED, rows=rows), open(OUT, "w"), indent=1)
