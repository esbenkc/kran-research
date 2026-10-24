"""Vera power gate: per-statement KL floors at the country token for several probes, from build's saved clean
post-norm states (data/clean/z_post.npy) and probe weights (npz: <name>__mu, __sd, __w, __b), with MY OWN
lm_head (tied embedding read from safetensors), MY OWN alarm line and MY OWN subset.

Line: PREREG s5 rule on the clean model: 2nd-highest score of the 200 line-setting (split 'refit') true statements. Caught = eval false with score > line. D_i = s_i - line.
quad (full vocab fp64 F_i, eps = rel*lam_max, rel in {1e-10,1e-8,1e-6}); cvx (free state, convex); sph (RMSNorm image).
Optimisation runs on the top-K words of p0 (K=32768); the found move is then scored with the FULL vocabulary.
Usage: vera_gate.py PROBES.npz name1,name2,... N_QUAD N_OPT OUT.json
"""
import sys, json, time, numpy as np, torch
from vera_floor import load_head, fisher, kl, opt_floors
torch.set_num_threads(4)
probes_f, names, NQ, NO, OUT = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
S = json.load(open("data/ref/split.json")); Z = np.load("data/clean/z_post.npy").astype(np.float64)
P = np.load(probes_f)
ev = np.array([e["split"] == "eval" for e in S]); ls = np.array([e["split"] == "refit" for e in S]); tr = np.array([e["true"] == 1 for e in S])
W, g = load_head()
A = {}
for nm in names:
    a = P[nm + "__w"].astype(np.float64) / P[nm + "__sd"].astype(np.float64)
    s = (Z - P[nm + "__mu"]) / P[nm + "__sd"] @ P[nm + "__w"].astype(np.float64) + float(P[nm + "__b"])
    st = np.sort(s[ls & tr])[::-1]; line = st[1]   # PREREG s5: 2nd-highest of the 200 line-setting (refit) true
    caught = np.where(ev & ~tr & (s > line))[0]
    A[nm] = dict(a=a, s=s, line=float(line), caught=caught)
    print(nm, "line", round(line, 4), "eval catch", len(caught), "/", int((ev & ~tr).sum()), flush=True)
rng = np.random.default_rng(1024)
union = np.unique(np.concatenate([A[n]["caught"] for n in names]))
sub = rng.permutation(union)[:NQ]
res = []
t0 = time.time()
for j, i in enumerate(sub):
    z0 = torch.tensor(Z[i]); F, p = fisher(W, z0)
    evals, Q = torch.linalg.eigh(F)
    row = dict(i=int(i), maxp=float(p.max()), lam_max=float(evals.max()))
    # top-K words for the optimisation
    K = 32768; top = torch.topk(p, K).indices; WK = W[top]
    for nm in names:
        if i not in set(A[nm]["caught"].tolist()):
            continue
        a = torch.tensor(A[nm]["a"]); D = float(A[nm]["s"][i] - A[nm]["line"])
        c = Q.T @ a
        r = dict(D=D, a_norm=float(a.norm()))
        for rel in (1e-10, 1e-8, 1e-6):
            lam = evals.clamp_min(0) + rel * evals.max()
            r[f"quad_{rel:g}"] = D * D / (2 * float((c * c / lam).sum()))
        # share of the probe direction on eigvals below 1e-6 * lam_max (per statement)
        r["flat_share_1e-6"] = float((c[evals < 1e-6 * evals.max()] ** 2).sum() / (c ** 2).sum())
        if j < NO:
            r.update(opt_floors(WK, W, g, z0, a, D, p))
        row[nm] = r
    res.append(row)
    print(j, i, {nm: {k: (round(v, 7) if isinstance(v, float) else v) for k, v in row[nm].items() if k in ("D", "quad_1e-08", "cvx", "sph")} for nm in names if nm in row}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(lines={n: A[n]["line"] for n in names}, n_caught={n: len(A[n]["caught"]) for n in names}, rows=res), open(OUT, "w"), indent=1)
