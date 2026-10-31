"""Sensitivity of the frozen status to each pre-data choice + factor nulls (own code, raw JSONL via vera_check.score).
Exact run-level bootstrap via convolution (D = mean_m(uF - lR) exactly), so no RNG enters the status."""
import json, sys, numpy as np
from scipy import stats
from vera_check import load, STRONG, CELLS, effects
import os
PROJ = next(p for p in ("../..", "../../kran-research/projects/decoy-usefulness") if os.path.exists(os.path.join(p, "env.py")))
RAW = os.path.join(PROJ, "raw")
rows = load(RAW); by = {(r["model"], r["arm"], r["seed"]): r for r in rows}
S = list(range(20))
def counts(models, field="use", keep=lambda r: True):
    return {m: {c: [by[(m, c, s)][field] for s in S if keep(by[(m, c, s)])] for c in CELLS} for m in models}
def pmf(k, n):
    p = k / n; x = np.arange(n + 1)
    o = np.exp(stats.binom.logpmf(x, n, p)) if 0 < p < 1 else (x == k).astype(float)
    return o / o.sum()
def exact_ci(cnt):
    # D = (1/K) sum_m (cF_m/nF_m - cL_m/nL_m); exact only when all n equal -> use common grid when equal, else MC
    K = len(cnt); ns = {len(cnt[m][c]) for m in cnt for c in ("useful_foreign", "useless_real")}
    if len(ns) == 1:
        n = ns.pop(); dist = np.ones(1)
        for m in cnt:
            f = pmf(sum(cnt[m]["useful_foreign"]), n); l = pmf(sum(cnt[m]["useless_real"]), n)
            dist = np.convolve(dist, np.convolve(f, l[::-1]))
        vals = (np.arange(len(dist)) - K * n) / (n * K); cdf = np.cumsum(dist)
        q = lambda p: float(vals[np.searchsorted(cdf, p - 1e-12)])
        return q(.025), q(.975), "exact"
    rng = np.random.default_rng(0); B = 20000
    Db = np.zeros(B)
    for m in cnt:
        a = np.array(cnt[m]["useful_foreign"]); b = np.array(cnt[m]["useless_real"])
        Db += (rng.binomial(len(a), a.mean(), B) / len(a) - rng.binomial(len(b), b.mean(), B) / len(b)) / K
    return float(np.percentile(Db, 2.5)), float(np.percentile(Db, 97.5)), "mc"
def status(lo, hi, D_m, closed=True):
    inside = (lo >= -.15 and hi <= .15) if closed else (lo > -.15 and hi < .15)
    if hi < 0 or inside: return "REJECTED"
    if lo > 0 and sum(d > 0 for d in D_m) >= 4: return "CONFIRMED"
    return "UNCONFIRMED"
def row(name, models, field="use", keep=lambda r: True):
    cnt = counts(models, field, keep)
    D_m = [np.mean(cnt[m]["useful_foreign"]) - np.mean(cnt[m]["useless_real"]) for m in models]
    D = float(np.mean(D_m)); lo, hi, how = exact_ci(cnt)
    h = stats.t.ppf(.975, len(models) - 1) * np.std(D_m, ddof=1) / np.sqrt(len(models))
    print(f"{name:42s} D {D:+.3f} CI95 [{lo:+.3f}, {hi:+.3f}] ({how})  t [{D-h:+.3f}, {D+h:+.3f}]  k+ {sum(d>0 for d in D_m)}/{len(models)}"
          f"  closed: {status(lo, hi, D_m)}  open: {status(lo, hi, D_m, False)}")
row("PRIMARY (PREREG s3 any tool)", STRONG)
row("credential-only (D2)", STRONG, "use_cred")
row("identity_seen runs only (s11)", STRONG, keep=lambda r: r["ident"] == 1)
for m in STRONG:
    row(f"leave out {m}", [x for x in STRONG if x != m])
# factor nulls: within (model, seed), flip the fit labels (uR<->lR, uF<->lF) or the identity labels (uR<->uF, lR<->lF)
Y = np.array([[[by[(m, c, s)]["use"] for s in S] for c in CELLS] for m in STRONG], float)
rng = np.random.default_rng(1); R = 100000
u_seed = (Y[:, 0] + Y[:, 1] - Y[:, 2] - Y[:, 3]) / 2; r_seed = (Y[:, 0] + Y[:, 2] - Y[:, 1] - Y[:, 3]) / 2
for name, x in (("usefulness U", u_seed), ("realism R", r_seed), ("D = uF - lR", Y[:, 1] - Y[:, 2])):
    obs = x.mean(); fl = rng.choice([-1, 1], size=(R,) + x.shape); nul = (fl * x).mean((1, 2))
    print(f"label-flip null {name:14s} obs {obs:+.3f}  null 95% [{np.percentile(nul,2.5):+.3f}, {np.percentile(nul,97.5):+.3f}]  p two-sided {(np.abs(nul) >= abs(obs)-1e-12).mean():.4f}")
# blocking effect (not blocked vs useful+real, same record)
nb = np.array([[by[(m, "not_blocked", s)]["use"] for s in S] for m in STRONG], float)
print(f"blocked useful+real {Y[:,0].mean():.3f} vs not blocked {nb.mean():.3f} ({int(nb.sum())}/{nb.size})")
for var in ("useless_real_bucket", "useless_real_explicit"):
    v = np.array([[by[(m, var, s)]["use"] for s in S] for m in STRONG], float)
    d = v - Y[:, 2]
    print(f"variant {var:24s} rate {v.mean():.3f} vs useless+real {Y[:,2].mean():.3f}; per model {v.mean(1).round(2).tolist()}; "
          f"paired seed-flip p {((np.abs((rng.choice([-1,1],size=(R,)+d.shape)*d).mean((1,2))) >= abs(d.mean())-1e-12).mean()):.4f}")

# D7 (exploratory, after main results): explicit scope, foreign identity
fe = np.array([[by[(m, "useless_foreign_explicit", s)]["use"] for s in S] for m in STRONG], float)
re_ = np.array([[by[(m, "useless_real_explicit", s)]["use"] for s in S] for m in STRONG], float)
d = re_ - fe
print(f"D7 useless_foreign_explicit rate {fe.mean():.3f}; per model {fe.mean(1).round(2).tolist()}; "
      f"explicit realism (real - foreign) {d.mean():+.3f}, paired seed-flip p "
      f"{((np.abs((rng.choice([-1,1],size=(R,)+d.shape)*d).mean((1,2))) >= abs(d.mean())-1e-12).mean()):.4f}")
dF = Y[:, 3] - fe
print(f"explicit vs mild scope, foreign identity: useless+foreign {Y[:,3].mean():.3f} -> explicit {fe.mean():.3f} (diff {(-dF).mean():+.3f})")
