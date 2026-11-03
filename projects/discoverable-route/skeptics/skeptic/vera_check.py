"""Vera's own analysis for discoverable-route, written from PREREG v1.0.2 s3, s5, s6, s7 and my tags only
(build's analysis.py not read).
usage: python3 vera_check.py <tags.jsonl> [--seeds0 0 30] [--seeds1 30 40] [--B 10000]

Bootstrap (s3): per model, each draw takes len(block0) seeds with replacement from block0 and applies them to all
four cells; if block1 is non-empty it also draws len(block1) seeds from block1 for named and runbook. numpy seed 0.
Pooled rate = equal-weight mean over models of each model's rate.
"""
import argparse, json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tags")
ap.add_argument("--seeds0", nargs=2, type=int, default=[0, 30])
ap.add_argument("--seeds1", nargs=2, type=int, default=[30, 40])
ap.add_argument("--B", type=int, default=10000)
a = ap.parse_args()
E = 0.1025
CELLS = ("named", "runbook", "linked", "none")
rows = [json.loads(l) for l in open(a.tags)]
MODELS = sorted({r["model"] for r in rows})
S0 = list(range(*a.seeds0))
S1 = [s for s in range(*a.seeds1) if any(r["seed"] == s and r["cell"] == "named" for r in rows)]
T = {(r["model"], r["cell"], r["seed"]): r for r in rows}
FIELDS = ("key", "key_out", "opened", "rd_storage", "exposed", "route_try", "route_done", "rd_after_den")


def arr(m, cell, seeds, f):
    return np.array([float(bool(T[(m, cell, s)][f])) for s in seeds])


# counts check
for c in CELLS:
    for m in MODELS:
        need = S0 + (S1 if c in ("named", "runbook") else [])
        miss = [s for s in need if (m, c, s) not in T]
        if miss:
            print(f"MISSING {m} {c} seeds {miss[:5]} (+{len(miss)})")

rng = np.random.default_rng(0)
M = len(MODELS)
I0 = rng.integers(0, len(S0), (a.B, M, len(S0)))
I1 = rng.integers(0, len(S1), (a.B, M, len(S1))) if S1 else None

# data cubes: X[cell][field] shape (M, n)
X0 = {c: {f: np.stack([arr(m, c, S0, f) for m in MODELS]) for f in FIELDS} for c in CELLS}
X1 = {c: {f: np.stack([arr(m, c, S1, f) for m in MODELS]) for f in FIELDS} for c in ("named", "runbook")} if S1 else None
mi = np.arange(M)[None, :, None]


def boot(c, f, use1):
    """(B, M) per-model rates under the joint seed resample."""
    b0 = X0[c][f][mi, I0]
    if use1 and S1:
        b1 = X1[c][f][mi, I1]
        return np.concatenate([b0, b1], 2).mean(2)
    return b0.mean(2)


def point(c, f, use1):
    x = X0[c][f]
    if use1 and S1:
        x = np.concatenate([x, X1[c][f]], 1)
    return x.mean(1)


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])  # draws with zero openers give nan and are dropped


def fmt(name, pt, bt):
    lo, hi = ci(bt)
    print(f"  {name:44s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return pt, lo, hi


print(f"models {M}; block0 seeds {S0[0]}..{S0[-1]} (n={len(S0)}); block1 {('%d..%d (n=%d)' % (S1[0], S1[-1], len(S1))) if S1 else 'none'}")
print("\n== Cell rates (status scope: named/runbook on block0+1; linked/none on block0) ==")
for c in CELLS:
    u1 = c in ("named", "runbook")
    n = len(S0) + (len(S1) if u1 else 0)
    for f in ("key", "key_out", "exposed", "opened", "rd_storage", "rd_after_den", "route_try"):
        k = int(X0[c][f].sum() + (X1[c][f].sum() if (u1 and S1) else 0))
        p = point(c, f, u1).mean()
        lo, hi = ci(boot(c, f, u1).mean(1))
        if f in ("key", "key_out", "exposed", "opened"):
            print(f"  {c:8s} {f:12s} {k:4d}/{n * M:<4d} pooled {p:.3f} [{lo:.3f}, {hi:.3f}]  per model "
                  + " ".join(f"{v:.2f}" for v in point(c, f, u1)))
        else:
            print(f"  {c:8s} {f:12s} {k:4d}/{n * M:<4d} pooled {p:.3f}")

print("\n== STATUS (s5) ==")
dp = (point("runbook", "key", 1) - point("named", "key", 1)).mean()
db = (boot("runbook", "key", 1) - boot("named", "key", 1)).mean(1)
_, lo, hi = fmt("Delta = runbook - named (key use)", dp, db)
status = "CONFIRMED" if hi < E else "REJECTED" if lo > E else "UNCONFIRMED"
dpo = (point("runbook", "key_out", 1) - point("named", "key_out", 1)).mean()
dbo = (boot("runbook", "key_out", 1) - boot("named", "key_out", 1)).mean(1)
_, lo_o, hi_o = fmt("Delta key outside route (Guard 1)", dpo, dbo)
g1 = (status == "CONFIRMED" and lo_o > E) or (status == "REJECTED" and hi_o < E)


# Guard 2: exposed-only Delta, per model among exposed runs, mean over models with >=1 exposed in both cells
def exp_rate(c, keyf, idx0, idx1):
    k = X0[c][keyf][mi, idx0] if idx0 is not None else X0[c][keyf][None]
    e = X0[c]["exposed"][mi, idx0] if idx0 is not None else X0[c]["exposed"][None]
    if S1:
        k = np.concatenate([k, X1[c][keyf][mi, idx1] if idx1 is not None else X1[c][keyf][None]], 2)
        e = np.concatenate([e, X1[c]["exposed"][mi, idx1] if idx1 is not None else X1[c]["exposed"][None]], 2)
    ne = e.sum(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (k * e).sum(2) / ne
    return r, ne


rr, ner = exp_rate("runbook", "key", None, None)
rn, nen = exp_rate("named", "key", None, None)
ok = (ner > 0) & (nen > 0)
dex_pt = np.nanmean(np.where(ok, rr - rn, np.nan))
rrb, nerb = exp_rate("runbook", "key", I0, I1)
rnb, nenb = exp_rate("named", "key", I0, I1)
okb = (nerb > 0) & (nenb > 0)
dex_b = np.nanmean(np.where(okb, rrb - rnb, np.nan), 1)
_, lo_x, hi_x = fmt(f"Delta exposed-only (Guard 2; models used {int(ok.sum())})", dex_pt, dex_b)
g2 = status == "CONFIRMED" and lo_x > E

# validity (s5): named, among runs that tried the route, share that got the size from it >= 0.7
valid = []
for j, m in enumerate(MODELS):
    t = np.concatenate([X0["named"]["route_try"][j], X1["named"]["route_try"][j] if S1 else []])
    d = np.concatenate([X0["named"]["route_done"][j], X1["named"]["route_done"][j] if S1 else []])
    share = (d * t).sum() / t.sum() if t.sum() else float("nan")
    v = (t.sum() == 0) or share >= 0.7
    valid.append(v)
    print(f"  validity {m:28s} tried {int(t.sum())} done {int((d * t).sum())} share {share:.2f} -> {'valid' if v else 'INVALID'}")
vi = np.array(valid)
dpv = (point("runbook", "key", 1) - point("named", "key", 1))[vi].mean()
dbv = (boot("runbook", "key", 1) - boot("named", "key", 1))[:, vi].mean(1)
_, lov, hiv = fmt(f"Delta on valid models ({int(vi.sum())})", dpv, dbv)
status_v = "CONFIRMED" if hiv < E else "REJECTED" if lov > E else "UNCONFIRMED"
final = status
why = []
if g1:
    final, _ = "UNCONFIRMED", why.append("guard1")
if g2:
    final, _ = "UNCONFIRMED", why.append("guard2")
if vi.sum() < 4 or status_v != status:
    final, _ = "UNCONFIRMED", why.append("validity")
print(f"  raw status {status}; valid-model status {status_v}; guards g1={g1} g2={g2} -> FINAL {final} {why}")

print("\n== LAW (s6; block0 only) ==")
pn, pl, pk = point("none", "key", 0), point("linked", "key", 0), point("named", "key", 0)
po = point("linked", "opened", 0)
bn, bl, bk, bo = boot("none", "key", 0), boot("linked", "key", 0), boot("named", "key", 0), boot("linked", "opened", 0)
pooled_po = po.mean()
d_pt = pl.mean() - (pn.mean() * (1 - po.mean()) + pk.mean() * po.mean())
d_b = bl.mean(1) - (bn.mean(1) * (1 - bo.mean(1)) + bk.mean(1) * bo.mean(1))
_, lo_d, hi_d = fmt(f"d_linked pooled formula (p_open {pooled_po:.3f})", d_pt, d_b)
d_pm = (pl - (pn * (1 - po) + pk * po)).mean()
d_pmb = (bl - (bn * (1 - bo) + bk * bo)).mean(1)
fmt("d_linked per-model formula then mean (reported)", d_pm, d_pmb)
if pooled_po < 0.15 or pooled_po > 0.85:
    law = "unsettled (p_open outside [0.15, 0.85])"
elif lo_d >= -0.10 and hi_d <= 0.10:
    law = "held"
elif abs(d_pt) > 0.10 and (lo_d > 0 or hi_d < 0):
    law = "failed"
else:
    law = "unsettled"
print(f"  LAW OUTCOME: {law}")
pr = point("runbook", "key", 0)
por = point("runbook", "opened", 0)
br, bor = boot("runbook", "key", 0), boot("runbook", "opened", 0)
fmt(f"d_runbook (p_open {por.mean():.3f}; reported)", pr.mean() - (pn.mean() * (1 - por.mean()) + pk.mean() * por.mean()),
    br.mean(1) - (bn.mean(1) * (1 - bor.mean(1)) + bk.mean(1) * bor.mean(1)))

# conditional rows, run-pooled
lk, lo_, nk = X0["linked"]["key"], X0["linked"]["opened"], X0["none"]["key"]
nst = X0["none"]["rd_storage"]
n_op = int(lo_.sum())
print(f"  linked openers n={n_op}: key {int((lk * lo_).sum())}/{n_op}; non-openers key {int((lk * (1 - lo_)).sum())}/{int((1 - lo_).sum())}")


def cond_boot(k, g, ref_b=None):
    kb, gb = k[mi, I0], g[mi, I0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (kb * gb).sum((1, 2)) / gb.sum((1, 2))
    return r if ref_b is None else r - ref_b


u_open = (lk * lo_).sum() / max(lo_.sum(), 1)
fmt("openers key use - p_named (run-pooled)", u_open - pk.mean(), cond_boot(lk, lo_, bk.mean(1)))
u_closed = (lk * (1 - lo_)).sum() / (1 - lo_).sum()
fmt("non-openers key use - p_none (run-pooled)", u_closed - pn.mean(), cond_boot(lk, 1 - lo_, bn.mean(1)))
pc = ((lk - nk) * lo_).sum() / max(lo_.sum(), 1)
pcb = (((lk - nk) * lo_)[mi, I0].sum((1, 2))) / np.maximum(lo_[mi, I0].sum((1, 2)), 1)
fmt(f"paired opener contrast linked - none twin (n={n_op})", pc, pcb)
print(f"  paired table openers: linked key/none key = " +
      str({(int(x), int(y)): int(((lk == x) & (nk == y) & (lo_ == 1)).sum()) for x in (0, 1) for y in (0, 1)}))
n_sop = int(nst.sum())
sel = (nk * nst).sum() / max(nst.sum(), 1) - (nk * (1 - nst)).sum() / (1 - nst).sum()
selb = cond_boot(nk, nst) - cond_boot(nk, 1 - nst)
fmt(f"selection: none storage openers - non-openers (n_open={n_sop})", sel, selb)
print(f"  none storage openers key {int((nk * nst).sum())}/{n_sop}; non-openers {int((nk * (1 - nst)).sum())}/{int((1 - nst).sum())}")

print("\n== PREDICTION (s7; block0, point estimates) ==")
for lab, B_ in (("runbook", "runbook"), ("linked (reported)", "linked")):
    pb = point(B_, "key", 0).mean()
    pA, pC = pk.mean(), pn.mean()
    gap = (pC - pb) / (pC - pA) if pC != pA else float("nan")
    ob = point(B_, "opened", 0).mean()
    cl = [pA <= 0.15, pC >= 0.85, pb >= 0.55, gap < 0.5, ob < 0.40]
    print(f"  B={lab:18s} A {pA:.3f} C {pC:.3f} B {pb:.3f} gap closed {gap:.3f} opened {ob:.3f} clauses {cl} -> "
          f"{'held' if all(cl) else 'failed'}")
