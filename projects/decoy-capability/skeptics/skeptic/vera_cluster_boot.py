"""Cluster check (PREREG s11) and two-level bootstrap (original s6) on Vera's own rescore."""
import json, numpy as np
from scipy import stats
from vera_stats import per_model, summary, boot, status
R0 = json.load(open("vera_rescore_263af97.json")); rows, res = R0["rows"], R0["result"]; fm = res["fitted"]; per = res["per_model"]
x = np.array([per[m]["x"] for m in fm]); us = np.array([per[m]["u_static"] for m in fm]); uf = np.array([per[m]["u_stateful"] for m in fm])
o = np.argsort(x); xs = x[o]; gaps = np.diff(xs); g = gaps.argmax()
print("sorted x", dict(zip([fm[i] for i in o], xs.round(3))), "max gap", gaps[g].round(3), "between", xs[g].round(3), xs[g+1].round(3))
hi = o[g+1:]; Rfull = x.max()-x.min()
def tci(xx, y, R, l=.9):
    r = stats.linregress(xx, y); h = stats.t.ppf(1-(1-l)/2, len(xx)-2)*r.stderr; return round(r.slope*R, 3), [round((r.slope-h)*R, 3), round((r.slope+h)*R, 3)]
for lab, R in (("range of group", x[hi].max()-x[hi].min()), ("full range", Rfull)):
    print("within high group (n=%d), scaled to %s:" % (len(hi), lab), "I", tci(x[hi], us[hi]-uf[hi], R), "static", tci(x[hi], us[hi], R), "stateful", tci(x[hi], uf[hi], R))
lo = o[:g+1]
print("group means: low", dict(static=us[lo].mean().round(3), stateful=uf[lo].mean().round(3)), "high", dict(static=us[hi].mean().round(3), stateful=uf[hi].mean().round(3)))
eps = [dict(model=r["model"], arm=r["arm"], seed=r["seed"], touch=r["touch"], use=r["use"], score=r["score"]) for r in rows
       if r["model"] in fm and ((r["arm"] in ("static", "stateful") and r["seed"] < 30) or (r["arm"] == "nobait" and 100 <= r["seed"] <= 114))]
pm = per_model(eps, fm); s = summary(pm, fm); b = boot(pm, fm, s["R"], B=10000, seed=0)
print("two-level bootstrap (Vera's own, B=10000):", {k: (np.round(v, 3).tolist() if hasattr(v, "tolist") else v) for k, v in b.items()}, "original-s6 status", status(s, b))
