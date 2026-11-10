"""Vera: PREREG s9.1 two-form comparison on ALL balanced random-direction cues (u seed 1110):
frozen sweep k=0..5 (both seeds) + extension k=6,7 (bf16 cast) + no-cast k=5 and k=8 (fp32 cue, matched x).
Forms, all fitted on log T by least squares (equal weight per run):
  F1  log T = a + b log x                 (free power law, 2 params)
  F2  log T = a - 2 log x                 (fixed -2, 1 param)
  F3  log T = a - 1 log x                 (fixed -1, 1 param)
  F4  log T = a - log x + log(log(K/x))   (Saxe-type x^-1 ln(K/x), params a, K > max x)
Reports RSS, AIC = n log(RSS/n) + 2p, and local slopes along the curve (median T per x).
"""
import json, math, glob, os, numpy as np
from scipy import optimize
import vera_check as V

runs = V.load()
ext_x = json.load(open(f"{V.P}/data/scan_ext.json")) if os.path.exists(f"{V.P}/data/scan_ext.json") else {}


def x_of(r):
    if V.is_sweep(r):
        return V.XS[r["alpha"]]
    d = json.load(open(f"{V.P}/data/runs/{r['name']}.json"))
    cs = d.get("cue_size", {})
    for k in ("x", "achieved_x", "x_achieved", "x_matched"):
        if k in cs:
            return cs[k]
    return None


sel = [r for r in runs if (V.is_sweep(r) or r["name"].startswith(("ext_", "nocast"))) and r["T"] is not None
       and r["u_file"] == "data/cue_u.npy" and r["cue_dir"] == "rand"]
pts = []
for r in sel:
    x = x_of(r)
    if x is None:
        print("no x for", r["name"], json.load(open(f"{V.P}/data/runs/{r['name']}.json")).get("cue_size")); continue
    pts.append((r["name"], x, r["T"]))
pts.sort(key=lambda p: -p[1])
for p in pts: print(f"{p[0]:28s} x {p[1]:9.4f}  T {p[2]}")
lx = np.log([p[1] for p in pts]); ly = np.log([p[2] for p in pts]); n = len(pts)

res = {}
b, a = np.polyfit(lx, ly, 1); rss = ((ly - (a + b * lx)) ** 2).sum(); res["F1 free"] = (rss, 2, dict(a=a, b=b))
for name, s in (("F2 slope -2", -2.0), ("F3 slope -1", -1.0)):
    a = (ly - s * lx).mean(); rss = ((ly - a - s * lx) ** 2).sum(); res[name] = (rss, 1, dict(a=a))
def f4(p):
    a, lk = p
    K = math.exp(lk)
    if K <= math.exp(lx.max()) * 1.0001: return 1e9
    return ((ly - (a - lx + np.log(np.log(K / np.exp(lx))))) ** 2).sum()
best = min((optimize.minimize(f4, [ly.mean(), lk0], method="Nelder-Mead") for lk0 in np.linspace(lx.max() + 0.1, lx.max() + 10, 25)), key=lambda r: r.fun)
res["F4 x^-1 ln(K/x)"] = (best.fun, 2, dict(a=best.x[0], K=math.exp(best.x[1])))
out = {}
for k, (rss, p, par) in res.items():
    aic = n * math.log(rss / n) + 2 * p
    out[k] = dict(rss=rss, k=p, aic=aic, params=par)
    print(f"{k:18s} RSS {rss:7.3f}  AIC {aic:7.2f}  {par}")
# local slopes on the median T per distinct x
from collections import defaultdict
g = defaultdict(list)
for _, x, T in pts: g[round(x, 4)].append(T)
xs = sorted(g, reverse=True); med = [float(np.median(g[x])) for x in xs]
loc = [math.log(med[i + 1] / med[i]) / math.log(xs[i + 1] / xs[i]) for i in range(len(xs) - 1)]
print("x:", xs); print("median T:", med); print("local slopes:", [round(v, 2) for v in loc])
out["local"] = dict(x=xs, T=med, slopes=loc); out["points"] = pts
json.dump(out, open(os.path.join(V.OUT, "vera_forms.json"), "w"), indent=1)
