"""Iris: fits on toy_real.json. OLS of log T on log alpha over uncensored cue runs (all seeds), per optimiser;
local slopes between neighbouring alphas (median T); plateau gap of censored runs vs alpha (last curve point)."""
import json, math
import numpy as np
d = json.load(open("toy_real.json"))
out = {}
for opt in ("adamw", "sgd"):
    rr = [o for o in d["runs"] if o["opt"] == opt and o["arm"] == "cue" and o["alpha"] > 0]
    unc = [o for o in rr if not o["censored"]]
    x = np.log([o["alpha"] for o in unc]); y = np.log([o["T"] for o in unc])
    X = np.c_[np.ones_like(x), x]; b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = ((y - X @ b) ** 2).sum() / (len(y) - 2); se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    al = sorted({o["alpha"] for o in unc})
    med = {a: float(np.median([o["T"] for o in unc if o["alpha"] == a])) for a in al}
    loc = [(round(math.sqrt(a1 * a2), 3), round(math.log(med[a2] / med[a1]) / math.log(a2 / a1), 2)) for a1, a2 in zip(al, al[1:])]
    # bottom half / top half slopes
    def ols(sub):
        xx = np.log([o["alpha"] for o in sub]); yy = np.log([o["T"] for o in sub])
        return round(float(np.polyfit(xx, yy, 1)[0]), 2)
    half = len(al) // 2
    bot = [o for o in unc if o["alpha"] <= al[half]]; top = [o for o in unc if o["alpha"] >= al[half]]
    cen = [o for o in rr if o["censored"]]
    plat = {}
    for a in sorted({o["alpha"] for o in cen}):
        plat[round(a, 4)] = round(float(np.mean([o["curve"][-1][1] for o in cen if o["alpha"] == a])), 4)
    pa = np.array(list(plat.keys())); pg = np.array(list(plat.values()))
    m = pg > 0.005
    pslope = round(float(np.polyfit(np.log(pa[m]), np.log(pg[m]), 1)[0]), 2) if m.sum() >= 2 else None
    out[opt] = dict(slope=round(float(b[1]), 3), se=round(se, 3), ci95=[round(b[1] - 1.96 * se, 3), round(b[1] + 1.96 * se, 3)],
                    n=len(unc), alpha_range=[round(al[0], 4), round(al[-1], 4)], median_T=med, local_slopes=loc,
                    slope_bottom_half=ols(bot), slope_top_half=ols(top), censored_alphas=sorted(plat),
                    plateau_gap=plat, plateau_gap_slope_vs_alpha=pslope)
json.dump(out, open("fit_toy.json", "w"), indent=1)
print(json.dumps(out, indent=1))
