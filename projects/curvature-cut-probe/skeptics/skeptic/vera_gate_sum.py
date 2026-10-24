"""Summarise Vera's gate json(s): per-probe floors and ratios to the full probe.
Per statement (or pair) values; aggregate A = mean over rows; aggregate D4 = n_caught/1192 * mean of the cheapest
(1 - 119/n_caught) fraction (Felix D4: sum of the cheapest n_caught-119 per-statement floors / 1192, estimated from the
subsample). Paired ratio = median over rows where both probes caught the false statement. Bootstrap CI over rows (2000).
"""
import sys, json, numpy as np
rows = []; ncaught = None
for f in sys.argv[1:]:
    r = json.load(open(f)); rows += r["rows"]; ncaught = r["n_caught"]
names = [n for n in ncaught]
rng = np.random.default_rng(0)
out = {}
for key in ("quadF_1e-08", "quadP_1e-08", "quadF_1e-10", "quadF_1e-06", "cvxF", "sphF", "sphP"):
    vals = {n: np.array([x[n][key] for x in rows if n in x and key in x[n]]) for n in names}
    if all(len(v) == 0 for v in vals.values()):
        continue
    line = [key]
    agg = {}
    for n in names:
        v = vals[n]
        if len(v) == 0:
            continue
        nc = ncaught[n]; keep = max(1, int(round(len(v) * (1 - 119 / nc))))
        d4 = nc / 1192 * np.sort(v)[:keep].mean()
        agg[n] = dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), d4=float(d4))
    for n in names:
        if n == "unproj" or n not in agg:
            continue
        both = [(x[n][key], x["unproj"][key]) for x in rows if n in x and "unproj" in x and key in x[n] and key in x["unproj"]]
        pr = np.array([a / b for a, b in both])
        boot = [np.mean(rng.choice(vals[n], len(vals[n]))) / np.mean(rng.choice(vals["unproj"], len(vals["unproj"]))) for _ in range(2000)]
        agg[n]["ratio_mean"] = agg[n]["mean"] / agg["unproj"]["mean"]
        agg[n]["ratio_mean_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        agg[n]["ratio_d4"] = agg[n]["d4"] / agg["unproj"]["d4"]
        agg[n]["ratio_paired_median"] = float(np.median(pr)) if len(pr) else None
        agg[n]["n_paired"] = len(pr)
    out[key] = agg
    print(key)
    for n, a in agg.items():
        print("  %-10s n=%2d mean %.3g median %.3g D4 %.3g" % (n, a["n"], a["mean"], a["median"], a["d4"]),
              ("ratio mean %.3g [%.3g, %.3g]  D4 %.3g  paired median %.3g (n=%d)" % (a["ratio_mean"], *a["ratio_mean_ci95"], a["ratio_d4"], a["ratio_paired_median"], a["n_paired"])) if "ratio_mean" in a else "")
json.dump(out, open("skeptic/vera_gate_summary.json", "w"), indent=1)
