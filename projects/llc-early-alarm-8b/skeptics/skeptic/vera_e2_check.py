"""Vera: rebuild E2 (8B tail, k2 s5, u32 and u88, c in {0.033, 0.1, 0.3}, gamma 1) from the raw per-draw losses,
and test whether the 8B SGLD walk is a free random walk (noise only).
(1) lambda-hat = n beta * (mean of draws 26-100 over both chains - L(w*)), batch-means SE 5x30; diff vs stored.
(2) scaling with c: lambda-hat / c (constant if the readout is set by the step size).
(3) noise-only distance: a pure Gaussian walk with step variance eps per coordinate has |w_t - w*| ~ sqrt(eps t d),
    d = 2,424,832. Compare with the stored dist_from_wstar at t = 25, 50, 75, 100.
(4) difference u88 - u32 at each c, and its ratio across c."""
import json, glob, math, os, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "/Users/esben/work/theory-1124-llc-early-alarm-8b/kran-research/projects/llc-early-alarm-8b/data/llc/e2/"
d_params = 2424832
rows = {}
for f in sorted(glob.glob(os.path.join(D, "m8_*_c*.json"))):
    r = json.load(open(f))
    L = np.array(r["losses"], float)
    if L.shape[1] == r["burn"] + r["draws"]:
        L = L[:, r["burn"]:]
    d = L.reshape(-1)
    lam = r["nbeta"] * (d.mean() - r["L_wstar_all_rows"])
    bm = np.array([r["nbeta"] * (b.mean() - r["L_wstar_all_rows"]) for b in d.reshape(5, -1)])
    se = bm.std(ddof=1) / math.sqrt(5)
    ts = [25, 50, 75, 100] if len(r["dist_from_wstar"][0]) == 4 else None
    walk = [math.sqrt(r["eps"] * t * d_params) for t in ts] if ts else None
    rows[(r["step"], r["c"])] = (lam, se)
    print(f"u{r['step']:<3} c={r['c']:<6} lambda-hat {lam:.4f} (stored {r['lambda_hat']:.4f}, diff {lam - r['lambda_hat']:+.1e})"
          f"  SE {se:.4f} (stored {r['bm_se']:.4f})  lambda/c {lam / r['c']:.2f}")
    if walk:
        for ch, dist in enumerate(r["dist_from_wstar"]):
            print(f"      chain {ch}: |w - w*| at t=25/50/75/100 {[round(x, 2) for x in dist]}  vs pure-noise sqrt(eps t d) {[round(x, 2) for x in walk]}"
                  f"  ratio {[round(a / b, 4) for a, b in zip(dist, walk)]}")
steps = sorted({k[0] for k in rows}); cs = sorted({k[1] for k in rows})
if len(steps) == 2:
    for c in cs:
        if (steps[0], c) in rows and (steps[1], c) in rows:
            a, b = rows[(steps[1], c)], rows[(steps[0], c)]
            diff = a[0] - b[0]; se = math.hypot(a[1], b[1])
            print(f"c={c}: u{steps[1]} - u{steps[0]} = {diff:+.4f} (SE {se:.4f}, z {diff / se:+.1f}); diff/c {diff / c:+.2f}")
