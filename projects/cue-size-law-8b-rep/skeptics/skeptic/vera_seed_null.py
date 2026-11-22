"""Null test of the frozen seed-level clause on the final data (Vera, theory-1122).
Shuffle size labels within each seed (20,000 draws, numpy seed 0), recompute per-seed OLS slopes of ln T on
ln share, the t-interval over the 3 seeds (df 2), and count how often the interval lies inside [-1.4, -0.7]
(the CONFIRMED clause) or excludes 0. Input: my checker's output check_final.json (T rebuilt from raw curves)."""
import json, sys, numpy as np
from scipy import stats
c = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "check_final.json"))
runs = [r for r in c["runs"].values() if r["arm"] == "cue" and r["k"] is not None and r["alpha"] > 0]
seeds = sorted({r["seed"] for r in runs})
X, Y = {}, {}
for s in seeds:
    rs = sorted([r for r in runs if r["seed"] == s], key=lambda r: r["k"])
    X[s] = np.log([r["share_logged"] for r in rs]); Y[s] = np.log([r["T"] for r in rs])
def slope(x, y): x = x - x.mean(); return float((x * (y - y.mean())).sum() / (x * x).sum())
def ci(b):
    b = np.asarray(b); m = b.mean(); h = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b)); return m, m - h, m + h
obs = ci([slope(X[s], Y[s]) for s in seeds]); print("observed seed-level", obs)
rng = np.random.default_rng(0); N = 20000; inside = excl0 = 0
for _ in range(N):
    m, lo, hi = ci([slope(X[s], rng.permutation(Y[s])) for s in seeds])
    inside += (lo >= -1.4 and hi <= -0.7); excl0 += (hi < 0 or lo > 0)
print(f"null: interval inside [-1.4,-0.7] in {inside}/{N} = {inside/N:.5f}; excludes 0 in {excl0}/{N} = {excl0/N:.4f}")
# luck check: the same mean with a wider seed sd, at n=3 (t, df 2)
for name, sd in (("8B parent run theory-1116, 5 seeds (check_parent_l8.json)", 0.03143725491215378),
                 ("0.6B reference, 5 seeds (analysis.json matched_06b_ref)", 0.03953534184391683)):
    h = stats.t.ppf(0.975, 2) * sd / np.sqrt(3); print(f"with sd {sd:.4f} of the {name}: [{obs[0]-h:.4f}, {obs[0]+h:.4f}]")
