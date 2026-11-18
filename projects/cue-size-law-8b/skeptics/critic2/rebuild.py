"""critic2 (Claude fallback) own rebuild, theory-1118-cue-size-law-8b. Reads only data/*.json. json+numpy+scipy."""
import json, glob, os, sys
import numpy as np
from scipy import stats
P = sys.argv[1] if len(sys.argv) > 1 else "."
TH = 0.10
def load(f): return json.load(open(f))
def T_of(run):
    pts = {}
    for p in run.get("dev_curve", []): pts[int(p["step"])] = p
    for p in run.get("dense_curve") or []: pts.setdefault(int(p["step"]), p)
    g0 = run["gap0"]
    c = sorted((s, p["gap"] - g0, p["D"]) for s, p in pts.items() if s > 0)
    dmis = max((abs(a - b) for _, a, b in c), default=0.0)
    for i in range(len(c) - 1):
        if c[i][1] >= TH and c[i + 1][1] >= TH: return c[i][0], False, dmis, max(x[1] for x in c)
    return run.get("budget"), True, dmis, max((x[1] for x in c), default=0.0)
def slope(xs, ys):
    r = stats.linregress(np.log(xs), np.log(ys)); return r.slope
def tci(v):
    v = np.asarray(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)); h = stats.t.ppf(0.975, len(v) - 1) * se
    return m, m - h, m + h, se
out = {}
runs = {os.path.basename(f)[:-5]: load(f) for f in glob.glob(f"{P}/data/runs/*.json")}
r06 = {os.path.basename(f)[:-5]: load(f) for f in glob.glob(f"{P}/data/runs_06b/*.json")}
mism = 0; dmax = 0.0; cells = 0
for name, r in {**runs, **r06}.items():
    T, cen, dm, _ = T_of(r); dmax = max(dmax, dm)
    if "_c" in name: continue
    cells += 1
    if T != r["T_dense"]: mism += 1; print("T MISMATCH", name, T, r["T_dense"])
out["T_rederived_vs_T_dense"] = f"{cells - mism}/{cells} equal; max |D - (gap-gap0)| {dmax:.2e}"
def arm(pref, d, seeds):
    per = {}; Ts = {}
    for s in seeds:
        xs, ys = [], []
        for k in range(6):
            r = d[f"{pref(s)}_k{k}_s{s}"]; T, cen, _, _ = T_of(r)
            assert not cen; xs.append(r["share"]); ys.append(T); Ts.setdefault(k, []).append(T)
        per[s] = slope(xs, ys)
    return per, Ts
p8, T8 = arm(lambda s: "m8" if s < 3 else "l8", runs, range(5))
p6, T6 = arm(lambda s: "m06", r06, range(5))
m8 = tci(list(p8.values())); m6 = tci(list(p6.values()))
out["8B per-seed"] = [round(v, 4) for v in p8.values()]
out["8B seed-level"] = f"{m8[0]:.4f} [{m8[1]:.4f},{m8[2]:.4f}] -> x{2**-m8[0]:.3f} [{2**-m8[2]:.3f},{2**-m8[1]:.3f}]"
out["0.6B per-seed"] = [round(v, 4) for v in p6.values()]
out["0.6B seed-level"] = f"{m6[0]:.4f} [{m6[1]:.4f},{m6[2]:.4f}] -> x{2**-m6[0]:.3f} [{2**-m6[2]:.3f},{2**-m6[1]:.3f}]"
out["band check (-1.4,-0.7) 8B CI inside"] = bool(m8[1] > -1.4 and m8[2] < -0.7)
w = stats.ttest_ind(list(p8.values()), list(p6.values()), equal_var=False)
a, b = np.array(list(p8.values())), np.array(list(p6.values()))
va, vb = a.var(ddof=1) / 5, b.var(ddof=1) / 5; df = (va + vb) ** 2 / (va**2 / 4 + vb**2 / 4)
dd = a.mean() - b.mean(); h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
out["S1 Welch diff"] = f"{dd:.4f} [{dd-h:.4f},{dd+h:.4f}] df {df:.2f}"
out["medians 8B"] = [float(np.median(T8[k])) for k in range(6)]
out["medians 0.6B"] = [float(np.median(T6[k])) for k in range(6)]
xs, ys = [], []
for s in range(5):
    for k in range(6):
        r = runs[f"{'m8' if s < 3 else 'l8'}_k{k}_s{s}"]; xs.append(r["share"]); ys.append(T_of(r)[0])
pr = stats.linregress(np.log(xs), np.log(ys)); h = stats.t.ppf(0.975, len(xs) - 2) * pr.stderr
out["8B pooled OLS 30 runs"] = f"{pr.slope:.4f} [{pr.slope-h:.4f},{pr.slope+h:.4f}]"
s3 = np.array([p8[s] for s in range(3)]); h = stats.t.ppf(0.975, 2) * s3.std(ddof=1) * np.sqrt(1 + 1 / 3)
out["predictive range one new seed from seeds 0-2"] = f"[{s3.mean()-h:.4f},{s3.mean()+h:.4f}]; seeds 3,4 = {p8[3]:.4f},{p8[4]:.4f}"
for c in ("m8_c1_s0", "l8_c1_s0", "l8_c2_s0"):
    r = runs[c]; T, cen, _, mx = T_of(r)
    out[f"control {c}"] = f"switched={not cen}; steps_run {r.get('steps_run')}; max D {mx:.4f}"
eq = 0; tot = 0; pts = 0
for n, r in runs.items():
    if n.startswith("l8_") and int(n[-1]) < 3:
        m = runs.get("m8_" + n[3:])
        if m is None: continue
        tot += 1
        a1 = [(p["step"], p["D"]) for p in (r.get("dev_curve", []) + (r.get("dense_curve") or []))]
        b1 = [(p["step"], p["D"]) for p in (m.get("dev_curve", []) + (m.get("dense_curve") or []))]
        n1 = min(len(a1), len(b1)); pts += n1
        eq += int(a1[:n1] == b1[:n1] and T_of(r)[0] == T_of(m)[0])
out["mini vs laptop copies"] = f"{eq}/{tot} equal curves + T (over common eval prefix, {pts} points)"
g = load(f"{P}/data/gates.json")
out["gates keys"] = list(g.keys())
json.dump(out, open("/tmp/k2/rebuild.json", "w"), indent=1)
for k, v in out.items(): print(k, ":", v)
# --- table and S4 halves
tab = {}
for k in range(6):
    tab[k] = ([T_of(runs[f"{'m8' if s < 3 else 'l8'}_k{k}_s{s}"])[0] for s in range(5)], [T_of(r06[f"m06_k{k}_s{s}"])[0] for s in range(5)])
    print("table k", k, tab[k])
def halves(pref, d):
    lo, hi = [], []
    for s in range(5):
        rs = [d[f"{pref(s)}_k{k}_s{s}"] for k in range(6)]
        lo.append(slope([r["share"] for r in rs[:3]], [T_of(r)[0] for r in rs[:3]]))
        hi.append(slope([r["share"] for r in rs[3:]], [T_of(r)[0] for r in rs[3:]]))
    return np.mean(lo), np.mean(hi)
print("8B halves (largest3, smallest3) mean of per-seed:", [round(v, 3) for v in halves(lambda s: "m8" if s < 3 else "l8", runs)])
xs = np.log([runs[f"{'m8' if s < 3 else 'l8'}_k{k}_s{s}"]["share"] for s in range(5) for k in range(6)])
ys = np.log([T_of(runs[f"{'m8' if s < 3 else 'l8'}_k{k}_s{s}"])[0] for s in range(5) for k in range(6)])
x3 = xs[:, None] ** np.arange(3); co, res, *_ = np.linalg.lstsq(x3, ys, rcond=None)
e = ys - x3 @ co; cov = (e @ e / (len(ys) - 3)) * np.linalg.inv(x3.T @ x3)
print("8B quadratic curvature t:", round(co[2] / np.sqrt(cov[2, 2]), 2))
