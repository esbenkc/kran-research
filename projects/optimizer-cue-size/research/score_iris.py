"""Iris: score my frozen prediction (room #20300, research.md R8) on this room's runs.
Per run: T (rl.py's T over all evals), rise shape of D(t) (power exponent b and exp rate r over D in [0.005, 0.12]
before T). Per group (opt, freeze_a, lr): OLS slope of log T on log x over uncensored runs, bands from R8.
Rebuild: /usr/bin/python3 score_iris.py [runs_dir]   (numpy only). Exploratory: the status comes from analyze.py only."""
import glob, json, os, sys
import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/work/theory-1112-optimizer-sets/kran-research/projects/optimizer-cue-size/data/runs")
BANDS = {("adamw", False): (-1.1, -0.8), ("sgd", False): (-1.3, -0.8), ("sign", False): (-1.3, -0.7),
         ("sgd", True): (-2.1, -1.5)}
rows = []
for f in sorted(glob.glob(f"{RUNS}/*.json")):
    d = json.load(open(f)); a = d.get("args", {})
    if a.get("arm") != "cue" or a.get("cue_dir") != "rand" or "smoke" in d.get("name", "") or "pilot" in d.get("name", ""):
        continue
    c = {r["step"]: r["D"] for r in d.get("dev_curve", [])}
    c.update({r["step"]: r["D"] for r in d.get("dense_curve", []) if "D" in r})
    T = d.get("T"); cens = bool(d.get("censored"))
    st = sorted(s for s in c if (T is None or s <= T) and 0.005 <= c[s] <= 0.12)
    b = r = None
    if len(st) >= 3:
        t = np.array(st, float); y = np.log([c[s] for s in st])
        r = float(np.polyfit(t, y, 1)[0]); b = float(np.polyfit(np.log(t), y, 1)[0])
    rows.append(dict(run=d["name"], opt=a.get("opt"), freeze_a=bool(a.get("freeze_a")), lr=a.get("lr"),
                     seed=a.get("seed"), x=d["cue_size"].get("x") or a.get("cue_alpha"), T=T, censored=cens,
                     n_rise=len(st), b=b, r=r))
groups = {}
for w in rows:
    groups.setdefault((w["opt"], w["freeze_a"], w["lr"]), []).append(w)
out = {"runs": rows, "groups": []}
for (opt, fa, lr), ws in sorted(groups.items(), key=lambda kv: str(kv[0])):
    u = [w for w in ws if not w["censored"] and w["T"]]
    g = {"opt": opt, "freeze_a": fa, "lr": lr, "n": len(ws), "n_uncensored": len(u),
         "n_sizes": len({round(w["x"], 3) for w in u})}
    if g["n_sizes"] >= 2:
        g["slope"] = float(np.polyfit(np.log([w["x"] for w in u]), np.log([w["T"] for w in u]), 1)[0])
        lo, hi = BANDS.get((opt, fa), (None, None))
        g["band"] = [lo, hi]; g["in_band"] = (lo is not None and lo <= g["slope"] <= hi)
    smallest = sorted({round(w["x"], 3) for w in u})[:2]
    g["b_at_2_smallest"] = [w["b"] for w in u if round(w["x"], 3) in smallest]
    out["groups"].append(g)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_iris.json"), "w"), indent=1)
for w in rows:
    print(f"{w['run'][:34]:34s} {w['opt']:5s} A{'frz' if w['freeze_a'] else 'trn'} lr {w['lr']:<7g} s{w['seed']} x {w['x']:.2f} "
          f"T {w['T']} {'CENS' if w['censored'] else ''} b {w['b'] if w['b'] is None else round(w['b'], 2)} "
          f"r {w['r'] if w['r'] is None else round(w['r'], 4)}")
for g in out["groups"]:
    print(g)
