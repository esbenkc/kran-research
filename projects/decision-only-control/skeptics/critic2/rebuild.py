"""critic2 (Claude fallback) independent rebuild of the headline cells from saved raw reads.
Own code, not summ.py. Prompt-clustered bootstrap. Run: python3 skeptics/critic2/rebuild.py"""
import glob, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[2]
os.chdir(ROOT)
z = np.load("data/pool/test/cols.npz")
var, base, nb = z["variant"], z["base_row"], 2800
ni = np.zeros(nb, int); xi = np.zeros(nb, int)
ni[base[var == 0]] = np.where(var == 0)[0]; xi[base[var == 1]] = np.where(var == 1)[0]
keep = ~z["x1_orig"][ni].astype(bool); w = np.exp(z["logw"][ni].astype(float)); pr = z["prompt_row"][ni]
up = np.unique(pr); idx = [np.where(pr == p)[0] for p in up]
rng = np.random.default_rng(777)
B = [np.concatenate([idx[j] for j in rng.integers(0, len(up), len(up))]) for _ in range(1500)]
ALL = np.arange(nb)
out = {}

def met(c, r):
    cn, cx, k, ww = c[ni][r], c[xi][r], keep[r], w[r]
    co = cn.mean(); sP = (cn - cx)[k].mean() / co
    sR = (co - (ww * cn).sum() / ww.sum()) / (np.clip(1 - ww, 0, None).mean() * co)
    return np.array([sP, sR, sP - sR, co])

def q(a): return [round(float(x), 4) for x in np.percentile(a, [2.5, 97.5])]

def rep(run, step):
    c = np.load(f"data/runs/{run}/reads/c_{step}.npy").astype(float)
    c0 = np.load(f"data/runs/{run}/reads/c_0.npy").astype(float)
    p, p0 = met(c, ALL), met(c0, ALL)
    bs = np.array([met(c, b) for b in B]); b0 = np.array([met(c0, b) for b in B])
    return {"s_P": [round(p[0], 4), q(bs[:, 0])], "s_R": [round(p[1], 4), q(bs[:, 1])], "gap": [round(p[2], 4), q(bs[:, 2])],
            "dP": [round(p[0] - p0[0], 4), q(bs[:, 0] - b0[:, 0])], "cheat_off": round(p[3], 4), "start_s_P": round(p0[0], 4)}

out["d010_s0@1500"] = rep("d010_s0", 1500)
out["d086_s0@300"] = rep("d086_s0", 300)
out["null086_s0@300"] = rep("null086_s0", 300)
steps = sorted(int(Path(f).stem[2:]) for f in glob.glob("data/runs/d010_s0/reads/c_*.npy") if Path(f).stem[2:].isdigit())
out["d010_s0_s_P_curve"] = {s: round(float(met(np.load(f"data/runs/d010_s0/reads/c_{s}.npy").astype(float), ALL)[0]), 4) for s in steps}
early, late = [s for s in steps if s <= 500], [s for s in steps if 1100 <= s <= 1500]
C = {s: np.load(f"data/runs/d010_s0/reads/c_{s}.npy").astype(float) for s in early + late}
d = lambda r: np.mean([met(C[s], r)[1] for s in late]) - np.mean([met(C[s], r)[1] for s in early])
out["s_R_late_minus_early"] = [round(float(d(ALL)), 4), q([d(b) for b in B[:1000]])]
sv = np.load("data/reads/served_d010_s0_1500.npz"); cx, cn, sp = sv["c_x"].astype(float), sv["c_notx"].astype(float), sv["prompt_row"]
f = lambda m: (cn[m] - cx[m]).mean() / cn[m].mean()
su = np.unique(sp); si = [np.where(sp == u)[0] for u in su]; r2 = np.random.default_rng(5)
out["served_d010_s0_1500_s_P"] = [round(float(f(np.arange(len(cx)))), 4),
                                  q([f(np.concatenate([si[j] for j in r2.integers(0, len(su), len(su))])) for _ in range(2000)])]
print(json.dumps(out, indent=1))
