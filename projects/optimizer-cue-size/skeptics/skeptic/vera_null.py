"""Vera null tests on the headline runs (own T from the raw curves, vera_check helpers).

N1 exact optimizer-label permutation: within each cue size, every way to pick which runs carry the 'sgd' label among
   the AdamW+SGD runs of that size (C(4,2)=6 per size, 6^4 = 1,296 relabelings). Null of d_SGD = b_SGD - b_AdamW.
N2 cue-size label shuffle: 20,000 shuffles of x among the runs of each optimizer (slope null per optimizer and for d).
N4 Freedman-Lane residual permutation (the right null for an interaction when the levels differ): fit the reduced
   model ln T = a_opt + b ln x (common slope), permute its residuals 20,000 times, refit the full model, read d.
   N1 mixes the optimizers' very different levels (SGD's T is 5-10x AdamW's at equal x), so it has little power.
N3 seeds are real: max |P(X|off) s0 - s1| over shared eval steps per (optimizer, size).
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as vc  # noqa: E402

SGD_LR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
OUT = os.path.dirname(os.path.abspath(__file__))


def headline_runs():
    out = []
    for f in sorted(vc.glob.glob(f"{vc.P}/data/runs/*.json")):
        r = json.load(open(f))
        a = r["args"]
        if a.get("arm") != "cue" or a.get("freeze_a") or a.get("group") != 8 or not a.get("no_cast"):
            continue
        if not any(abs(a["cue_alpha"] - s) / s < 1e-6 for s in vc.SIZES) or a["seed"] not in (0, 1):
            continue
        if a["opt"] == "sgd" and abs(a["lr"] - SGD_LR) > 1e-12:
            continue
        if a["opt"] in ("adamw", "sign") and abs(a["lr"] - 1e-4) > 1e-12:
            continue
        cur = vc.merged_curve(r)
        T, c = vc.my_T(cur)
        out.append(dict(opt=a["opt"], alpha=a["cue_alpha"], seed=a["seed"], lx=float(np.log(r["cue_size"]["x"])),
                        lT=float(np.log(T)), cens=c, curve=cur))
    return out


def slope(lx, ly):
    return np.polyfit(lx, ly, 1)[0]


runs = headline_runs()
res = {}
for other in ("sgd", "sign"):
    sub = [r for r in runs if r["opt"] in ("adamw", other)]
    if sum(r["opt"] == other for r in sub) < 4:
        continue
    lx = np.array([r["lx"] for r in sub])
    ly = np.array([r["lT"] for r in sub])
    lab = np.array([r["opt"] == other for r in sub])
    obs = slope(lx[lab], ly[lab]) - slope(lx[~lab], ly[~lab])
    # N1 exact within-size relabeling
    groups = {}
    for i, r in enumerate(sub):
        groups.setdefault(r["alpha"], []).append(i)
    per = []
    for a, idx in sorted(groups.items()):
        k = int(lab[idx].sum())
        per.append([set(c) for c in itertools.combinations(idx, k)])
    null = []
    for combo in itertools.product(*per):
        sel = set().union(*combo)
        m = np.array([i in sel for i in range(len(sub))])
        null.append(slope(lx[m], ly[m]) - slope(lx[~m], ly[~m]))
    null = np.array(null)
    p1 = float(np.mean(null <= obs)) if obs < 0 else float(np.mean(null >= obs))
    # N2 size shuffle within optimizer
    rng = np.random.default_rng(0)
    n2 = []
    for _ in range(20000):
        lxs = lx.copy()
        for m in (lab, ~lab):
            lxs[m] = rng.permutation(lx[m])
        n2.append(slope(lxs[lab], ly[lab]) - slope(lxs[~lab], ly[~lab]))
    n2 = np.array(n2)
    p2 = float(np.mean(np.abs(n2) >= abs(obs)))
    # N4 Freedman-Lane
    Xr = np.c_[lab.astype(float), (~lab).astype(float), lx]
    br, *_ = np.linalg.lstsq(Xr, ly, rcond=None)
    fit_r = Xr @ br
    resid = ly - fit_r
    n4 = []
    for _ in range(20000):
        yp = fit_r + rng.permutation(resid)
        n4.append(slope(lx[lab], yp[lab]) - slope(lx[~lab], yp[~lab]))
    n4 = np.array(n4)
    p4 = float(np.mean(np.abs(n4) >= abs(obs)))
    print(f"{other}-adamw: N4 Freedman-Lane null 95% {np.round(np.percentile(n4, [2.5, 97.5]), 3)}, two-sided p {p4:.5f} ({int(np.sum(np.abs(n4) >= abs(obs)))}/20000)")
    res[other] = dict(n4_q=np.percentile(n4, [2.5, 97.5]).tolist(), n4_p=p4, d_obs=float(obs), n1_size=len(null), n1_q=np.percentile(null, [2.5, 50, 97.5]).tolist(), n1_p=p1,
                      n2_q=np.percentile(n2, [2.5, 97.5]).tolist(), n2_p=p2)
    print(f"{other}-adamw: d_obs {obs:.3f} | N1 exact relabel ({len(null)}): null 95% {np.round(res[other]['n1_q'], 3)}, "
          f"one-sided p {p1:.4f} | N2 size shuffle: null 95% {np.round(res[other]['n2_q'], 3)}, two-sided p {p2:.5f}")

# N3 seeds real
seeds = {}
for r in runs:
    seeds.setdefault((r["opt"], r["alpha"]), {})[r["seed"]] = {c["step"]: c["px_off"] for c in r["curve"]}
n3 = {}
for k, v in sorted(seeds.items()):
    if 0 in v and 1 in v:
        common = sorted(set(v[0]) & set(v[1]) - {0})
        n3[f"{k[0]}_{k[1]}"] = max(abs(v[0][s] - v[1][s]) for s in common)
print("N3 max |P(X|off) s0-s1| per cell:", {k: round(x, 3) for k, x in n3.items()})
res["n3"] = n3
json.dump(res, open(f"{OUT}/vera_null.json", "w"), indent=1)
