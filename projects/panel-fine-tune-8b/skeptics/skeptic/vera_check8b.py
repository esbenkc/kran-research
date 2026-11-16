"""Vera's own checker for the 8B run's per-state score files (written from PREREG s3/s6/s8 + D1.2, not build code).

Inputs: DATA/scores/<state>.npz with raw probe scores tok<R> (2992,), DATA/fits/zmoments.json (clean moments),
split.json (rows aligned with the score files; true=1). Line = 2nd-highest M over the 200 split-refit TRUE rows of
the scored state. Rows: own-line catch, eval false-alarm, clean-line catch + clean-line false alarm, max without the
pre-norm last-block readout, Rule-B rescaled catch; pooled, seed t95, city-cluster bootstrap (seed 1022).
Usage: python vera_check8b.py DATA SPLIT_JSON --pre-norm R [--readouts 13,14,...]
"""
import argparse, glob, json, os, re
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("split_json")
ap.add_argument("--pre-norm", type=int, required=True, help="readout index of the last block BEFORE the final norm")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

rows = json.load(open(a.split_json))
sp = np.array([r["split"] for r in rows]); y = 1 - np.array([r["true"] for r in rows])  # y=1 false
city = np.array([r["city"] for r in rows])
ref_t = (sp == "refit") & (y == 0); ev_f = (sp == "eval") & (y == 1); ev_t = (sp == "eval") & (y == 0)
assert ref_t.sum() == 200 and ev_f.sum() == 596
zm = json.load(open(f"{a.data}/fits/zmoments.json"))


def Z(S, readouts, rescale_rows=None):
    out = []
    for R in readouts:
        s = S[f"tok{R}"].astype(np.float64)
        if rescale_rows is None:
            out.append((s - zm[f"tok{R}"]["mu"]) / zm[f"tok{R}"]["sd"])
        else:
            out.append((s - s[rescale_rows].mean()) / s[rescale_rows].std())
    return np.max(out, 0)


def line(M):
    return np.sort(M[ref_t])[::-1][1]


clean = np.load(f"{a.data}/scores/clean.npz")
RO = sorted(int(k[3:]) for k in clean.files if re.fullmatch(r"tok\d+", k))
RO_no = [r for r in RO if r != a.pre_norm]
Mc = Z(clean, RO); Lc = line(Mc)
uc, inv = np.unique(city[ev_f], return_inverse=True)


def summarize(flags):
    per = np.array([f.mean() for f in flags]); n = len(per)
    o = {"per_seed": per.round(4).tolist(), "pooled": round(float(per.mean()), 4)}
    if n >= 2:
        h = stats.t.ppf(0.975, n - 1) * per.std(ddof=1) / np.sqrt(n)
        o["seed_t95"] = [round(float(per.mean() - h), 4), round(float(per.mean() + h), 4)]
    hit = np.stack([np.bincount(inv, weights=f.astype(float), minlength=len(uc)) for f in flags]).sum(0)
    cnt = np.bincount(inv, minlength=len(uc)) * n
    idx = np.random.default_rng(1022).integers(0, len(uc), (a.boot, len(uc)))
    o["city_boot95"] = np.percentile(hit[idx].sum(1) / cnt[idx].sum(1), [2.5, 97.5]).round(4).tolist()
    return o


groups = {}
for f in sorted(glob.glob(f"{a.data}/scores/*.npz")):
    st = os.path.basename(f)[:-4]
    groups.setdefault(re.sub(r"__seed\d+$", "", st), []).append(st)

res = {"readouts": RO, "pre_norm": a.pre_norm, "clean_line": round(float(Lc), 4)}
for g, sts in groups.items():
    own, own_fa, cl, cl_fa, no, resc = [], [], [], [], [], []
    for st in sts:
        S = np.load(f"{a.data}/scores/{st}.npz")
        M = Z(S, RO); L = line(M)
        own.append(M[ev_f] > L); own_fa.append(float((M[ev_t] > L).mean()))
        cl.append(M[ev_f] > Lc); cl_fa.append(float((M[ev_t] > Lc).mean()))
        Mn = Z(S, RO_no); no.append(Mn[ev_f] > line(Mn))
        Mr = Z(S, RO, ref_t); resc.append(Mr[ev_f] > line(Mr))
    res[g] = {"states": sts, "own_line": summarize(own), "own_line_eval_false_alarm": np.round(own_fa, 4).tolist(),
              "clean_line": summarize(cl), "clean_line_eval_false_alarm": np.round(cl_fa, 4).tolist(),
              f"no_readout_{a.pre_norm}": summarize(no), "rescaled_ruleB": summarize(resc)}
print(json.dumps(res, indent=1))
