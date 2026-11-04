"""Main-figure data for transcoder-neuron: reads data/raw/L{8..11}_units.csv + _pushes.csv (run.py) and writes
figures/transcoder-neuron/data.json.

  python figure_data.py --raw data/raw --out figures/transcoder-neuron/data.json

PRIMARY (PREREG v2.6, READ arm): pred = total_r (transcoder's response to the removal push, read through the real
model's logit gradient), measured = meas_local_r (real MLP's response, same push, same gradient). R^2 about y = x, pooled over
seeds; 95% interval by bootstrap over units (2,000 draws). Theory (Prediction C): 1 - sum|t-r|^2 / sum|r|^2 over the
same pushes. Random push: rnd_total_g vs rnd_meas_g. Law (Prediction B): per-unit median over contexts of
|meas - pred| / (|meas| + |pred|) against the joint cosine.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/pushes")
ap.add_argument("--out", default="figures/transcoder-neuron/data.json")
ap.add_argument("--blog", default="", help="also copy data.json + plot.js to <blog>/assets/figures/transcoder-neuron/")
ap.add_argument("--layers", default="8,9,10,11")
ap.add_argument("--suffix", default="", help="'_smoke' for smoke files")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--law", action="store_true", help="keep the per-feature law points (variants page only)")
ap.add_argument("--text", default="figure_text.json", help="optional overrides for title/subtitle/notes")
ap.add_argument("--status", default="data/status.json", help="analyze.py output; its cell numbers win (cross-checked)")
ap.add_argument("--points", type=int, default=160, help="scatter points kept per layer and group")
args = ap.parse_args()

# column names for the theory sums (Prediction C); first match wins
ERR2 = ["err2", "tr_sq", "t_minus_r_sq", "tr2"]
RSQ = ["r2v", "r_sq", "r_norm_sq", "r2"]
GROUPS = ["NEURON", "GAP", "OTHER", "REST", "ORACLE"]  # GAP (full run) = OTHER (smoke): the gap zone
# PREREG v2.6 (8ece4089): PRIMARY uses the real-model gradient for both sides
PRED, MEAS = "total_r", "meas_local_r"
RND = [("rnd_total_r", "rnd_meas_r"), ("rnd_total_g", "rnd_meas_g")]
rng = np.random.default_rng(0)


def r2_yx(pred, meas):
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    den = ((meas - meas.mean()) ** 2).sum()
    return float(1 - ((meas - pred) ** 2).sum() / den) if den > 0 else None


def boot_ci(df, pcol, mcol):
    """95% interval of R^2 about y = x, resampling units (all contexts of a drawn unit come along)."""
    gb = [g for _, g in df.groupby("unit")]
    P = [g[pcol].to_numpy() for g in gb]
    M = [g[mcol].to_numpy() for g in gb]
    vals = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(gb), len(gb))
        v = r2_yx(np.concatenate([P[i] for i in idx]), np.concatenate([M[i] for i in idx]))
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def rnd(v, k=3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), k)


cells, layers_out, points, law = [], [], [], []
for L in [int(x) for x in args.layers.split(",")]:
    up = os.path.join(args.raw, f"L{L}{args.suffix}_units.csv")
    pp = os.path.join(args.raw, f"L{L}{args.suffix}_pushes.csv")
    if not os.path.exists(pp):
        continue
    U = pd.read_csv(up)
    P = pd.read_csv(pp)
    if "pre_ok" in P.columns:
        P = P[P.pre_ok > 0]
    e2c, rsc = pick(P, ERR2), pick(P, RSQ)
    # DEFINED gate: a perfect transcoder (pred = measured_local) must predict the real end-to-end change
    defined = r2_yx(P[MEAS], P.e2e_real) if "e2e_real" in P.columns else None
    for g in GROUPS:
        d = P[P.group == g]
        if d.empty:
            continue
        c = {"layer": L, "group": g, "n_units": int(d.unit.nunique()), "n_pushes": int(len(d)),
             "r2": rnd(r2_yx(d[PRED], d[MEAS])), "ci": [rnd(v) for v in boot_ci(d, PRED, MEAS)],
             "seeds": [rnd(r2_yx(s[PRED], s[MEAS])) for _, s in d.groupby("seed")],
             "random": next((rnd(r2_yx(d[a], d[b])) for a, b in RND if a in d.columns), None),
             "graph_e2e": rnd(r2_yx(d.total_g, d.e2e_real)) if "e2e_real" in d.columns else None,
             "theory": rnd(1 - d[e2c].sum() / d[rsc].sum()) if e2c and rsc else None}
        cells.append(c)
        keep = d.sample(min(args.points, len(d)), random_state=L)
        points += [{"layer": L, "group": g, "pred": rnd(a, 4), "meas": rnd(b, 4)}
                   for a, b in zip(keep[PRED], keep[MEAS])]
        if g != "ORACLE":
            rel = (d[MEAS] - d[PRED]).abs() / (d[MEAS].abs() + d[PRED].abs()).replace(0, np.nan)
            per = rel.groupby(d.unit).median()
            for k, v in per.items():
                if math.isfinite(v):
                    law.append({"layer": L, "group": g, "x": rnd(U.joint.iloc[int(k)]), "err": rnd(v)})
    layers_out.append({"layer": L, "defined": rnd(defined)})

# analyze.py's status.json is the source of the cell numbers; ours are the cross-check
ST = json.load(open(args.status)) if os.path.exists(args.status) else None
KEYS = {"r2": "R2", "ci": "R2_ci", "seeds": "R2_seed", "theory": "R2_theory", "random": "R2_rnd_push",
        "write": "R2_write_plus_real", "fonly": "R2_fonly", "graph_e2e": "R2_graph_vs_e2e"}
if ST:
    for c in cells:
        sc = ST["cells"].get(str(c["layer"]), ST["cells"].get(c["layer"], {})).get(c["group"])
        if not sc:
            continue
        for k, sk in KEYS.items():
            if sk in sc:
                mine, theirs = c.get(k), sc[sk]
                if isinstance(mine, (int, float)) and isinstance(theirs, (int, float)) and abs(mine - theirs) > 0.02:
                    print(f"CROSS-CHECK L{c['layer']} {c['group']} {k}: mine {mine} vs status.json {theirs}")
                c[k] = [rnd(v) for v in theirs] if isinstance(theirs, list) else rnd(theirs) if isinstance(theirs, (int, float)) else theirs

# status per layer (PREREG v2.5 status rule + v2.4 seed guard), for the figure's verdict column only
def verdict(nr, rr):
    if nr is None or rr is None:
        return None, None
    if rr >= 0.7:
        return "NO", "both faithful"
    if nr < 0.4:
        return "NO", "neuron copies fail too"
    if nr - rr < 0.1:
        return "NO", "no difference"
    if nr >= 0.7 and rr <= 0.4:
        return "YES", None
    return "NEITHER", None


for lo in layers_out:
    L = lo["layer"]
    cn = next((c for c in cells if c["layer"] == L and c["group"] == "NEURON"), None)
    cr = next((c for c in cells if c["layer"] == L and c["group"] == "REST"), None)
    if not (cn and cr):
        continue
    v, why = verdict(cn["r2"], cr["r2"])
    agree = sum(verdict(a, b)[0] == v for a, b in zip(cn["seeds"], cr["seeds"]))
    if lo["defined"] is not None and lo["defined"] < 0.7:
        v, why = "UNDEFINED", "logit read-out failed"
    elif v in ("YES", "NO") and agree < 2:
        v, why = "MIXED", f"{agree} of 3 seeds agree"
    lo.update({"verdict": v, "clause": why, "seed_agree": agree})
    sl = ST and ST["layers"].get(str(L))
    if sl:
        if sl.get("verdict") != v or (v == "NO" and sl.get("no_clause") != why):
            print(f"CROSS-CHECK L{L} verdict: mine {v}/{why} vs status.json {sl.get('verdict')}/{sl.get('no_clause')}")
        lo.update({"verdict": sl.get("verdict"), "clause": sl.get("no_clause"), "defined": rnd(sl.get("defined_R2")),
                   "seed_verdicts": sl.get("seed_verdicts")})
if ST and ST.get("law"):
    law = [{"layer": u["L"], "group": u["group"], "x": rnd(u["joint"]), "err": rnd(u["relerr_read"])}
           for u in ST["law"] if u.get("relerr_read") is not None and u["group"] != "ORACLE"]
    if ST.get("status"):
        print("status.json status:", ST["status"] if isinstance(ST["status"], str) else json.dumps(ST["status"])[:300])

# words first (Selma owns the wording; the title states the answer once the status is final)
TEXT = {
    "title": "Does a transcoder predict the real MLP only for features that copy one neuron?",
    "subtitle": "GPT-2 small, public transcoders. Each push turns one feature off by moving the MLP's input; "
                "we compare the transcoder's predicted effect on the next-token logit with the real MLP's effect.",
    "panelA": "How well the prediction matches, per layer and feature group",
    "panelA_note": "R\u00b2 about the diagonal: 1 = predicted exactly, 0 = no better than the average, below 0 = worse. "
                   "Line: 95% interval over features.",
    "note": "",
}
if os.path.exists(args.text):
    TEXT.update(json.load(open(args.text)))
data = {**TEXT, "status": ST.get("status") if ST else None, "layers": layers_out, "cells": cells,
        **({"law": law} if args.law else {}),
        **({"points": points} if args.points else {}),
        "thresholds": {"neuron_yes": 0.7, "rest_yes": 0.4}}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(data, open(args.out, "w"), separators=(",", ":"))
if args.blog:
    import shutil
    dst = os.path.join(args.blog, "assets/figures/transcoder-neuron")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(args.out, os.path.join(dst, "data.json"))
    print("copied data.json to", dst)
print(f"wrote {args.out}: {len(cells)} cells, {len(points)} points, {len(law)} law units")
for c in cells:
    print(f"L{c['layer']} {c['group']:6s} R2 {c['r2']} ci {c['ci']} seeds {c['seeds']} theory {c['theory']} "
          f"random {c['random']} graph_e2e {c['graph_e2e']} n={c['n_units']}/{c['n_pushes']}")
for lo in layers_out:
    print(lo)
