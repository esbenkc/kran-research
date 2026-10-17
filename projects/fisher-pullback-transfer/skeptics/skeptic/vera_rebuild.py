"""Vera's independent rebuild of the transfer R^2 from build's raw run files.

Reads only data files (clean acts, probe weights, F/Sigma, run npz/json). Never imports build code.
Usage: python vera_rebuild.py [--runs DIR] [--fisher FILE:KEY]
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vera_common import (D, auroc, load, metrics, pred_drop_exact, pred_drop_gauss, r2, r_P, test_set)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(D / "runs"))
ap.add_argument("--fisher", default="", help="npz path:key of the primary F (default fisher_sigma.npz:F_common)")
ap.add_argument("--out", default="rebuild")
ap.add_argument("--nulls", action="store_true")
ap.add_argument("--glob", default="*__seed*.json")
args = ap.parse_args()

L = load()
W = L["W"]
TEST = 1
targets = L["sel"]["targets"]
scored = L["sel"]["scored_probes"]
src_of = {p["name"]: p["source"] for p in L["probes"]}

F = None
if args.fisher:
    path, key = args.fisher.split(":")
    F = np.load(path)[key]
Ms, lam = metrics(L, 1.0, F)
print(f"lambda = {lam:.6f}")

sec_meta = json.load(open(D / "probes/secondary.json"))
SW = np.load(D / "probes/secondary_weights.npz")["W"]

# ---------- load runs ----------
runs = defaultdict(list)
for jf in sorted(glob.glob(str(Path(args.runs) / args.glob))):
    j = json.load(open(jf))
    if j.get("lambda_kl") == 0 or "lam0" in jf:
        continue
    npz = jf.replace(".json", ".npz")
    acts = jf.replace(".json", "__acts.npz")  # build split activations out of the run npz at ~16:13 CEST
    z = np.load(acts) if Path(acts).exists() else np.load(npz)
    runs[j["target"]].append((j, z, jf))
print({t: [(r[0]["seed"], r[0]["success"], r[0]["stop_step"]) for r in v] for t, v in runs.items()})


def acts_attacked(z, src):
    return z[f"attacked_acts__{src}"].astype(np.float64)


rows = []  # per (target, seed, probe)
checks = []
for t, rl in runs.items():
    w = W[L["name2i"][t]]
    Xw, yw = test_set(L, t, TEST)
    Gw = (Xw @ w)[yw].mean() - (Xw @ w)[~yw].mean()
    for j, z, jf in rl:
        if not j["success"]:
            continue
        Aw = acts_attacked(z, src_of[t])
        assert Aw.shape == Xw.shape, (Aw.shape, Xw.shape)
        sw_att = Aw @ w
        dw = (sw_att[yw].mean() - sw_att[~yw].mean()) - Gw
        checks.append(dict(target=t, seed=j["seed"], my_delta_w=dw, their_delta_w=j["delta_w"], my_Gw=Gw,
                           their_Gw=j["G_w"], my_auc_w=auroc(sw_att, yw), their_auc_w=j["target_auroc_attacked"],
                           retain_kl=j.get("retain_kl_heldout"), stop=j["stop_step"], lam=j["lambda_kl"],
                           loss=j.get("loss", j.get("dprime_sd"))))
        ratio = dw / Gw
        their = {r["probe"]: r for r in j["probes"]}
        # primary
        for u_name in scored:
            if u_name == t:
                continue
            u = W[L["name2i"][u_name]]
            Xu, yu = test_set(L, u_name, TEST)
            Au = acts_attacked(z, src_of[u_name])
            meas = auroc(Xu @ u, yu) - auroc(Au @ u, yu)
            row = dict(set="primary", target=t, seed=j["seed"], probe=u_name, meas=meas,
                       their_meas=their[u_name]["auroc_clean"] - their[u_name]["auroc_attacked"],
                       same_family=src_of[u_name] == src_of[t])
            for k, P in Ms.items():
                row[k] = pred_drop_exact(Xu, yu, u, w, ratio, r_P(P, u, w))
                row[k + "_gauss"] = pred_drop_gauss(Xu, yu, u, w, ratio, r_P(P, u, w))
            row["shared"] = pred_drop_exact(Xu, yu, u, w, ratio, 1.0)
            row["shared_gauss"] = pred_drop_gauss(Xu, yu, u, w, ratio, 1.0)
            row["u_ones"] = u.sum() / np.linalg.norm(u) / np.sqrt(512)
            rows.append(row)
        # secondary: same-label probes of this target, kept only
        for m in sec_meta:
            if m["target"] != t or not m["kept"]:
                continue
            u = SW[m["idx"]]
            meas = auroc(Xw @ u, yw) - auroc(Aw @ u, yw)
            row = dict(set="secondary", target=t, seed=j["seed"], probe=m["name"], meas=meas, same_family=True)
            for k, P in Ms.items():
                row[k] = pred_drop_gauss(Xw, yw, u, w, ratio, r_P(P, u, w))
                row[k + "_exact"] = pred_drop_exact(Xw, yw, u, w, ratio, r_P(P, u, w))
            row["shared"] = pred_drop_gauss(Xw, yw, u, w, ratio, 1.0)
            rows.append(row)

print("\nper-run consistency (mine vs theirs):")
for c in checks:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})


def seed_mean(rows, setname, keys):
    g = defaultdict(list)
    for r in rows:
        if r["set"] == setname:
            g[(r["target"], r["probe"])].append(r)
    out = []
    for (t, p), rs in g.items():
        d = dict(target=t, probe=p, n_seeds=len(rs), same_family=rs[0]["same_family"])
        for k in keys:
            if k in rs[0]:
                d[k] = float(np.mean([r[k] for r in rs]))
        out.append(d)
    return out


KEYS = ["meas", "their_meas", "plain", "whitened", "fisher", "shared", "plain_gauss", "whitened_gauss",
        "fisher_gauss", "shared_gauss", "plain_exact", "whitened_exact", "fisher_exact", "u_ones"]


def report(pairs, label, preds=("fisher", "whitened", "plain", "shared")):
    meas = np.array([p["meas"] for p in pairs])
    res = {k: r2([p[k] for p in pairs], meas) for k in preds if k in pairs[0]}
    print(f"\n[{label}] n_pairs={len(pairs)}  SD(meas)={meas.std():.4f}  mean(meas)={meas.mean():.4f}  "
          f"#|meas|>0.02={int((np.abs(meas) > 0.02).sum())}")
    for k, v in res.items():
        pv = np.array([p[k] for p in pairs])
        print(f"   R2_{k:<15} = {v:8.4f}   (SD pred {pv.std():.4f}, pearson {np.corrcoef(pv, meas)[0,1] if pv.std()>0 else float('nan'):.3f})")
    return res


def verdict(res, a="fisher", b="whitened", c="plain", s="shared"):
    ok1 = res[a] >= 0.6
    ok2 = res[a] - max(res[b], res[c]) >= 0.10
    ok3 = res[a] - res[s] >= 0.10
    return dict(R2_F_ge_0_6=ok1, margin_metrics=res[a] - max(res[b], res[c]), margin_shared=res[a] - res[s],
                CONFIRMED=bool(ok1 and ok2 and ok3))


prim = seed_mean(rows, "primary", KEYS)
sec = seed_mean(rows, "secondary", KEYS)
summary = {}
if prim:
    print("\nmax |my meas - their meas| over primary pairs:",
          max(abs(p["meas"] - p["their_meas"]) for p in prim))
    res = report(prim, "PRIMARY exact map (verdict)")
    summary["primary"] = dict(R2=res, verdict=verdict(res))
    print("   verdict:", summary["primary"]["verdict"])
    report(prim, "primary Gaussian sensitivity", ("fisher_gauss", "whitened_gauss", "plain_gauss", "shared_gauss"))
    nf = [p for p in prim if not p["same_family"]]
    report(nf, "primary, excluding same-dataset pairs")
    # per target
    for t in targets:
        pt = [p for p in prim if p["target"] == t]
        if pt:
            report(pt, f"primary target {t}")
    # bootstrap over targets
    rng = np.random.default_rng(0)
    tl = sorted({p["target"] for p in prim})
    bs = defaultdict(list)
    for _ in range(10000):
        pick = rng.choice(tl, len(tl), replace=True)
        ps = [p for t in pick for p in prim if p["target"] == t]
        m = np.array([p["meas"] for p in ps])
        if m.std() == 0:
            continue
        for k in ("fisher", "whitened", "plain", "shared"):
            bs[k].append(r2([p[k] for p in ps], m))
    print("\nbootstrap over targets (95% CI):",
          {k: tuple(np.round(np.percentile(v, [2.5, 97.5]), 3)) for k, v in bs.items()})
if sec:
    res = report(sec, "SECONDARY same-label stack, gap map")
    summary["secondary"] = dict(R2=res, verdict=verdict(res))
    print("   verdict:", summary["secondary"]["verdict"])
    report(sec, "secondary exact-map sensitivity", ("fisher_exact", "whitened_exact", "plain_exact"))

Path(args.out).mkdir(exist_ok=True)
json.dump(dict(primary_pairs=prim, secondary_pairs=sec, checks=checks, summary=summary, lam=lam),
          open(Path(args.out) / "rebuild.json", "w"), indent=1, default=float)
