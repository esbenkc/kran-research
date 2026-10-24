"""Main figure data (blog assets/figures/curvature-cut-probe/data.json) + per-cell summary (results/price.json).

Per arm x seed x checkpoint, from the saved ckpt_<step>.npz (scores of every probe on all statements, per-token KL):
  catch      = share of the 596 eval false above the RE-SET line (2nd-highest of the 200 line-setting true, this state)
  catch_clean= the same at the CLEAN model's line of that probe
  kl         = mean over the 1192 eval statements of KL(p_clean || p_tuned) at the country-token position
  kl_all     = mean over every eval token
  shift_true / shift_false = mean score change of eval true / false statements vs step 0 (probe logits)
PRICE (PREREG s5 as changed by DEVIATIONS D1): arm price = smallest seed-mean kl over checkpoints 50..300 whose pooled
catch <= 0.20; per-seed price = smallest kl over that seed's checkpoints 50..300 with catch <= 0.20 (None if never).
The first-crossing price of PREREG s5 (linear interpolation of the pooled curve) is reported next to it.
Status (PREREG s7 with D1) is computed from these numbers.

usage: python figure/export_fig.py [--floors data/floor/band.json] [--blog ../blog.kran.ai]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def threshold(neg_scores, fpr=0.01):
    """common.threshold, copied so this script needs no torch: flag iff score > t, t = k-th highest of n negatives,
    k = floor(fpr (n + 1)); n = 200 -> 2nd highest."""
    s = np.sort(np.asarray(neg_scores))[::-1]
    return float(s[max(1, int(np.floor(fpr * (len(s) + 1)))) - 1])


_SPLIT = json.loads((HERE / "data/ref/split.json").read_text())  # common.rows(): y = 1 - true (1 = false)
Y = np.array([1 - r["true"] for r in _SPLIT])
SPLIT = np.array([r["split"] for r in _SPLIT])
EVPOS = np.where((SPLIT == "eval") & (Y == 1))[0]  # eval false (y = 1 = false)
EVNEG = np.where((SPLIT == "eval") & (Y == 0))[0]
LINE = np.where((SPLIT == "refit") & (Y == 0))[0]  # the 200 line-setting true statements

EV = np.where(SPLIT == "eval")[0]
STEPS = [0, 50, 100, 150, 200, 250, 300]
HIDE = 0.20
# arm -> (run-dir glob pattern with {s}, probe column name)
ARMS = {
    "U": [("data/runs/unproj__lam10000__seed{s}", "unproj"), ("data/runs1023/r29__lam10000__seed{s}", "29")],
    "P": [("data/runs/proj__lam10000__seed{s}", "proj")],
    "R": [("data/runs/rand__lam10000__seed{s}", "rand")],
}


def cell(run, col):
    rows, s0 = [], None
    for st in STEPS:
        f = run / f"ckpt_{st}.npz"
        if not f.exists():
            return rows if len(rows) == len(STEPS) else None
        z = np.load(f)
        names = [str(x) for x in (z["probes"] if "probes" in z.files else z["scored"])]
        sc = z["scores"][:, names.index(col)].astype(np.float64)
        kl, koff, cpos = z["kl"], z["koff"], z["cpos"]
        if st == 0:
            s0 = sc
            line0 = threshold(sc[LINE])
        line = threshold(sc[LINE])
        rows.append({
            "step": st,
            "catch": float((sc[EVPOS] > line).mean()),
            "catch_clean": float((sc[EVPOS] > line0).mean()),
            "line": line,
            "kl": float(np.mean([kl[koff[i] + cpos[i]] for i in EV])),
            "kl_all": float(np.concatenate([kl[koff[i]:koff[i + 1]] for i in EV]).mean()),
            "shift_true": float((sc[EVNEG] - s0[EVNEG]).mean()),
            "shift_false": float((sc[EVPOS] - s0[EVPOS]).mean()),
        })
    return rows


def seed_price(rows):
    v = [r["kl"] for r in rows if r["step"] >= 50 and r["catch"] <= HIDE]
    return min(v) if v else None


def first_cross(pooled):
    for a, b in zip(pooled, pooled[1:]):
        if b["catch"] <= HIDE < a["catch"]:
            t = (a["catch"] - HIDE) / (a["catch"] - b["catch"])
            return a["kl"] + t * (b["kl"] - a["kl"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="data/floor/band.json")
    ap.add_argument("--blog", default=str(HERE.parent / "blog.kran.ai"))
    ap.add_argument("--horizon", type=int, default=300, help="DEVIATIONS D8: 150 or 300")
    ap.add_argument("--u-source", choices=["new", "1023", "both"], default="new",
                    help="which U runs count (DEVIATIONS: 1023 reused iff the new-code s0 repro is within 20 percent)")
    a = ap.parse_args()
    global STEPS
    STEPS = [x for x in STEPS if x <= a.horizon]

    runs = {}
    for arm, srcs in ARMS.items():
        for pat, col in srcs:
            if arm == "U" and a.u_source == "new" and "runs1023" in pat:
                continue
            if arm == "U" and a.u_source == "1023" and "runs1023" not in pat:
                continue
            for s in range(3):
                run = HERE / pat.format(s=s)
                r = cell(run, col) if run.exists() else None
                if r:
                    tag = "1023" if "runs1023" in pat else "new"
                    runs.setdefault(arm, []).append({"seed": s, "src": tag, "rows": r})

    out = {"hide": HIDE, "arms": {}}
    curves, pooled_all, sp = [], [], []
    for arm, rr in runs.items():
        # the status uses ONE run per (arm, seed): for U prefer the new-code run when both exist (--u-source both = D10 slip case)
        by_seed = {}
        for r in rr:
            if r["seed"] not in by_seed or r["src"] == "new":  # D10 slip case: new-code seed replaces the 1023 seed
                by_seed[r["seed"]] = r
        use = [by_seed[s] for s in sorted(by_seed)]
        pooled = []
        for k, st in enumerate(STEPS):
            pooled.append({"arm": arm, "step": st,
                           "catch": float(np.mean([r["rows"][k]["catch"] for r in use])),
                           "kl": float(np.mean([r["rows"][k]["kl"] for r in use])),
                           "kl_all": float(np.mean([r["rows"][k]["kl_all"] for r in use]))})
        hid = [p["kl"] for p in pooled if p["step"] >= 50 and p["catch"] <= HIDE]
        price = min(hid) if hid else None
        sprice = [seed_price(r["rows"]) for r in use]
        out["arms"][arm] = {
            "n_seeds": len(use), "seeds": [r["seed"] for r in use], "src": [r["src"] for r in use],
            "price": price, "price_first_cross": first_cross(pooled),
            "seed_price": sprice, "max_kl": max(p["kl"] for p in pooled),
            "catch_300": pooled[-1]["catch"], "pooled": pooled,
            "per_seed": [{"seed": r["seed"], "src": r["src"], "rows": r["rows"]} for r in use],
            "all_runs": [{"seed": r["seed"], "src": r["src"], "price": seed_price(r["rows"]),
                          "catch_300": r["rows"][-1]["catch"], "kl_300": r["rows"][-1]["kl"]} for r in rr],
        }
        for r in use:
            for row in r["rows"]:
                curves.append({"arm": arm, "seed": r["seed"], "step": row["step"], "catch": row["catch"],
                               "kl": row["kl"], "kl_all": row["kl_all"]})
        pooled_all += pooled
        sp += [{"arm": arm, "seed": r["seed"], "kl": p} for r, p in zip(use, sprice)]

    # status, PREREG s7 with D1 (None-safe: a seed that never hides has price +inf)
    inf = float("inf")
    A = out["arms"]

    def p_(arm):
        return A[arm]["price"] if arm in A and A[arm]["price"] is not None else inf

    def sp_(arm):
        return [inf if v is None else v for v in A.get(arm, {}).get("seed_price", [])]

    st = "UNCONFIRMED"
    if all(arm in A and A[arm]["n_seeds"] >= 2 for arm in "UPR"):
        pu, pp, pr = p_("U"), p_("P"), p_("R")
        r_falls = pr < inf and pu < inf and pr <= 1.5 * pu  # DEVIATIONS D7: one-sided
        yes_a = pp >= 3 * pu and min(sp_("P")) > max(sp_("U"))
        yes_b = pp == inf and A["P"]["catch_300"] >= 0.5
        no = pp <= 1.5 * pu and max(sp_("P")) <= 1.5 * max(sp_("U"))
        if r_falls and (yes_a or yes_b):
            st = "CONFIRMED"
        elif no:
            st = "REJECTED"
        out["status_inputs"] = {"price_U": pu, "price_P": pp, "price_R": pr, "ratio_P_U": pp / pu if pu < inf else None,
                                "ratio_R_U": pr / pu if pu < inf else None, "R_falls": r_falls, "yes_ratio": yes_a,
                                "yes_survive": yes_b, "no": no}
    out["status"] = st
    # U s0 reproduction (PREREG s4): new-code s0 price within 20% of the 1023 s0 price -> the 1023 seeds count
    u = {(r["seed"], r["src"]): seed_price(r["rows"]) for r in runs.get("U", [])}
    if (0, "new") in u and (0, "1023") in u and u[(0, "1023")]:
        ratio = (u[(0, "new")] or float("inf")) / u[(0, "1023")]
        out["u_repro"] = {"price_new_s0": u[(0, "new")], "price_1023_s0": u[(0, "1023")], "ratio": ratio,
                          "pass": abs(ratio - 1) <= 0.2}

    res = HERE / "data/results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "price.json").write_text(json.dumps(out, indent=1, default=lambda x: None))

    fl = HERE / a.floors
    floors = json.loads(fl.read_text()) if fl.exists() else None
    fig = {"terms": json.loads((HERE / "figure/terms.json").read_text())}  # terms + ratio first: oracle figure gate reads data.json[:3000]
    if floors and "U" in A and A["U"]["price"]:
        pu = A["U"]["price"]
        fig["ratio"] = {
            # predicted ratio = a RANGE over methods (Tove #6159): band.json "ratio_pred": {arm: [{m, v, n}, ...]}
            "pred": floors.get("ratio_pred") or {k: [{"m": "exact minimum", "v": floors[k]["cert"] / floors["U"]["cert"],
                                                       "n": floors.get("n", "?")}] for k in A if k != "U"},
            "measured": {k: (A[k]["price"] / pu if A[k]["price"] else None) for k in A if k != "U"},
            "seed": [{"arm": d["arm"], "seed": d["seed"], "r": (d["kl"] / pu if d["kl"] else None)}
                     for d in sp if d["arm"] != "U"],
        }
    fig.update({
        "arm_order": [x for x in ["U", "R", "P"] if x in A],
        "price": {k: v["price"] for k, v in A.items()},
        "seed_price": sp,
        "max_kl": {k: v["max_kl"] for k, v in A.items()},
        # absolute panel: only the proven floor (Selma #6125: the sphere search stays in the foldout)
        "floors": {k: {"pred": None, "naive": floors[k]["cert"]} for k in A} if floors else None,
        "hide": HIDE,
    })
    dst = Path(a.blog) / "assets/figures/curvature-cut-probe/data.json"
    if floors:
        dst.write_text(json.dumps(fig, indent=1))
        print("wrote", dst)
    else:
        print("no floors file yet; figure data not written")
    print(json.dumps({k: {kk: v[kk] for kk in ("n_seeds", "src", "price", "price_first_cross", "seed_price", "catch_300")}
                      for k, v in A.items()}, indent=1))
    print("u_repro:", json.dumps(out.get("u_repro")))
    print("status:", st, json.dumps(out.get("status_inputs"), default=str))


if __name__ == "__main__":
    main()
