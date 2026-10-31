"""PREREG v1 analysis: curve fit, gate gap with paired bootstrap, split, status (both labelings), scout clauses,
secondaries. Reads data/phase0/{p_table,phase0}.json, data/runs/*.jsonl, data/judge/*.json, data/ft/*.json.
Run: env/.venv/bin/python swarm-theory-1031-refusal-dose-law/analyze.py   -> data/results.json, data/fig_data.json"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

P = Path(__file__).parent
PT = json.loads((P / "data/phase0/p_table.json").read_text())
INFO = json.loads((P / "data/phase0/phase0.json").read_text())
M5 = [6, 12, 18, 24, 30]
NB, SEED = 2000, 0
STRATA = [(0, 50), (50, 100), (100, 200)]  # paired bootstrap: resample within strata; a cell of n uses strata < n


def load_cells():
    cells = {}
    for f in sorted((P / "data/judge").glob("*.json")):
        j = json.loads(f.read_text())
        c = j["cell"]
        labs = np.array(j["labels"])
        n = len(labs)
        key = c if c != "base" else None
        cells[c] = {"n": n, "comply": (labs == "COMPLY").astype(float), "nonref": (labs != "REFUSE").astype(float),
                    "incoh": float((labs == "INCOHERENT").mean()), "unk": int(np.char.startswith(labs, "UNK").sum())}
        if key is not None:
            t = PT[key]
            for q in ("p", "d", "p_last", "u"):
                cells[c][q] = float(np.mean(t[q][:n]))
        else:
            for q in ("p", "d", "p_last", "u"):
                cells[c][q] = 0.0
        ft = P / "data/ft" / f"{c}.json"
        if ft.exists() and c != "base":
            fj = json.loads(ft.read_text())
            cells[c]["p_down"] = float(np.mean(fj["p_down"][:min(n, 100)]))
    return cells


def unit(c):
    return c.split("@")[0]


def group(c):
    u = unit(c)
    for g in ("TWIN", "D_", "RN", "RD", "NN", "PM", "FK", "G_on", "G_off"):
        if u.startswith(g):
            return g.rstrip("_")
    return "G" if u == "G" else "base"


def fit(x, k, n, iters=50):
    """Binomial MLE of sigmoid(a + b x) on aggregated counts (== reply-level MLE when x is per cell)."""
    X = np.stack([np.ones_like(x), x], 1)
    th = np.array([np.log((k.sum() + .5) / (n.sum() - k.sum() + .5)), 0.0])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (k - n * mu)
        H = (X * (n * mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        th = th + step
        if np.abs(step).max() < 1e-10:
            break
    return th


def sig(th, x):
    return 1 / (1 + np.exp(-(th[0] + th[1] * np.asarray(x))))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def analyze(cells, lab, idx=None):
    """One pass (point estimate if idx is None, else a bootstrap resample: dict stratum -> index array)."""

    def rate(c):
        v = cells[c][lab]
        if idx is None:
            return v.mean()
        ii = np.concatenate([idx[s] for s in range(3) if STRATA[s][1] <= len(v)])
        return v[ii].mean()

    fitset = [c for c in cells if group(c) in ("TWIN", "D", "RN", "RD")]
    literal = [c for c in cells if group(c) in ("RN", "RD")]
    out = {}
    for name, cs in (("fit", fitset), ("literal", literal), ("dialed", [c for c in fitset if group(c) in ("TWIN", "D")])):
        if len(cs) < 2:
            continue
        x = np.array([cells[c]["p"] for c in cs])
        y = np.array([rate(c) for c in cs])
        n = np.array([cells[c]["n"] for c in cs], float)
        th = fit(x, y * n, n)
        out[name] = {"theta": th.tolist(), "r2": r2(y, sig(th, x)), "cells": len(cs)}
        out[name]["gap"] = {m: float(100 * (rate(f"G@{m}") - sig(th, cells[f"G@{m}"]["p"])))
                            for m in M5 if f"G@{m}" in cells}
    base = rate("base")
    eff = {c: 100 * (rate(c) - base) for c in cells if cells[c]["n"] == 200 and c != "base"}
    out["base"] = 100 * base
    out["eff"] = eff
    if all(f"{u}@30" in cells for u in ("G", "G_on", "G_off")):
        eg, eon, eoff = eff["G@30"], eff["G_on@30"], eff["G_off@30"]
        out["split"] = {"eff_G": eg, "eff_on": eon, "eff_off": eoff, "on_minus_G": eon - eg,
                        "off_ratio": eoff / eg if eg != 0 else float("nan")}
    out["split_all"] = {m: {u: eff.get(f"{u}@{m}") for u in ("G", "G_on", "G_off")} for m in M5}
    # pure dose (clause 4): sigmoid in d on the fit set
    if len(fitset) >= 2:
        x = np.array([cells[c]["d"] for c in fitset])
        y = np.array([rate(c) for c in fitset])
        n = np.array([cells[c]["n"] for c in fitset], float)
        th = fit(x, y * n, n)
        out["dose_r2"] = r2(y, sig(th, x))
        out["dose_theta"] = th.tolist()
    return out


def boot(cells, lab):
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(NB):
        idx = [rng.integers(a, b, b - a) for a, b in STRATA]
        reps.append(analyze(cells, lab, idx))
    return reps


def ci(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def status(pt, reps):
    gap = pt["fit"]["gap"]
    gci = {m: ci([r["fit"]["gap"][m] for r in reps]) for m in gap}
    a_yes_n = sum(abs(gap[m]) <= 10 and gci[m][0] >= -15 and gci[m][1] <= 15 for m in gap)
    a_no_n = sum(abs(gap[m]) > 10 and (gci[m][0] > 0 or gci[m][1] < 0) for m in gap)
    a_yes, a_no = a_yes_n >= 4, a_no_n >= 2
    sp = pt.get("split")
    out = {"gap": gap, "gap_ci": gci, "a_yes_pins": a_yes_n, "a_no_pins": a_no_n, "a_yes": a_yes, "a_no": a_no}
    if sp:
        ratio_ci = ci([r["split"]["off_ratio"] for r in reps])
        onm_ci = ci([r["split"]["on_minus_G"] for r in reps])
        b_yes = abs(sp["on_minus_G"]) <= 10 and sp["eff_off"] < 0.25 * sp["eff_G"]
        b_no = ratio_ci[0] >= 0.25 or (abs(sp["on_minus_G"]) > 10 and (onm_ci[0] > 0 or onm_ci[1] < 0))
        out.update(split=sp, off_ratio_ci=ratio_ci, on_minus_G_ci=onm_ci, b_yes=b_yes, b_no=b_no,
                   eff_G30_ci=ci([r["split"]["eff_G"] for r in reps]))
        gate_ok = sp["eff_G"] >= 20
        out["gate_check"] = gate_ok
        if not gate_ok:
            out["status"] = "UNCONFIRMED"
        elif a_no or b_no:
            out["status"] = "REJECTED"
        elif a_yes and b_yes:
            out["status"] = "CONFIRMED"
        else:
            out["status"] = "UNCONFIRMED"
    return out


def main():
    cells = load_cells()
    res = {"n_cells": len(cells), "cells": {}}
    for c, v in cells.items():
        res["cells"][c] = {k: (float(v[k].mean()) if isinstance(v[k], np.ndarray) else v[k]) for k in v}
        res["cells"][c]["group"] = group(c)
    if "base" not in cells or "G@30" not in cells:
        (P / "data/results.json").write_text(json.dumps(res, indent=1, default=_j))
        print("not enough cells yet", len(cells))
        return
    res["labelings"] = {}
    for lab in ("comply", "nonref"):
        pt = analyze(cells, lab)
        reps = boot(cells, lab) if "fit" in pt else []
        st = status(pt, reps) if reps else {}
        res["labelings"][lab] = {"point": pt, "status": st,
                                 "eff_ci": {c: ci([r["eff"][c] for r in reps]) for c in pt["eff"]} if reps else {}}
    sts = [res["labelings"][l]["status"].get("status") for l in ("comply", "nonref")]
    res["STATUS"] = sts[0] if sts[0] == sts[1] else "UNCONFIRMED"
    res["status_by_labeling"] = dict(zip(("comply", "nonref"), sts))
    pt = res["labelings"]["comply"]["point"]
    # scout clauses
    cl = {}
    if "literal" in pt:
        cl["c1_literal_gap"] = pt["literal"]["gap"]
        cl["c1_held"] = all(abs(g) <= 10 for g in pt["literal"]["gap"].values()) and len(pt["literal"]["gap"]) == 5
        cl["c1_fitset_gap"] = pt["fit"]["gap"]
    if "fit" in pt:
        cl["c2_r2_fit"] = pt["fit"]["r2"]
        cl["c2_r2_dialed"] = pt.get("dialed", {}).get("r2")
        cl["c2_held"] = pt["fit"]["r2"] >= 0.8
    cl["c4_dose_r2"] = pt.get("dose_r2")
    cl["c4_held"] = pt.get("dose_r2") is not None and pt["dose_r2"] < 0.3
    # clause 3: first-token Spearman over gate + 100 RN at m=30 (TEST100)
    ftb = P / "data/ft/base.json"
    if ftb.exists():
        b = np.array(json.loads(ftb.read_text())["score"])
        # signed cos(w_j, r_hat) for RN units: p is linear in (vec . r_hat) with the same coefficient as the gate,
        # so cos_j = p(RN_j@30) / p(G@30) * cos_gate
        gp = np.mean(PT["G@30"]["p"][:100])
        rows = []
        for u in ["G"] + [f"RN_{j}" for j in INFO["RN"]]:
            f = P / "data/ft" / f"{u}@30.json"
            if not f.exists():
                continue
            s = np.array(json.loads(f.read_text())["score"])
            ch = s - b
            c = np.mean(PT[f"{u}@30"]["p"][:100]) / gp * INFO["gate_cos"]
            rows.append((u, c, ch.mean(), ch[0::2].mean(), ch[1::2].mean()))
        if len(rows) > 10:
            cs = np.array([r[1] for r in rows]); chg = np.array([r[2] for r in rows])
            cl["c3_n_units"] = len(rows)
            cl["c3_split_half_spearman"] = float(spearmanr(np.abs([r[3] for r in rows]), np.abs([r[4] for r in rows]))[0])
            cl["c3_spearman_abs"] = float(spearmanr(np.abs(cs), np.abs(chg))[0])
            cl["c3_spearman_signed"] = float(spearmanr(cs, chg)[0])
            cl["c3_held"] = cl["c3_spearman_abs"] >= 0.7
            cl["c3_gate_rank_by_change"] = int(1 + sum(-r[2] > -rows[0][2] for r in rows[1:]))
            res["ft_units"] = [{"unit": r[0], "signed_cos": float(r[1]), "score_change": float(r[2])} for r in rows]
    cl["held_all"] = all(cl.get(k) for k in ("c1_held", "c2_held", "c3_held", "c4_held"))
    # secondaries
    sec = {}
    fitset = [c for c in cells if group(c) in ("TWIN", "D", "RN", "RD")]
    x = np.array([cells[c]["u"] for c in fitset]); y = np.array([cells[c]["comply"].mean() for c in fitset])
    n = np.array([cells[c]["n"] for c in fitset], float)
    thu = fit(x, y * n, n)
    half = (cells["base"]["comply"].mean() + 1) / 2
    ustar = (np.log(half / (1 - half)) - thu[0]) / thu[1] if thu[1] != 0 else None
    sec["arditi_u_half"] = float(ustar) if ustar is not None else None
    sec["arditi_held"] = ustar is not None and 0.25 <= ustar <= 0.75
    th = np.array(pt["fit"]["theta"])
    sec["NN"] = {c: {"obs": 100 * cells[c]["comply"].mean(), "pred": float(100 * sig(th, cells[c]["p"])), "p": cells[c]["p"]}
                 for c in cells if group(c) == "NN"}
    if sec["NN"]:
        sec["NN_mean_abs_err"] = float(np.mean([abs(v["obs"] - v["pred"]) for v in sec["NN"].values()]))
    base50 = 100 * cells["base"]["comply"][:50].mean()
    sec["NN_pos_m30_effect_vs_base50"] = {c: 100 * cells[c]["comply"].mean() - base50 for c in cells
                                          if group(c) == "NN" and c.endswith("@30") and int(unit(c)[3:]) in INFO["NN_pos"]}
    sec["twin_vs_gate"] = {m: {"G": 100 * cells[f"G@{m}"]["comply"][:100].mean(),
                               "TWIN": [100 * cells[f"TWIN_{k}@{m}"]["comply"].mean() for k in range(3) if f"TWIN_{k}@{m}" in cells]}
                           for m in M5 if f"G@{m}" in cells}
    for g in ("PM", "FK"):
        sec[g] = {c: {"obs": 100 * cells[c]["comply"].mean(), "pred": float(100 * sig(th, cells[c]["p"])),
                      "p": cells[c]["p"], "incoh": 100 * cells[c]["incoh"]} for c in cells if group(c) == g}
    dn = [c for c in fitset if "p_down" in cells[c]]
    if len(dn) > 5:
        xd = np.array([cells[c]["p_down"] for c in dn]); yd = np.array([cells[c]["comply"].mean() for c in dn])
        nd = np.array([cells[c]["n"] for c in dn], float)
        thd = fit(xd, yd * nd, nd)
        sec["downstream"] = {"L_star": INFO["L_star"], "r2": r2(yd, sig(thd, xd)), "cells": len(dn),
                             "gate_gap": {m: float(100 * (cells[f"G@{m}"]["comply"].mean() - sig(thd, cells[f"G@{m}"]["p_down"])))
                                          for m in M5 if "p_down" in cells.get(f"G@{m}", {})}}
    sec["incoherent_pct"] = {c: round(100 * cells[c]["incoh"], 1) for c in cells}
    res["secondary"] = sec
    (P / "data/results.json").write_text(json.dumps(res, indent=1, default=_j))
    # figure data
    fig = {"cells": [{"cell": c, "group": group(c), "unit": unit(c), "m": (c.split("@")[1] if "@" in c else None),
                      "p": cells[c]["p"], "d": cells[c]["d"], "n": cells[c]["n"],
                      "comply": 100 * cells[c]["comply"].mean(), "nonref": 100 * cells[c]["nonref"].mean(),
                      "incoh": 100 * cells[c]["incoh"],
                      "ci": [100 * q for q in wilson(cells[c]["comply"].sum(), cells[c]["n"])]} for c in cells],
           "curve": {l: res["labelings"][l]["point"]["fit"]["theta"] for l in res["labelings"]},
           "curve_literal": res["labelings"]["comply"]["point"].get("literal", {}).get("theta"),
           "status": res["STATUS"]}
    (P / "data/fig_data.json").write_text(json.dumps(fig, indent=1, default=_j))
    print(json.dumps({"STATUS": res["STATUS"], "by": res["status_by_labeling"],
                      "comply": {k: res["labelings"]["comply"]["status"].get(k) for k in
                                 ("gap", "gap_ci", "a_yes_pins", "a_no_pins", "split", "off_ratio_ci", "b_yes", "b_no")},
                      "fit_r2": pt["fit"]["r2"], "base": pt["base"], "clauses": cl}, indent=1, default=float))


def _j(o):
    return o.item() if hasattr(o, "item") else str(o)


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


if __name__ == "__main__":
    main()
