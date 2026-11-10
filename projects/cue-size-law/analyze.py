"""PREREG s5 analysis, mechanically, from the saved run curves (numpy + scipy, seconds; no torch).

Inputs : data/runs/<run>.json (rl.py), data/scan.json (x per alpha, x_real, gap0), data/scan_<variant>.json (x of the
         variant directions), PREREG.md (rules), DEVIATIONS.md (changes).
Outputs: data/summary.json (every number the post uses), data/figure_data.json (the one figure), proof-matrix.md.

    /usr/bin/python3 analyze.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).parent
B = 8192
ALPHA_TOP = 0.03125
SWEEP = [ALPHA_TOP / 2 ** k for k in range(6)]


def load_runs():
    runs = {}
    for p in sorted((HERE / "data" / "runs").glob("*.json")):
        r = json.load(open(p))
        runs[r["name"]] = r
    return runs


def ols(lx, lt):
    """Slope, intercept, slope se, n of OLS lt ~ lx."""
    lx, lt = np.asarray(lx, float), np.asarray(lt, float)
    n = len(lx)
    X = np.vstack([lx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(X, lt, rcond=None)
    res = lt - X @ coef
    se = np.sqrt(res @ res / (n - 2) / ((lx - lx.mean()) ** 2).sum()) if n > 2 else float("nan")
    return float(coef[0]), float(coef[1]), float(se), n


def boot(rows, key, draws=2000, seed=0):
    """Percentile bootstrap of the OLS slope, runs resampled within each alpha (PREREG s5)."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["alpha"], []).append(r)
    sl = []
    for _ in range(draws):
        s = [g[i] for g in by.values() for i in rng.integers(0, len(g), len(g))]
        lx = np.log([r[key] for r in s]); lt = np.log([r["T"] for r in s])
        if np.ptp(lx) == 0:
            continue
        sl.append(ols(lx, lt)[0])
    return [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]


def tobit(lx, lt, cens, logB):
    """Right-censored normal regression lt = a + b lx + s e; MLE; Wald CI from the numerical Hessian."""
    lx, lt, cens = map(np.asarray, (lx, lt, cens))

    def nll(p):
        b, a, ls = p
        s = np.exp(ls)
        mu = a + b * lx
        ll = np.where(cens, stats.norm.logsf((logB - mu) / s), stats.norm.logpdf((lt - mu) / s) - ls)
        return -ll.sum()

    b0, a0, _, _ = ols(lx[~cens], lt[~cens]) if (~cens).sum() > 2 else (-1.0, 8.0, 0, 0)
    r = optimize.minimize(nll, [b0, a0, np.log(0.3)], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10,
                                                                                         "maxiter": 20000})
    h = 1e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e_i, e_j = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (nll(r.x + e_i + e_j) - nll(r.x + e_i - e_j) - nll(r.x - e_i + e_j) + nll(r.x - e_i - e_j)) / (4 * h * h)
    try:
        se = float(np.sqrt(np.linalg.inv(H)[0, 0]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return {"slope": float(r.x[0]), "intercept": float(r.x[1]), "sigma": float(np.exp(r.x[2])), "se": se,
            "ci95": [float(r.x[0] - 1.96 * se), float(r.x[0] + 1.96 * se)], "nll": float(r.fun), "n": int(len(lx)),
            "n_censored": int(cens.sum())}


def form_fit(lx, lt, cens, logB, form):
    """Censored likelihood of two fixed-exponent forms (PREREG s9.1): x^-2, and x^-1 ln(K/x) with K free."""
    x = np.exp(lx)

    def mu(p):
        if form == "x^-2":
            return p[0] - 2 * lx
        K = np.exp(p[2]) + x.max()  # K > max x so ln(K/x) > 0
        return p[0] - lx + np.log(np.log(K / x))

    def nll(p):
        s = np.exp(p[1])
        m = mu(p)
        ll = np.where(cens, stats.norm.logsf((logB - m) / s), stats.norm.logpdf((lt - m) / s) - p[1])
        return -ll.sum()

    best = None
    for k0 in ([0.0] if form == "x^-2" else [np.log(1.0), np.log(10.0), np.log(100.0), np.log(1000.0)]):
        p0 = [float(np.mean(lt + (2 if form == "x^-2" else 1) * lx)), np.log(0.5)] + ([] if form == "x^-2" else [k0])
        r = optimize.minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 40000})
        best = r if best is None or r.fun < best.fun else best
    out = {"form": form, "nll": float(best.fun), "params": [float(v) for v in best.x]}
    if form != "x^-2":
        out["K"] = float(np.exp(best.x[2]) + x.max())
    return out


def last_half_slope(curve):
    pts = [(c["step"], c["D"]) for c in curve if c["step"] >= B / 2]
    if len(pts) < 3:
        return None
    s, d = np.array(pts).T
    return float(np.polyfit(s, d, 1)[0] * 1000)  # D change per 1,000 updates


def core(rows, xof, x_real):
    """Primary/Tobit/forms fits, prereg parts and status (PREREG s5, s9, s10) on one set of switch times."""
    unc = [r for r in rows if not r["censored"] and r["in_fit"]]
    S = {}

    # primary fit (status on log x; prereg parts on log alpha, s10.2)
    for key in ("x", "alpha"):
        if len({r["alpha"] for r in unc}) >= 3:
            b, a0, se, n = ols(np.log([r[key] for r in unc]), np.log([r["T"] for r in unc]))
            tq = stats.t.ppf(0.975, n - 2)
            ci_t = [b - tq * se, b + tq * se]
            ci_b = boot(unc, key)
            wider = ci_t if (ci_t[1] - ci_t[0]) >= (ci_b[1] - ci_b[0]) else ci_b
            S[f"primary_{key}"] = {"slope": b, "intercept": a0, "se": se, "n": n, "ci_boot": ci_b, "ci_t": ci_t,
                                   "ci_status": wider, "n_alphas": len({r["alpha"] for r in unc})}
        else:
            S[f"primary_{key}"] = None
    P = S["primary_x"]
    # halves (s9.1)
    for nm, ks in (("top_k0_2", (0, 1, 2)), ("bottom_k3_5", (3, 4, 5))):
        h = [r for r in unc if r["k"] in ks]
        S[f"half_{nm}"] = (ols(np.log([r["x"] for r in h]), np.log([r["T"] for r in h]))[0]
                           if len({r["alpha"] for r in h}) >= 2 else None)
    if S["half_top_k0_2"] is not None and S["half_bottom_k3_5"] is not None:
        S["halves_differ_gt_0.5"] = abs(S["half_top_k0_2"] - S["half_bottom_k3_5"]) > 0.5
    # local slopes between neighbouring alphas (median T over seeds, uncensored only)
    med = {}
    for al in SWEEP:
        Ts = [r["T"] for r in rows if r["alpha"] == al]
        if Ts and all(t is not None for t in Ts):
            med[al] = float(np.median(Ts))
    S["median_T"] = {str(k): v for k, v in med.items()}
    S["local_slopes_x"] = [[a1, a2, float(np.log(med[a2] / med[a1]) / np.log(xof[a2] / xof[a1]))]
                           for a1, a2 in zip(SWEEP, SWEEP[1:]) if a1 in med and a2 in med]
    # Tobit + forms over ALL sweep runs in the fit
    allr = [r for r in rows if r["in_fit"]]
    lx = np.log([r["x"] for r in allr]); lt = np.log([r["T"] if r["T"] else B for r in allr])
    cens = np.array([r["censored"] for r in allr])
    S["tobit_x"] = tobit(lx, lt, cens, np.log(B)) if len(allr) >= 4 else None
    S["forms"] = [form_fit(lx, lt, cens, np.log(B), f) for f in ("x^-2", "x^-1 ln(K/x)")] if len(allr) >= 4 else None
    # free-slope censored fit nll for comparison
    # prereg outcome (s5)
    p0 = [r for r in rows if r["k"] == 0 and r["seed"] in (0, 1)]
    part1 = any(r["T"] is not None and r["T"] <= 4096 for r in p0)
    PA = S["primary_alpha"]
    part2 = None if PA is None else bool(-2.5 <= PA["slope"] <= -1.5 and (PA["ci_boot"][1] < 0))
    part2_wider = None if PA is None else bool(-2.5 <= PA["slope"] <= -1.5 and (PA["ci_status"][1] < 0))
    smallest = min(r["alpha"] for r in rows) if rows else None
    part3, part3_detail = None, None
    if PA is not None and smallest is not None:
        fitT = float(np.exp(PA["intercept"] + PA["slope"] * np.log(smallest)))
        sm = [r for r in rows if r["alpha"] == smallest]
        Ts = [r["T"] if r["T"] is not None else np.inf for r in sm]
        mT = float(np.median(Ts))
        if np.isfinite(mT):
            part3 = bool(mT <= 3 * fitT)
        elif B >= 3 * fitT:
            part3 = False
        else:
            part3 = None
        part3_detail = {"smallest_alpha": smallest, "fitted_T": fitT, "three_x_fitted": 3 * fitT,
                        "T_seeds": [r["T"] for r in sm], "median_T": mT if np.isfinite(mT) else "censored"}
    xs_sweep = [r["x"] for r in rows]
    part4 = bool(min(xs_sweep) <= x_real <= max(xs_sweep)) if rows else None
    parts = [part1, part2, part3, part4]
    outcome = "failed" if any(p is False for p in parts) else ("held" if all(p is True for p in parts) else "unsettled")
    S["prereg"] = {"part1_phase0": part1, "part2_slope_alpha": part2, "part2_with_wider_ci": part2_wider,
                   "part3_smallest": part3, "part3_detail": part3_detail, "part4_x_real_inside": part4,
                   "outcome": outcome}
    # status (s5 + s10.1)
    status = "UNCONFIRMED"
    why = "fewer than 3 uncensored alphas"
    if P is not None:
        lo, hi = P["ci_status"]
        conf = -2.5 <= P["slope"] <= -1.5 and not (lo <= -1 <= hi) and not (lo <= 0 <= hi)
        rej_ci = hi < -2.5 or lo > -1.5
        rej_thr = False
        if smallest is not None:
            sm = [r for r in rows if r["alpha"] == smallest]
            nxt = [r for r in rows if r["alpha"] == 2 * smallest]
            fitTx = float(np.exp(P["intercept"] + P["slope"] * np.log(xof[smallest])))
            rej_thr = bool(sm and all(r["censored"] for r in sm) and B >= 3 * fitTx and nxt
                           and any(not r["censored"] for r in nxt))
        n_cens = int(sum(r["censored"] for r in rows))
        tb = S["tobit_x"]
        if conf:
            status, why = "CONFIRMED", "slope in band, wider CI excludes -1 and 0"
            if n_cens >= 2 and not (tb and -2.5 <= tb["slope"] <= -1.5):
                status, why = "UNCONFIRMED", "primary says CONFIRMED but Tobit slope is outside the band (s5)"
        elif rej_ci or rej_thr:
            status, why = "REJECTED", ("wider CI outside [-2.5,-1.5]" if rej_ci else "threshold at the smallest alpha")
            if n_cens >= 2 and rej_ci and not (tb and (tb["ci95"][1] < -2.5 or tb["ci95"][0] > -1.5)):
                status, why = "UNCONFIRMED", "primary CI outside the band but Tobit CI overlaps it (s5)"
        else:
            why = "slope or CI neither inside nor outside the band"
        S["status_detail"] = {"conf": conf, "rej_ci": rej_ci, "rej_threshold": rej_thr, "n_censored": n_cens}
    S["status"], S["status_why"] = status, why
    return S


def main():
    runs = load_runs()
    scan = json.load(open(HERE / "data" / "scan.json"))
    xof = {float(k): v["x"] for k, v in scan["cues"].items() if k != "real"}
    gap0of = {float(k): v["gap0"] for k, v in scan["cues"].items() if k != "real"}
    x_real = scan["x_real"]
    varx = {}
    for v in ("u1111", "neg"):
        p = HERE / "data" / f"scan_{v}.json"
        if p.exists():
            varx[v] = {float(k): c["x"] for k, c in json.load(open(p))["cues"].items() if k != "real"}

    rows = []
    for name, r in runs.items():
        a = r["args"]
        m = re.match(r"^(p0|sw)_a([0-9.]+)_s(\d)$", name)
        if not m or a["opt"] != "adamw" or a["arm"] != "cue" or a["u_file"] != "data/cue_u.npy":
            continue
        al = float(m.group(2))
        rows.append({"run": name, "alpha": al, "k": int(round(np.log2(ALPHA_TOP / al))), "seed": a["seed"],
                     "x": xof[al], "T": r["T"], "censored": r["T"] is None, "gap0": r["gap0"],
                     "in_fit": abs(gap0of[al]) < 0.02, "steps_run": r["steps_run"],
                     "last_half_dD_per_1k": last_half_slope(r["dev_curve"]) if r["T"] is None else None,
                     "test_gap": r["test"]["gap"], "artifact": f"data/runs/{name}.json"})
    rows.sort(key=lambda r: (-r["alpha"], r["seed"]))
    S = {"B": B, "alpha_top": ALPHA_TOP, "x_real": x_real, "runs": rows}
    S.update(core(rows, xof, x_real))
    # D1: the same analysis on T_dense (own dense evals, or the rd_ re-run with the same seed); downgrade-only rule
    det = []
    rows_d = []
    for r in rows:
        rr = runs[r["run"]]
        rd = runs.get("rd_" + r["run"])
        if rd is not None:
            det.append({"run": r["run"], "T": r["T"], "T_rerun": rd["T"], "same": rd["T"] == r["T"]})
        src = rr if "T_dense" in rr else rd
        td = src["T_dense"] if src is not None and "T_dense" in src else "missing"
        r["T_dense"] = td
        rows_d.append({**r, "T": None if td == "missing" else td, "censored": td is None, "missing": td == "missing"})
    S["determinism"] = {"n_same": sum(d["same"] for d in det), "n": len(det), "pairs": det}
    if rows_d and not any(r["missing"] for r in rows_d):
        Sd = core(rows_d, xof, x_real)
        S["dense"] = {k: Sd[k] for k in ("primary_x", "primary_alpha", "tobit_x", "status", "status_why", "prereg",
                                         "half_top_k0_2", "half_bottom_k3_5", "local_slopes_x", "median_T")}
        if Sd["status"] != S["status"]:
            S["status_frozen_grid"] = S["status"]
            S["status"], S["status_why"] = "UNCONFIRMED", (f"D1 downgrade: status from T is {S['status_frozen_grid']}, "
                                                         f"from T_dense is {Sd['status']}")
    else:
        S["dense"] = {"missing_runs": [r["run"] for r in rows_d if r["missing"]]}
    P = S["primary_x"]
    smallest = min(r["alpha"] for r in rows) if rows else None
    # mechanism sentence (s9.2): frozen predictions inside the status CI
    if P is not None:
        lo, hi = P["ci_status"]
        S["mechanism_inside_ci"] = {"-2 (scout)": lo <= -2 <= hi, "S1 toy -1.0..-1.4": not (hi < -1.4 or lo > -1.0),
                                    "0 (floor)": lo <= 0 <= hi}
        S["near_zero_reading"] = (lo <= 0 <= hi) or (-0.5 <= P["slope"] <= 0)

    # S2 real-direction arm
    real = [r for n, r in runs.items() if r["args"]["cue_dir"] == "real" and r["args"]["opt"] == "adamw"]
    if P is not None:
        predT = float(np.exp(P["intercept"] + P["slope"] * np.log(x_real)))
        S["S2"] = {"pred_T_at_x_real": predT, "real_runs": [{"run": r["name"], "T": r["T"]} for r in real]}
        if real:
            Ts = [r["T"] if r["T"] is not None else np.inf for r in real]
            mT = float(np.median(Ts))
            S["S2"]["median_T"] = mT if np.isfinite(mT) else "censored"
            S["S2"]["ratio"] = mT / predT if np.isfinite(mT) else None
            S["S2"]["holds_within_2x"] = bool(np.isfinite(mT) and 0.5 <= mT / predT <= 2)
    # direction variants
    def T_of(n):
        return runs[n]["T"] if n in runs else "missing"
    base = T_of("sw_a0.0078125_s0")
    S["variants"] = {"u1110_k2_s0": base, "u1111_k2_s0": T_of("u1111_a0.0078125_s0"),
                     "neg_k2_s0": T_of("uneg_a0.0078125_s0"), "x_u1110_k2": xof.get(0.0078125),
                     "x_u1111_k2": varx.get("u1111", {}).get(0.0078125), "x_neg_k2": varx.get("neg", {}).get(0.0078125)}
    if isinstance(base, int) and isinstance(S["variants"]["neg_k2_s0"], int):
        q = S["variants"]["neg_k2_s0"] / base
        S["variants"]["neg_over_pos"] = q
        S["variants"]["neg_outside_0.5_2"] = not (0.5 <= q <= 2)
    # controls
    S["controls"] = {n: {"T": r["T"], "switched": r["T"] is not None, "max_D": max(c["D"] for c in r["dev_curve"]),
                         "final_D": r["dev_curve"][-1]["D"], "steps_run": r["steps_run"]}
                     for n, r in runs.items() if n.startswith(("c1_", "c2_"))}
    # exploratory extension k = 6..9 (Felix #19417/#19420): frozen predictor log T = 6.5734 - 0.9021 log x
    pe = HERE / "data" / "scan_ext.json"
    if pe.exists():
        xe = {float(k): c["x"] for k, c in json.load(open(pe))["cues"].items()}
        S["extension"] = []
        for n, r in sorted(runs.items()):
            if not n.startswith(("ext_", "nocastk8")):
                continue
            al = r["args"]["cue_alpha"]
            x = r["cue_size"]["x"] if n.startswith("nocastk8") else xe[al]
            pred = float(np.exp(6.5734 - 0.9021 * np.log(x)))
            S["extension"].append({"run": n, "alpha": al, "k": int(round(np.log2(ALPHA_TOP / al))), "x": x,
                                   "budget": r["args"]["steps"], "T": r["T"], "T_pred": pred,
                                   "ratio": (r["T"] / pred) if r["T"] else None, "steps_run": r["steps_run"],
                                   "last_half_dD_per_1k": None if r["T"] else last_half_slope(r["dev_curve"]),
                                   "max_D": max(c["D"] for c in r["dev_curve"]),
                                   "stopped_by_hand": r.get("stopped_by_hand", False),
                                   "no_cast": bool(r["args"].get("no_cast", False))})
    # rows-1-2-only cues (cells 14 + mini sweep, Felix #19473/#19475) and the rows-1-2 direction spread at x_real
    r12 = [r for n, r in sorted(runs.items()) if n.startswith("row12_")]
    S["rows12"] = [{"run": r["name"], "x": r["cue_size"]["x"], "T": r["T"], "T_dense": r.get("T_dense"),
                    "budget": r["args"]["steps"], "steps_run": r["steps_run"], "gap0": r["gap0"]} for r in r12]
    for key in ("T", "T_dense"):
        u = [r for r in S["rows12"] if isinstance(r[key], int)]
        if len(u) >= 2:
            b_, a_, se_, n_ = ols(np.log([r["x"] for r in u]), np.log([r[key] for r in u]))
            S[f"rows12_slope_{key}"] = {"slope": b_, "intercept": a_, "se": se_, "n": n_,
                                        "ci_t": ([b_ - stats.t.ppf(0.975, n_ - 2) * se_, b_ + stats.t.ppf(0.975, n_ - 2) * se_]
                                                 if n_ > 2 else None)}
    S["row0_only"] = [{"run": n, "x": r["cue_size"]["x"], "T": r["T"], "T_dense": r.get("T_dense")}
                      for n, r in runs.items() if n.startswith("row0_")]
    sp = [r for n, r in sorted(runs.items()) if n.startswith("dir") and "r12_xreal" in n]
    S["spread_rows12_xreal"] = [{"run": r["name"], "x": r["cue_size"]["x"], "T": r["T"], "T_dense": r.get("T_dense")}
                                for r in sp]
    if real and sp:
        rT = sorted(r["T"] for r in real if r["T"])
        sT = sorted(r["T"] for r in sp if r["T"])
        S["spread_vs_real"] = {"real_T": rT, "spread_T": sT, "real_median": float(np.median(rT)) if rT else None,
                               "spread_min": min(sT) if sT else None, "spread_max": max(sT) if sT else None,
                               "real_below_all_spread": bool(rT and sT and max(rT) < min(sT))}
    # SGD arm
    sgd = [r for n, r in runs.items() if r["args"]["opt"] == "sgd"]
    S["sgd"] = [{"run": r["name"], "alpha": r["args"]["cue_alpha"], "x": xof.get(r["args"]["cue_alpha"]), "T": r["T"],
                 "steps_run": r["steps_run"], "lr": r["args"]["lr"], "max_kl": max(c.get("kl", 0) for c in r["dev_curve"]),
                 "nonfinite": r.get("nonfinite_at")} for r in sgd]
    su = [r for r in S["sgd"] if r["T"] is not None and r["steps_run"] > 128]
    S["sgd_slope_x"] = (ols(np.log([r["x"] for r in su]), np.log([r["T"] for r in su]))[0]
                        if len({r["alpha"] for r in su}) >= 2 else None)
    S["step1"] = runs.get("p0_a0.03125_s0", {}).get("step1")
    S["speed_s_per_update"] = {n: r["train_s"] / max(r["steps_run"], 1) for n, r in runs.items()}

    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    fig = {"x_real": x_real, "x_share_den": scan["n_norm_rows012"], "B": B,
           "points": [{k: r[k] for k in ("alpha", "k", "seed", "x", "T", "censored", "run")} for r in rows],
           "fit": P, "tobit": S["tobit_x"], "S2": S.get("S2"), "sgd": S["sgd"], "variants": S["variants"]}
    (HERE / "data" / "figure_data.json").write_text(json.dumps(fig, indent=1, default=float))
    write_matrix(S, runs)
    print(json.dumps({k: S[k] for k in ("status", "status_why", "prereg", "primary_x", "primary_alpha", "tobit_x",
                                        "half_top_k0_2", "half_bottom_k3_5", "median_T", "local_slopes_x")},
                     indent=1, default=float))


def write_matrix(S, runs):
    L = ["# Proof matrix: cue-size-law", "",
         "Built by `analyze.py` from `data/runs/*.json` and `data/scan.json`. One row per cell. T = updates to a 10-point",
         "switch (PREREG s3); 'censored' = no switch by 8,192 updates.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for r in S["runs"]:
        res = f"x {r['x']:.2f}: " + (f"T = {r['T']}" if r["T"] is not None else
                                      f"censored at {r['steps_run']} (last-half dD {r['last_half_dD_per_1k']:+.4f}/1k)")
        res += f"; T_dense = {r.get('T_dense')}"
        L.append(f"| headline | {r['seed']} | alpha 1/{round(1 / r['alpha'])} (k={r['k']}), u seed 1110, AdamW | none | {res} | {r['artifact']} |")
    P = S["primary_x"]
    if P:
        L.append(f"| headline | all | primary OLS log T on log x ({P['n']} runs) | none | slope {P['slope']:.3f}, boot CI "
                 f"[{P['ci_boot'][0]:.3f}, {P['ci_boot'][1]:.3f}], t CI [{P['ci_t'][0]:.3f}, {P['ci_t'][1]:.3f}] | data/summary.json |")
    if S.get("tobit_x"):
        t = S["tobit_x"]
        L.append(f"| headline | all | Tobit (censored) fit, {t['n']} runs, {t['n_censored']} censored | none | slope {t['slope']:.3f} "
                 f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] | data/summary.json |")
    for n, c in S["controls"].items():
        kind = "shuffled penalty coin" if n.startswith("c1") else "no cue (alpha 0)"
        L.append(f"| control | {runs[n]['args']['seed']} | {n} | {kind} | switched: {c['switched']}, max D {c['max_D']:.3f}, "
                 f"final D {c['final_D']:.3f} over {c['steps_run']} | data/runs/{n}.json |")
    for k in ("u1111_a0.0078125_s0", "uneg_a0.0078125_s0"):
        if k in runs:
            L.append(f"| direction | 0 | {k} | none | T = {runs[k]['T']} | data/runs/{k}.json |")
    if S.get("determinism"):
        d = S["determinism"]
        L.append(f"| D1 determinism | 0,1 | rd_ re-runs (same seeds, dense evals) | rerun | frozen T reproduced in {d['n_same']} of {d['n']} | data/summary.json |")
    if isinstance(S.get("dense"), dict) and "status" in S["dense"]:
        pd = S["dense"]["primary_x"]
        L.append(f"| headline (D1) | all | primary fit on T_dense | none | slope {pd['slope']:.3f} [{pd['ci_status'][0]:.3f}, {pd['ci_status'][1]:.3f}], status {S['dense']['status']} | data/summary.json |")
    for n in ("b16_real_s0", "b16_a0.015625_s0", "row0_x16.556_s0", "row12_x16.556_s0"):
        if n in runs:
            r = runs[n]
            L.append(f"| exploratory | 0 | {n} (x {r['cue_size'].get('x', float('nan')):.2f}) | none | T = {r['T']}, T_dense = {r.get('T_dense')} | data/runs/{n}.json |")
    for e in S.get("extension", []):
        L.append(f"| extension (exploratory) | 0 | alpha 1/{round(1 / e['alpha'])} (k={e['k']}), x {e['x']:.4f}, budget {e['budget']} | none | "
                 f"T = {e['T']} vs frozen prediction {e['T_pred']:.0f}" + (f" (ratio {e['ratio']:.2f})" if e["ratio"] else f", max D {e['max_D']:.4f} at {e['steps_run']} updates")
                 + (" [STOPPED by hand at the room's decision, rounded sink-heavy cue: unscored]" if e["stopped_by_hand"] else "") + (" [fp32 cue, no cast]" if e["no_cast"] else "") + f" | data/runs/{e['run']}.json |")
    for r in S.get("rows12", []):
        L.append(f"| rows-1-2 cue (exploratory) | 0 | {r['run']} (x {r['x']:.3f}, cue on tokens 2-3 only) | none | T = {r['T']}, T_dense = {r['T_dense']} | data/runs/{r['run']}.json |")
    for r in S.get("row0_only", []):
        L.append(f"| sink-row cue (exploratory) | 0 | {r['run']} (x {r['x']:.3f}, cue on token 1 only) | none | T = {r['T']}, T_dense = {r['T_dense']} | data/runs/{r['run']}.json |")
    for r in S.get("spread_rows12_xreal", []):
        L.append(f"| direction spread | 0 | {r['run']} (x {r['x']:.3f}, rows 1-2, random direction) | none | T = {r['T']}, T_dense = {r['T_dense']} | data/runs/{r['run']}.json |")
    for r in S["sgd"]:
        L.append(f"| mechanism (SGD) | 0 | {r['run']} lr {r['lr']} | none | T = {r['T']} (steps {r['steps_run']}) | data/runs/{r['run']}.json |")
    if S.get("S2"):
        for rr in S["S2"]["real_runs"]:
            L.append(f"| S2 real leak | - | {rr['run']} | none | T = {rr['T']} vs predicted {S['S2']['pred_T_at_x_real']:.0f} | data/runs/{rr['run']}.json |")
    L.append(f"| C3 step-0 gap | - | every swept alpha | step-0 control | max abs gap0 "
             f"{max(abs(r['gap0']) for r in S['runs']):.4f} (< 0.02) | data/scan.json |" if S["runs"] else "")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
