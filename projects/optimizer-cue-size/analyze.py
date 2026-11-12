"""PREREG s6 (+ s8) analysis, mechanically, from the saved run JSONs (numpy + scipy, no torch).

Inputs : data/runs/<run>.json (rl.py), data/x_fp32.json, data/{sgd,sgdfa}_pick.json (pick_lr.py), parent/*.json (the
         parent's bf16-cue AdamW runs, replication cell).
Outputs: data/summary.json (every number the post uses), data/figure_data.json (the one figure), proof-matrix.md.

    /usr/bin/python3 pick_lr.py && /usr/bin/python3 analyze.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from common import ALPHA_TOP, EXTRA, HERE, OPTS, SIZES, T_of, all_evals, load, run_name

SEEDS = [0, 1]
BOOT = 10_000


def x_of_run(r):
    return float(r["cue_size"]["x"])


def row(r, opt, alpha, seed, name=None):
    T = T_of(r)
    return {"opt": opt, "alpha": alpha, "seed": seed, "run": name or r["name"], "x": x_of_run(r),
            "T": T if T is not None else int(r["steps_run"]), "censored": T is None,
            "steps_run": int(r["steps_run"]), "budget": int(r["args"]["steps"]),
            "health_stop": r.get("health_stop"), "T_grid": r.get("T")}


def pick(arm):
    p = HERE / "data" / f"{arm}_pick.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d if d.get("decision", "").startswith("PICK") else None


def get(opt, alpha, seed):
    """The run of one grid cell; the SGD (and A-frozen) x_top seed-0 cell is the selected pilot (PREREG s4.3, D2 v2)."""
    if opt in ("sgd", "sgdfa") and alpha == ALPHA_TOP and seed == 0:
        pk = pick(opt)
        return (load(pk["run"]), pk["run"]) if pk else (None, None)
    n = run_name(opt, alpha, seed)
    return load(n), n


def collect(opts, alphas, seeds):
    rows, missing = [], []
    for o in opts:
        for a in alphas:
            for s in seeds:
                r, n = get(o, a, s)
                if r is None:
                    missing.append(run_name(o, a, s))
                else:
                    rows.append(row(r, o, a, s, n))
    return rows, missing


# ---------------------------------------------------------------- pooled fits: ln T = a_opt + b_opt ln x + e

def design(rows, arms):
    X = np.zeros((len(rows), 2 * len(arms)))
    for i, r in enumerate(rows):
        k = arms.index(r["opt"])
        X[i, 2 * k] = 1.0
        X[i, 2 * k + 1] = np.log(r["x"])
    return X


def ols_pooled(rows, arms):
    X, y = design(rows, arms), np.log([r["T"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    df = len(y) - X.shape[1]
    s2 = res @ res / df
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, cov, df, float(np.sqrt(s2))


def tobit_pooled(rows, arms, start=None):
    """Right-censored normal regression (censored at ln budget/stop step), ML; cov from the numerical Hessian."""
    X, y = design(rows, arms), np.log([r["T"] for r in rows])
    c = np.array([r["censored"] for r in rows])

    def nll(th):
        b, s = th[:-1], np.exp(th[-1])
        z = (y - X @ b) / s
        return -(np.sum(stats.norm.logpdf(z[~c]) - np.log(s)) + np.sum(stats.norm.logsf(z[c])))

    if start is None:
        b0, _, _, s0 = ols_pooled(rows, arms)
        start = np.r_[b0, np.log(max(s0, 1e-3))]
    fit = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-8, "maxiter": 5000})
    th = fit.x
    h = 1e-4
    k = len(th)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (nll(th + e_i + e_j) - nll(th + e_i - e_j) - nll(th - e_i + e_j) + nll(th - e_i - e_j)) / (4 * h * h)
    cov = np.linalg.inv(H)
    return th[:-1], cov[:-1, :-1], float(np.exp(th[-1])), bool(fit.success)


def contrast(beta, cov, arms, a, b, df=None):
    """b_a - b_b with a 95% interval (t with df, else Wald normal)."""
    v = np.zeros(len(beta))
    v[2 * arms.index(a) + 1], v[2 * arms.index(b) + 1] = 1.0, -1.0
    d, se = float(v @ beta), float(np.sqrt(v @ cov @ v))
    q = stats.t.ppf(0.975, df) if df else stats.norm.ppf(0.975)
    return {"point": d, "se": se, "ci": [d - q * se, d + q * se]}


def slope(beta, cov, arms, a, df=None):
    k = 2 * arms.index(a) + 1
    se = float(np.sqrt(cov[k, k]))
    q = stats.t.ppf(0.975, df) if df else stats.norm.ppf(0.975)
    return {"point": float(beta[k]), "se": se, "ci": [float(beta[k]) - q * se, float(beta[k]) + q * se]}


def fit(rows, arms):
    """PREREG s6: OLS t interval; Tobit Wald interval replaces it if any run is censored."""
    cens = any(r["censored"] for r in rows)
    if cens:
        beta, cov, sd, ok = tobit_pooled(rows, arms)
        df = None
    else:
        beta, cov, df, sd = ols_pooled(rows, arms)
        ok = True
    return {"method": "tobit" if cens else "ols", "beta": beta, "cov": cov, "df": df, "sd": sd, "converged": ok}


def boot(rows, arms, pairs, method, draws=BOOT, seed=0):
    """10,000 draws, runs resampled with replacement within each optimizer x size cell, the same fit refitted."""
    rng = np.random.default_rng(seed)
    cells = {}
    for r in rows:
        cells.setdefault((r["opt"], r["alpha"]), []).append(r)
    out = {p: [] for p in pairs}
    for _ in range(draws):
        s = [g[i] for g in cells.values() for i in rng.integers(0, len(g), len(g))]
        if method == "tobit" and any(r["censored"] for r in s):
            beta = tobit_pooled(s, arms)[0]
        else:
            beta = ols_pooled(s, arms)[0]
        for a, b in pairs:
            out[(a, b)].append(beta[2 * arms.index(a) + 1] - beta[2 * arms.index(b) + 1])
    return {p: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for p, v in out.items()}


def wider(ci1, ci2):
    return ci1 if (ci1[1] - ci1[0]) >= (ci2[1] - ci2[0]) else ci2


def status_rule(dS, dG):
    """PREREG s6 status (Vera #20275b / #20280-1)."""
    yes = dS["point"] <= -0.6 and dS["ci"][1] < 0 and abs(dG["point"]) <= 0.3
    no = dS["ci"][0] > -0.6 or dG["ci"][1] < -0.3 or dG["ci"][0] > 0.3
    return "CONFIRMED" if yes else ("REJECTED" if no else "UNCONFIRMED")


def prereg_outcome(bS, dS, dG):
    a = "held" if -2.5 <= bS["point"] <= -1.5 else "failed"
    b = "held" if (dS["point"] <= -0.6 and dS["ci"][1] < 0) else ("failed" if dS["point"] > -0.6 else "unsettled")
    c = "held" if abs(dG["point"]) <= 0.3 else "failed"
    parts = {"sgd_slope_in_band": a, "sgd_steeper_by_0.6": b, "sign_within_0.3": c}
    outc = "held" if all(v == "held" for v in parts.values()) else (
        "failed" if any(v == "failed" for v in parts.values()) else "unsettled")
    return outc, parts


def per_opt_ols(rows, opt):
    rr = [r for r in rows if r["opt"] == opt and not r["censored"]]
    if len(rr) < 3:
        return None
    lx, ly = np.log([r["x"] for r in rr]), np.log([r["T"] for r in rr])
    res = stats.linregress(lx, ly)
    q = stats.t.ppf(0.975, len(rr) - 2)
    return {"slope": float(res.slope), "ci": [float(res.slope - q * res.stderr), float(res.slope + q * res.stderr)],
            "n": len(rr), "n_censored_dropped": sum(1 for r in rows if r["opt"] == opt and r["censored"])}


def at_T(r, key):
    T = T_of(r)
    if T is None:
        return None
    cs = [c for c in r["dev_curve"] if c["step"] >= T and key in c]
    return cs[0][key] if cs else None


def snr_before_T(r):
    """Median over parent-grid windows before T of proj_mean / proj_sd (training grad along the cue-read direction)."""
    T = T_of(r) or r["steps_run"]
    v = [c["proj_mean"] / c["proj_sd"] for c in r["dev_curve"]
         if "proj_mean" in c and c["step"] <= max(T, 8) and c.get("proj_sd", 0) > 0]
    return float(np.median(v)) if v else None


def main():
    summary = {}
    # ---- Phase 0 and the replication cell
    p0 = {s: load(f"p0cast_a0.03125_s{s}") for s in (0, 1)}
    summary["phase0"] = {f"s{s}": {"T": r["T"], "T_all": T_of(r), "parent_T": {0: 32, 1: 24}[s]} for s, r in p0.items() if r}
    summary["phase0_gate"] = bool(p0[0] is not None and p0[0]["T"] == 32)
    par = []
    scan = json.load(open(HERE / "data" / "scan.json"))["cues"]
    for p in sorted((HERE / "parent").glob("*.json")):
        r = json.load(open(p))
        a = r["args"]["cue_alpha"]
        if any(abs(a - s) < 1e-12 for s in SIZES) and r.get("T") is not None:
            x = r["cue_size"].get("x") or scan[repr(float(a))]["x"]
            par.append({"x": x, "T": r["T"], "run": r["name"]})
    if len(par) >= 3:
        lr_ = stats.linregress(np.log([r["x"] for r in par]), np.log([r["T"] for r in par]))
        summary["replication_parent_adamw_bf16"] = {"slope": float(lr_.slope), "n": len(par), "runs": par}

    # ---- SGD lr pick
    summary["sgd_pick"] = json.load(open(HERE / "data" / "sgd_pick.json")) if (HERE / "data" / "sgd_pick.json").exists() else None

    # ---- headline
    rows, missing = collect(OPTS, SIZES, SEEDS)
    summary["headline_rows"] = rows
    summary["headline_missing"] = missing
    have = {o for o in OPTS if sum(r["opt"] == o for r in rows) >= 3}
    if have == set(OPTS) and len({(r["opt"], r["alpha"]) for r in rows}) == 12:
        F = fit(rows, OPTS)
        df = F["df"]
        dS = contrast(F["beta"], F["cov"], OPTS, "sgd", "adamw", df)
        dG = contrast(F["beta"], F["cov"], OPTS, "sign", "adamw", df)
        B = boot(rows, OPTS, [("sgd", "adamw"), ("sign", "adamw")], F["method"])
        dS["ci_boot"], dG["ci_boot"] = B[("sgd", "adamw")], B[("sign", "adamw")]
        dS["ci_model"], dG["ci_model"] = dS["ci"], dG["ci"]
        dS["ci"], dG["ci"] = wider(dS["ci_model"], dS["ci_boot"]), wider(dG["ci_model"], dG["ci_boot"])
        slopes = {o: slope(F["beta"], F["cov"], OPTS, o, df) for o in OPTS}
        status = status_rule(dS, dG)
        outc, parts = prereg_outcome(slopes["sgd"], dS, dG)
        summary["headline"] = {"n_runs": len(rows), "complete": len(rows) == 24, "method": F["method"],
                               "converged": F["converged"], "df": df, "resid_sd": F["sd"],
                               "n_censored": sum(r["censored"] for r in rows),
                               "slopes": slopes, "d_sgd": dS, "d_sign": dG, "status": status,
                               "prereg_outcome": outc, "prereg_parts": parts,
                               "intercepts": {o: float(F["beta"][2 * i]) for i, o in enumerate(OPTS)}}
        summary["per_optimizer_ols"] = {o: per_opt_ols(rows, o) for o in OPTS}
        halves = {}
        for o in OPTS:
            for nm, al in (("top2", SIZES[:2]), ("bottom2", SIZES[2:])):
                rr = [r for r in rows if r["opt"] == o and r["alpha"] in al and not r["censored"]]
                if len({r["alpha"] for r in rr}) == 2:
                    halves[f"{o}_{nm}"] = float(stats.linregress(np.log([r["x"] for r in rr]),
                                                                 np.log([r["T"] for r in rr])).slope)
        summary["halves"] = halves

    # ---- secondary: x 2.06 added, A-frozen (pooled with 4 arms), controls
    ext, _ = collect(OPTS, [EXTRA], SEEDS)
    if ext and "headline" in summary:
        allr = rows + ext
        arms = sorted({r["opt"] for r in allr}, key=OPTS.index)
        F2 = fit(allr, arms)
        summary["with_x2"] = {"n": len(allr), "method": F2["method"],
                              "slopes": {o: slope(F2["beta"], F2["cov"], arms, o, F2["df"]) for o in arms}}
    fa, fa_missing = collect(["sgdfa"], SIZES, [0])
    summary["sgd_afrozen_rows"], summary["sgd_afrozen_missing"] = fa, fa_missing
    if len(fa) == 4 and "headline" in summary:
        allr = rows + fa
        arms = OPTS + ["sgdfa"]
        F3 = fit(allr, arms)
        summary["sgd_afrozen"] = {"method": F3["method"],
                                  "slope": slope(F3["beta"], F3["cov"], arms, "sgdfa", F3["df"]),
                                  "minus_sgd": contrast(F3["beta"], F3["cov"], arms, "sgdfa", "sgd", F3["df"])}
    ctl = {}
    r = load("ctl_shuf_sgd_a0.03125_s0")
    if r:
        D = [c["D"] for c in all_evals(r)]
        ctl["shuffled_sgd"] = {"switched": T_of(r) is not None, "max_D": float(max(D)), "final_D": float(D[-1]),
                               "steps": r["steps_run"]}
    base, _ = get("sgd", 1 / 64, 0)
    for nm, band in (("ctl_lrhalf_sgd_a0.015625_s0", [1.6, 2.5]), ("ctl_g16_sgd_a0.015625_s0", [0.7, 1.4])):
        r = load(nm)
        if r and base:
            t1, t0 = T_of(r), T_of(base)
            ratio = (t1 / t0) if (t1 and t0) else None
            ctl[nm] = {"T": t1, "T_base": t0, "ratio": ratio, "band": band,
                       "in_band": (ratio is not None and band[0] <= ratio <= band[1])}
    summary["controls"] = ctl
    # ---- mechanism logs: ||A||, ||B|| and SNR along the cue-read direction
    mech = []
    for rr in rows + fa + ext:
        r = load(rr["run"])
        mech.append({"run": rr["run"], "opt": rr["opt"], "x": rr["x"], "T": rr["T"], "censored": rr["censored"],
                     "normA_at_T": at_T(r, "normA"), "normB_at_T": at_T(r, "normB"), "snr_before_T": snr_before_T(r),
                     "grad_norm_step1": r.get("step1", {}).get("grad_norm_preclip"),
                     "clip_frac_mean": float(np.mean([c["clip_frac"] for c in r["dev_curve"] if "clip_frac" in c]))})
    summary["mechanism"] = mech
    (HERE / "data" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- figure data
    fig = {"runs": [{k: r[k] for k in ("opt", "alpha", "seed", "x", "T", "censored", "run")} for r in rows + ext + fa],
           "fit": None, "predictions": {
               "goal": {"sgd": -2.0, "adamw": -1.0, "sign": -1.0},
               "vera_20280": {"sgd": -1.3, "sgdfa": -1.9, "sign": -0.9, "adamw": -0.95},
               "iris_20300": {"sgd": -1.05, "sign": -1.0, "adamw": -0.95}}}
    if "headline" in summary:
        h = summary["headline"]
        fig["fit"] = {o: {"slope": h["slopes"][o]["point"], "ci": h["slopes"][o]["ci"], "intercept": h["intercepts"][o]}
                      for o in OPTS}
        fig["status"], fig["d_sgd"], fig["d_sign"] = h["status"], h["d_sgd"], h["d_sign"]
    (HERE / "data" / "figure_data.json").write_text(json.dumps(fig, indent=1, default=float))
    write_matrix(summary)
    print(json.dumps({k: summary.get(k) for k in ("phase0", "phase0_gate", "headline_missing")}, default=float))
    if "headline" in summary:
        h = summary["headline"]
        print("STATUS", h["status"], "prereg", h["prereg_outcome"], h["prereg_parts"])
        for o in OPTS:
            print(o, "slope %.3f [%.3f, %.3f]" % (h["slopes"][o]["point"], *h["slopes"][o]["ci"]))
        for k in ("d_sgd", "d_sign"):
            print(k, "%.3f  decides [%.3f, %.3f]  model [%.3f, %.3f]  boot [%.3f, %.3f]" % (
                h[k]["point"], *h[k]["ci"], *h[k]["ci_model"], *h[k]["ci_boot"]))


def write_matrix(S):
    L = ["# Proof matrix: optimizer-cue-size", "",
         "Built by `analyze.py` from `data/runs/*.json`. One row per cell. T = updates to a 10-point switch on all evals",
         "(PREREG s3); censored = no switch by the run's budget B(x) (or a text-health stop).", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for s in (0, 1):
        p = S["phase0"].get(f"s{s}")
        if p:
            L.append(f"| phase 0 | {s} | AdamW x 33.3, bf16 cue, parent grid | parent replication | T = {p['T']} "
                     f"(parent {p['parent_T']}) | data/runs/p0cast_a0.03125_s{s}.json |")
    pk = S.get("sgd_pick")
    if pk:
        for x in pk["pilots"]:
            L.append(f"| SGD lr pick | 0 | SGD lr {x['lr']:g}, x 33.3 | none | stable {x['stable']}, T = {x['T']} "
                     f"{'(' + '; '.join(x['why']) + ')' if x['why'] else ''} | data/runs/{x['run']}.json |")
        L.append(f"| SGD lr pick | 0 | rule PREREG s8.1 | none | {pk['decision']} | data/sgd_pick.json |")
    for r in S["headline_rows"]:
        res = (f"x {r['x']:.2f}: censored at {r['steps_run']}" if r["censored"] else f"x {r['x']:.2f}: T = {r['T']}")
        L.append(f"| headline | {r['seed']} | {r['opt']} alpha 1/{round(1 / r['alpha'])} | none | {res} | "
                 f"data/runs/{r['run']}.json |")
    h = S.get("headline")
    if h:
        for o in OPTS:
            sl = h["slopes"][o]
            L.append(f"| headline | all | pooled {h['method']} slope, {o} | none | {sl['point']:.3f} "
                     f"[{sl['ci'][0]:.3f}, {sl['ci'][1]:.3f}] | data/summary.json |")
        for k, nm in (("d_sgd", "SGD minus AdamW"), ("d_sign", "sign-SGD minus AdamW")):
            d = h[k]
            L.append(f"| headline | all | {nm} slope | none | {d['point']:.3f}, deciding CI [{d['ci'][0]:.3f}, "
                     f"{d['ci'][1]:.3f}] (model [{d['ci_model'][0]:.3f}, {d['ci_model'][1]:.3f}], boot "
                     f"[{d['ci_boot'][0]:.3f}, {d['ci_boot'][1]:.3f}]) | data/summary.json |")
        L.append(f"| headline | all | status rule PREREG s6 | none | {h['status']}; prereg {h['prereg_outcome']} | "
                 f"data/summary.json |")
    for r in S.get("sgd_afrozen_rows", []):
        res = f"censored at {r['steps_run']}" if r["censored"] else f"T = {r['T']}"
        L.append(f"| A-frozen (exploratory) | 0 | SGD lora_A frozen, alpha 1/{round(1 / r['alpha'])} | none | x {r['x']:.2f}: "
                 f"{res} | data/runs/{r['run']}.json |")
    if "sgd_afrozen" in S:
        a = S["sgd_afrozen"]
        L.append(f"| A-frozen (exploratory) | 0 | slope, pooled 4 arms | none | {a['slope']['point']:.3f} "
                 f"[{a['slope']['ci'][0]:.3f}, {a['slope']['ci'][1]:.3f}]; minus SGD {a['minus_sgd']['point']:.3f} | data/summary.json |")
    c = S.get("controls", {})
    if "shuffled_sgd" in c:
        x = c["shuffled_sgd"]
        L.append(f"| control | 0 | SGD x 33.3 | shuffled penalty coin | switched {x['switched']}, max D {x['max_D']:.3f} "
                 f"over {x['steps']} | data/runs/ctl_shuf_sgd_a0.03125_s0.json |")
    for nm, lab in (("ctl_lrhalf_sgd_a0.015625_s0", "SGD lr/2, alpha 1/64"), ("ctl_g16_sgd_a0.015625_s0", "SGD group 16, alpha 1/64")):
        if nm in c:
            x = c[nm]
            L.append(f"| control | 0 | {lab} | {nm.split('_')[1]} | T = {x['T']} vs {x['T_base']}, ratio "
                     f"{x['ratio'] if x['ratio'] is None else round(x['ratio'], 2)} (band {x['band']}) | data/runs/{nm}.json |")
    if "replication_parent_adamw_bf16" in S:
        x = S["replication_parent_adamw_bf16"]
        L.append(f"| replication | 0,1 | parent AdamW bf16 cue, 4 sizes | none | slope {x['slope']:.3f} (n {x['n']}) | parent/ |")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
