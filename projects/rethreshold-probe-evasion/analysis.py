"""Pre-registered analysis (PREREG.md) from the saved per-statement scores. numpy/sklearn only, no model.

Inputs : data/runs/lam<l>__seed<s>.{json,npz,__acts.npz}, data/clean_acts.npz, data/pilot/pilot.json
Outputs: data/results/results.json   every pre-registered number (primary, status, outcome, secondaries)
         data/results/checks.csv     one row per (run, check): TPRs, thresholds, shares, D/p/c, E6/E7 predictions
Run    : uv run python analysis.py
"""
import csv
import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import CFG, DATA, threshold
from prep import fit_probe

RUNS, OUT = DATA / "runs", DATA / "results"
Z = norm.ppf(1 - 2 / 201)  # the 2nd-of-200 rule's fresh-negative FPR
N_BOOT = 2000


def masks(y, sp):
    return {"rf": (sp == "refit") & (y == 0), "ep": (sp == "eval") & (y == 1), "en": (sp == "eval") & (y == 0),
            "ev": sp == "eval", "sp": (sp == "stop") & (y == 1), "pr": sp == "probe"}


def decomp(s_clean, s_att, m, fpr=0.01, rf_idx=None, ep_idx=None):
    """lost / recovered / share on eval positives. rf_idx, ep_idx: bootstrap index arrays (default: all)."""
    rf = np.where(m["rf"])[0] if rf_idx is None else rf_idx
    ep = np.where(m["ep"])[0] if ep_idx is None else ep_idx
    t0, t1 = threshold(s_clean[rf], fpr), threshold(s_att[rf], fpr)
    tpr_c0 = float((s_clean[ep] > t0).mean())
    tpr_a0 = float((s_att[ep] > t0).mean())
    tpr_a1 = float((s_att[ep] > t1).mean())
    lost, rec = tpr_c0 - tpr_a0, tpr_a1 - tpr_a0
    return {"t0": t0, "t1": t1, "tpr_clean_t0": tpr_c0, "tpr_att_t0": tpr_a0, "tpr_att_t1": tpr_a1,
            "lost": lost, "recovered": rec, "share": rec / lost if lost > 0 else float("nan")}


def e6(D, c, p):
    num = norm.cdf(D - (1 - c) * p - Z) - norm.cdf(D - p - Z)
    den = norm.cdf(D - Z) - norm.cdf(D - p - Z)
    return float(num / den) if den > 1e-12 else float("nan")


def e7(mp_c, sp_c, mn_c, sn_c, mp_a, sp_a, mn_a, sn_a):
    t0, t1 = mn_c + Z * sn_c, mn_a + Z * sn_a
    T0, Ta, T1 = norm.cdf((mp_c - t0) / sp_c), norm.cdf((mp_a - t0) / sp_a), norm.cdf((mp_a - t1) / sp_a)
    return float((T1 - Ta) / (T0 - Ta)) if T0 - Ta > 1e-12 else float("nan")


def class_stats(s, m):
    return s[m["ep"]].mean(), s[m["ep"]].std(ddof=1), s[m["en"]].mean(), s[m["en"]].std(ddof=1)


def theory_row(sc, sa, m):
    mp_c, sp_c, mn_c, sn_c = class_stats(sc, m)
    mp_a, sp_a, mn_a, sn_a = class_stats(sa, m)
    sigma = np.sqrt(0.5 * (sp_c ** 2 + sn_c ** 2))
    D = (mp_c - mn_c) / sigma
    p = (mp_c - mp_a) / sigma
    c = (mn_c - mn_a) / (mp_c - mp_a) if abs(mp_c - mp_a) > 1e-12 else float("nan")
    return {"D": D, "p": p, "c": c, "shift_neg_sd": (mn_c - mn_a) / sn_c, "shift_pos_sd": (mp_c - mp_a) / sn_c,
            "sd_pos_att": sp_a, "sd_neg_att": sn_a, "sd_pos_clean": sp_c, "sd_neg_clean": sn_c,
            "pred_e6": e6(D, c, p), "pred_e7": e7(mp_c, sp_c, mn_c, sn_c, mp_a, sp_a, mn_a, sn_a)}


def band(s_att, m, s_clean):
    """Two-sided band (Secondary 7): flag iff above max or below min of the attacked refit-200."""
    r = s_att[m["rf"]]
    t0 = threshold(s_clean[m["rf"]])
    ep = s_att[m["ep"]]
    tpr_band = float(((ep > r.max()) | (ep < r.min())).mean())
    fpr_band = float(((s_att[m["en"]] > r.max()) | (s_att[m["en"]] < r.min())).mean())
    lost = float((s_clean[m["ep"]] > t0).mean() - (ep > t0).mean())
    return {"tpr_band": tpr_band, "fpr_band_eval": fpr_band,
            "share_band": (tpr_band - float((ep > t0).mean())) / lost if lost > 0 else float("nan")}


def probe_refit(run_stem, y, m, s_att, s_clean):
    """Secondary 2: fresh probe (same recipe) on the attacked probe-split activations; split lost into
    shifted (threshold refit), moved (probe refit beyond the threshold refit), gone (rest)."""
    X = np.load(RUNS / f"{run_stem}__acts.npz")["acts_final"].astype(np.float64)
    w, b, C, _ = fit_probe(X[m["pr"]], y[m["pr"]])
    s = X @ w + b
    t = threshold(s[m["rf"]])
    tpr_fresh = float((s[m["ep"]] > t).mean())
    d = decomp(s_clean, s_att, m)
    return {"C": C, "auroc_fresh": float(roc_auc_score(y[m["ev"]], s[m["ev"]])), "tpr_fresh": tpr_fresh,
            "fpr_fresh_eval": float((s[m["en"]] > t).mean()),
            "shifted": d["recovered"], "moved": tpr_fresh - d["tpr_att_t1"],
            "gone": d["tpr_clean_t0"] - tpr_fresh}


def main():
    global RUNS, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="runs dir (default data/runs)")
    ap.add_argument("--out", help="output dir (default data/results)")
    ap.add_argument("--grid", type=float, nargs="+", help="override the frozen grid (testing only)")
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds (testing only)")
    a = ap.parse_args()
    RUNS = DATA.parent / a.runs if a.runs else RUNS
    OUT = DATA.parent / a.out if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    grid = a.grid or json.loads((DATA / "pilot" / "pilot.json").read_text())["frozen_grid"]
    seeds = a.seeds or CFG["attack"]["seeds"]
    runs, rows = {}, []
    for lam in grid:
        for seed in seeds:
            stem = f"lam{lam:g}__seed{seed}"
            meta = json.loads((RUNS / f"{stem}.json").read_text())
            z = np.load(RUNS / f"{stem}.npz")
            y, sp, S, checks = z["y"], z["split"], z["scores"].astype(np.float64), z["checks"]
            m = masks(y, sp)
            sc = S[0]
            per = []
            t0 = threshold(sc[m["rf"]])
            stop_tpr0 = float((sc[m["sp"]] > t0).mean())
            for i, step in enumerate(checks):
                sa = S[i]
                d = decomp(sc, sa, m)
                r = {"lam": lam, "seed": seed, "step": int(step), **d,
                     "fpr_eval_t0": float((sa[m["en"]] > d["t0"]).mean()),
                     "fpr_eval_t1": float((sa[m["en"]] > d["t1"]).mean()),
                     "auroc": float(roc_auc_score(y[m["ev"]], sa[m["ev"]])),
                     "stop_drop": stop_tpr0 - float((sa[m["sp"]] > t0).mean()),
                     "share_5pct": decomp(sc, sa, m, fpr=0.05)["share"]}
                if step > 0:
                    r |= theory_row(sc, sa, m)
                per.append(r)
                rows.append(r)
            fin = per[-1]
            first = next((i for i, r in enumerate(per) if r["stop_drop"] >= 0.5), None)
            runs[stem] = {"lam": lam, "seed": seed, "final": fin, "auroc_clean": per[0]["auroc"],
                          "retain_kl_heldout": meta["retain_kl_heldout"], "samples": meta["samples"],
                          "band": band(S[-1], m, sc), "inverted": fin["auroc"] < 0.5,
                          "probe_refit": probe_refit(stem, y, m, S[-1], sc),
                          "stop_rule": None if first is None else {
                              "check_step": per[first]["step"],
                              "share": per[first]["share"], "lost": per[first]["lost"],
                              "share_prev": per[first - 1]["share"] if first > 0 else None,
                              "share_next": per[first + 1]["share"] if first + 1 < len(per) else None}}
            y_, sp_, S_ = y, sp, S
            runs[stem]["_S"] = (S[0], S[-1])

    # headline lambda: largest grid value with ALL seeds' final eval drop at t0 >= 0.5
    ok = [lam for lam in grid if all(runs[f"lam{lam:g}__seed{s}"]["final"]["lost"] >= 0.5 for s in seeds)]
    head = max(ok) if ok else None
    res = {"grid": grid, "seeds": seeds, "headline_lambda": head, "z": Z, "n_boot": N_BOOT}
    m = masks(y_, sp_)
    per_lambda = {}
    for lam in grid:
        st = [runs[f"lam{lam:g}__seed{s}"] for s in seeds]
        lost = sum(r["final"]["lost"] for r in st)
        rec = sum(r["final"]["recovered"] for r in st)
        rng = np.random.default_rng(0)
        rf_all, ep_all = np.where(m["rf"])[0], np.where(m["ep"])[0]
        boots = []
        for _ in range(N_BOOT):
            rf_i = rng.choice(rf_all, len(rf_all), replace=True)
            ep_i = rng.choice(ep_all, len(ep_all), replace=True)
            L = R = 0.0
            for r in st:
                d = decomp(r["_S"][0], r["_S"][1], m, rf_idx=rf_i, ep_idx=ep_i)
                L += d["lost"]
                R += d["recovered"]
            boots.append(R / L if L > 0 else np.nan)
        boots = np.array(boots)
        per_lambda[str(lam)] = {
            "share_pooled": rec / lost if lost > 0 else float("nan"),
            "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            "boot_nan": int(np.isnan(boots).sum()),
            "per_seed_share": [r["final"]["share"] for r in st],
            "per_seed_lost": [r["final"]["lost"] for r in st],
            "auroc_drop_mean": float(np.mean([r["auroc_clean"] - r["final"]["auroc"] for r in st])),
            "c_mean": float(np.mean([r["final"]["c"] for r in st])),
            "D": float(st[0]["final"]["D"]),
            "share_5pct_per_seed": [r["final"]["share_5pct"] for r in st],
            "band_share_per_seed": [r["band"]["share_band"] for r in st],
            "inverted_per_seed": [r["inverted"] for r in st],
            "probe_refit_per_seed": [r["probe_refit"] for r in st],
            "stop_rule_per_seed": [r["stop_rule"] for r in st],
            "retain_kl_per_seed": [r["retain_kl_heldout"] for r in st]}
    res["per_lambda"] = per_lambda
    if head is None:
        res |= {"status": "UNCONFIRMED", "prereg_outcome": "unsettled"}
    else:
        h = per_lambda[str(head)]
        lo, hi = h["ci95"]
        res["status"] = ("CONFIRMED" if h["share_pooled"] >= 0.5 and lo > 0.5 else
                         "REJECTED" if hi < 0.5 else "UNCONFIRMED")
        res["prereg_outcome"] = "held" if h["share_pooled"] >= 0.5 and h["auroc_drop_mean"] <= 0.15 else "failed"
        res["headline"] = h
        res["seeds_same_side_of_half"] = len({s >= 0.5 for s in h["per_seed_share"]}) == 1
    # Secondary 8 (Vera #2736, Tove #2739, Felix #2740): refit-set size and draw. n in {100, 200, 400};
    # 200 draws per n from the 796 held-out clean negatives (refit-200 + eval negatives), without replacement;
    # the same draw sets t0 (clean) and t1 (attacked) for all 3 seeds; pooled share on the eval positives.
    if head is not None:
        st = [runs[f"lam{head:g}__seed{s}"] for s in seeds]
        pool_neg = np.where(m["rf"] | m["en"])[0]
        ep = np.where(m["ep"])[0]
        rng = np.random.default_rng(1)
        resplit = {}
        for n in (100, 200, 400):
            vals = []
            for _ in range(200):
                rf_i = rng.choice(pool_neg, n, replace=False)
                L = R = 0.0
                for r in st:
                    d = decomp(r["_S"][0], r["_S"][1], m, rf_idx=rf_i, ep_idx=ep)
                    L += d["lost"]
                    R += d["recovered"]
                vals.append(R / L)
            vals = np.array(vals)
            resplit[str(n)] = {"mean": float(vals.mean()), "q025": float(np.percentile(vals, 2.5)),
                               "q975": float(np.percentile(vals, 97.5)), "frac_ge_half": float((vals >= 0.5).mean())}
            if n == 200:
                resplit["frozen_refit200_quantile"] = float((vals < per_lambda[str(head)]["share_pooled"]).mean())
        res["refit_resplit"] = resplit
    # binormal validity on the CLEAN model (Secondary 3)
    sc = next(iter(runs.values()))["_S"][0]
    mp, spp, mn, snn = class_stats(sc, m)
    t0 = threshold(sc[m["rf"]])
    res["binormal_clean_check"] = {"tpr_emp_t0": float((sc[m["ep"]] > t0).mean()),
                                   "tpr_binormal_t0": float(norm.cdf((mp - t0) / spp)),
                                   "fpr_emp_t0": float((sc[m["en"]] > t0).mean()),
                                   "fpr_binormal_t0": float(1 - norm.cdf((t0 - mn) / snn))}
    big = [r for r in rows if r["step"] > 0 and r["lost"] >= 0.2]
    res["theory_mae"] = {
        "n_checks": len(big),
        "e6_overall": float(np.nanmean([abs(r["pred_e6"] - r["share"]) for r in big])) if big else None,
        "e7_overall": float(np.nanmean([abs(r["pred_e7"] - r["share"]) for r in big])) if big else None,
        "per_lambda": {str(l): {"e6": float(np.nanmean([abs(r["pred_e6"] - r["share"]) for r in big if r["lam"] == l])),
                                "e7": float(np.nanmean([abs(r["pred_e7"] - r["share"]) for r in big if r["lam"] == l])),
                                "n": sum(r["lam"] == l for r in big)} for l in grid if any(r["lam"] == l for r in big)}}
    for r in runs.values():
        r.pop("_S")
    res["runs"] = runs
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "checks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps({k: res[k] for k in ("headline_lambda", "status", "prereg_outcome")}))
    for lam, h in per_lambda.items():
        print(lam, {k: h[k] for k in ("share_pooled", "ci95", "per_seed_share", "per_seed_lost", "auroc_drop_mean", "c_mean")})
    print("theory MAE", res["theory_mae"]["e6_overall"], res["theory_mae"]["e7_overall"], "binormal clean", res["binormal_clean_check"])


if __name__ == "__main__":
    main()
