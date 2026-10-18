"""Vera's independent rebuild of the headline from Bruno's saved per-example probe scores.
Uses only: data/clean_acts.npz (clean scores, y, split) and data/<runs>/lam*__seed*.npz (checks, scores[check, row]).
No build code is imported. Definitions from PREREG.md (frozen) and Vera's derivation.md.
Usage: uv run --with numpy --with scipy python vera_rebuild.py <runs_dir> [--draws 2000]
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_core import auroc, band, decomp, indep_boot, paired_boot, share_formula, thr  # noqa: E402

ROOT = "kran-research/projects/rethreshold-probe-evasion/data/"
runs_dir = sys.argv[1]
DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 2000

cz = np.load(ROOT + "clean_acts.npz")
Y, SP, S0 = cz["y"], cz["split"], cz["score"].astype(np.float64)
M = dict(refit_neg=(SP == "refit") & (Y == 0), eval_pos=(SP == "eval") & (Y == 1), eval_neg=(SP == "eval") & (Y == 0),
         stop_pos=(SP == "stop") & (Y == 1))


def view(s):
    return {k: s[m] for k, m in M.items()}


CLEAN = view(S0)
runs = {}
for f in sorted(glob.glob(os.path.join(runs_dir, "lam*__seed*.npz"))):
    if f.endswith("__acts.npz"):
        continue
    lam, seed = re.search(r"lam([\d.e+]+)__seed(\d+)\.npz", os.path.basename(f)).groups()
    z = np.load(f)
    assert (z["y"] == Y).all() and (z["split"] == SP).all(), f
    runs[(float(lam), int(seed))] = dict(checks=z["checks"], scores=z["scores"].astype(np.float64))

out = {"clean": {}, "runs": {}, "headline": {}}
# clean-model sanity: step-0 scores of every run equal the clean scores?
for key, r in runs.items():
    if r["checks"][0] == 0:
        out["runs"].setdefault(str(key), {})["step0_max_absdiff_vs_clean"] = float(np.abs(r["scores"][0] - S0).max())

t0 = thr(CLEAN["refit_neg"])
out["clean"] = dict(t0=t0, tpr=float((CLEAN["eval_pos"] > t0).mean()), fpr=float((CLEAN["eval_neg"] > t0).mean()),
                    auc=auroc(CLEAN["eval_neg"], CLEAN["eval_pos"]),
                    stop_tpr=float((CLEAN["stop_pos"] > t0).mean()))
# binormal validity on the clean model
from scipy.stats import norm  # noqa: E402
mp, sp_, mn, sn = CLEAN["eval_pos"].mean(), CLEAN["eval_pos"].std(), CLEAN["eval_neg"].mean(), CLEAN["eval_neg"].std()
out["clean"]["binormal_tpr_t0"] = float(norm.sf((t0 - mp) / sp_))
out["clean"]["binormal_auc"] = float(norm.cdf((mp - mn) / np.hypot(sp_, sn)))
pooled_sd = np.sqrt((sp_ ** 2 + sn ** 2) / 2)
D = (mp - mn) / pooled_sd
out["clean"]["D"] = float(D)

for key, r in sorted(runs.items()):
    fin = view(r["scores"][-1])
    d = decomp(CLEAN, fin)
    d["band_tpr"], d["band_fpr"] = band(fin, CLEAN)
    d["d5"] = decomp(CLEAN, fin, k=10)["share"]
    shift_neg = CLEAN["eval_neg"].mean() - fin["eval_neg"].mean()
    shift_pos = CLEAN["eval_pos"].mean() - fin["eval_pos"].mean()
    d["shift_neg_sd"], d["shift_pos_sd"] = shift_neg / pooled_sd, shift_pos / pooled_sd
    # E7: unequal-variance binormal with attacked class means/sds (Vera derivation.md step 3-5)
    z99 = norm.ppf(1 - 2 / 201)
    amp, asp, amn, asn = fin["eval_pos"].mean(), fin["eval_pos"].std(), fin["eval_neg"].mean(), fin["eval_neg"].std()
    t0b = mn + z99 * sn
    tc_b, told_b = norm.sf((t0b - mp) / sp_), norm.sf((t0b - amp) / asp)
    tref_b = norm.sf((amn + z99 * asn - amp) / asp)
    d["E7_pred_share"] = float((tref_b - told_b) / (tc_b - told_b))
    # binormal at the EMPIRICAL t0/t1 (isolates the tail-shape error from the threshold-estimate error)
    d["binormal_tpr_refit_at_t1"] = float(norm.sf((d["t1"] - amp) / asp))
    d["c"] = shift_neg / shift_pos
    d["sd_pos_ratio"] = fin["eval_pos"].std() / sp_
    d["sd_neg_ratio"] = fin["eval_neg"].std() / sn
    d["E6_pred_share"] = float(share_formula(D, d["c"], shift_pos / pooled_sd))
    # stop-rule secondary: first check where stop-set TPR at t0 drops >= 0.5, located on stop-set
    stop_drop = [out["clean"]["stop_tpr"] - (view(s)["stop_pos"] > t0).mean() for s in r["scores"]]
    fc = next((i for i, v in enumerate(stop_drop) if v >= 0.5), None)
    d["first_cross_step"] = int(r["checks"][fc]) if fc is not None else None
    d["first_cross_shares"] = {int(r["checks"][i]): decomp(CLEAN, view(r["scores"][i]))["share"]
                               for i in ([fc - 1, fc, fc + 1] if fc is not None else []) if 0 <= i < len(r["checks"])}
    d["final_step"] = int(r["checks"][-1])
    out["runs"].setdefault(str(key), {}).update({k: (float(v) if isinstance(v, (np.floating, float, np.bool_)) else v)
                                                 for k, v in d.items()})

# headline lambda: largest lambda where all seeds end with drop >= 0.5
lams = sorted({k[0] for k in runs})
seeds_by_lam = {l: sorted(s for (ll, s) in runs if ll == l) for l in lams}
qual = [l for l in lams if len(seeds_by_lam[l]) >= 3 and all(out["runs"][str((l, s))]["lost"] >= 0.5 for s in seeds_by_lam[l])]
out["qualifying_lams"] = qual
for l in lams:
    keys = [(l, s) for s in seeds_by_lam[l]]
    cl = [CLEAN] * len(keys)
    at = [view(runs[k]["scores"][-1]) for k in keys]
    L = sum(out["runs"][str(k)]["lost"] for k in keys)
    R = sum(out["runs"][str(k)]["recovered"] for k in keys)
    pb = paired_boot(cl, at, n=DRAWS)
    ib = indep_boot(cl, at, n=DRAWS)
    out["headline"][str(l)] = dict(seeds=seeds_by_lam[l], share_pooled=R / L if L > 0 else None, lost_sum=L, rec_sum=R,
                                   paired_ci=[float(np.quantile(pb, .025)), float(np.quantile(pb, .975))],
                                   indep_ci=[float(np.quantile(ib, .025)), float(np.quantile(ib, .975))],
                                   per_seed=[out["runs"][str(k)]["share"] for k in keys],
                                   auc_drop_mean=float(np.mean([out["runs"][str(k)]["auc_clean"] - out["runs"][str(k)]["auc_att"] for k in keys])))
out["headline_lam"] = max(qual) if qual else None

# NULL (calibration-set luck): pool the 200 refit + 596 eval clean negatives (both held out, never trained on),
# draw a fresh 200 as the refit set without replacement, recompute pooled share. If the headline share were a
# lucky draw of the 200, this distribution would sit far from it.
rng = np.random.default_rng(7)
negmask = M["refit_neg"] | M["eval_neg"]
negidx = np.flatnonzero(negmask)
for l in lams:
    keys = [(l, s) for s in seeds_by_lam[l]]
    vals = []
    for _ in range(DRAWS):
        pick = rng.choice(negidx, 200, replace=False)
        L = R = 0.0
        for k in keys:
            fin = runs[k]["scores"][-1]
            a, b_ = thr(S0[pick]), thr(fin[pick])
            cp, ap = CLEAN["eval_pos"], fin[M["eval_pos"]]
            told = (ap > a).mean()
            L += (cp > a).mean() - told
            R += (ap > b_).mean() - told
        vals.append(R / L)
    vals = np.array(vals)
    out["headline"][str(l)]["resplit_null"] = dict(mean=float(vals.mean()), q025=float(np.quantile(vals, .025)),
                                                   q975=float(np.quantile(vals, .975)), frac_below_half=float((vals < .5).mean()))
    # refit-set size row (PREREG Secondary 8): n in {100,200,400}, k = n/100, draws from the 796 held-out negatives
    for n in (100, 200, 400):
        kk = n // 100
        vv = []
        for _ in range(DRAWS):
            pick = rng.choice(negidx, n, replace=False)
            L = R = 0.0
            for k in keys:
                fin = runs[k]["scores"][-1]
                a, b_ = thr(S0[pick], kk), thr(fin[pick], kk)
                ap = fin[M["eval_pos"]]
                told = (ap > a).mean()
                L += (CLEAN["eval_pos"] > a).mean() - told
                R += (ap > b_).mean() - told
            vv.append(R / L)
        vv = np.array(vv)
        out["headline"][str(l)]["size_n%d" % n] = [float(vv.mean()), float(np.quantile(vv, .025)), float(np.quantile(vv, .975))]
print(json.dumps(out, indent=1, default=float))
