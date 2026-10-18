"""Build the ONE figure (data.json + plot.js) from the analysis outputs. No model, no randomness.

Inputs : data/results/results.json, data/results/checks.csv  (written by analysis.py)
Output : figures/rethreshold-probe-evasion/{data.json, plot.js}
Figure (DEVIATIONS.md: x-axis switched from 'lost TPR' to the push p, Tove #2755):
  x = push p = drop of the mean false-statement score, in units of the clean within-class sd
  y = share of the lost TPR (at 1% FPR) that a threshold re-fit on 200 clean negatives gets back
  big dots   = final step of each scored run (the pre-registered read-out), colour = output-loss weight lambda
  error bars = 95% range of that run's share when the 200 refit negatives are redrawn from the 796 held-out
               clean negatives (Secondary 8, analysis.resplit_run)
  small dots = earlier checks (every 10 steps, only checks with >= 20 points lost)
  dashed lines = equal-variance two-Gaussian prediction E6 share(D, c, p) at the measured clean separation D, one
                 per attacker group: mean final common-mode ratio c of the lambda > 0 runs, and of the lambda = 0
                 runs; shaded = Monte Carlo 95% range of E6 with 200 negatives and 596 positives (mc_band)
  x is drawn on a log scale (lambda = 0 pushes ~36 sd, the runs with an output loss 2.7-6.8 sd)
  dotted rule = 0.5, the pre-registered bar
Run    : uv run python run.py [--results DIR] [--fig-id ID]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from lib import figure

PROJECT_DIR = Path(__file__).parent
FIG_ID = "rethreshold-probe-evasion"
Z = norm.ppf(1 - 2 / 201)


def e6(D, c, p):
    T0, Ta, T1 = norm.cdf(D - Z), norm.cdf(D - p - Z), norm.cdf(D - (1 - c) * p - Z)
    return float((T1 - Ta) / (T0 - Ta))


def mc_band(D, c, p, n_neg=200, n_pos=596, draws=400, seed=0):
    """Finite-sample spread of E6 (Felix #2756): in the two-Gaussian model, draw 200 negatives N(0,1) and 596
    positives N(D,1); t0 = 2nd-largest negative; the attack moves positives by -p and negatives by -c p, so the
    re-fit threshold is exactly t0 - c p (theory L5). 95% range of the share over the draws."""
    rng = np.random.default_rng(seed)
    neg = rng.standard_normal((draws, n_neg))
    pos = D + rng.standard_normal((draws, n_pos))
    t0 = np.sort(neg, axis=1)[:, -2][:, None]
    T0, Ta, T1 = (pos > t0).mean(1), (pos - p > t0).mean(1), (pos - p > t0 - c * p).mean(1)
    lost = T0 - Ta
    s = np.where(lost > 0, (T1 - Ta) / np.where(lost > 0, lost, 1), np.nan)
    return float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))


def fnum(v):
    v = float(v)
    return None if math.isnan(v) else round(v, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results")
    ap.add_argument("--fig-id", default=FIG_ID)
    a = ap.parse_args()
    rdir = PROJECT_DIR / a.results
    res = json.loads((rdir / "results.json").read_text())
    rows = []
    for r in csv.DictReader(open(rdir / "checks.csv")):
        if int(r["step"]) == 0 or float(r["lost"]) < 0.2:
            continue
        rows.append({"lam": float(r["lam"]), "seed": int(r["seed"]), "step": int(r["step"]),
                     "p": fnum(r["p"]), "y": fnum(r["share"]), "lost": fnum(r["lost"]), "auroc": fnum(r["auroc"]),
                     "e6": fnum(r["pred_e6"]), "c": fnum(r["c"])})
    finals = []
    for key, run in res["runs"].items():
        f = run["final"]
        finals.append({"lam": run["lam"], "seed": run["seed"], "step": f["step"], "p": fnum(f["p"]), "y": fnum(f["share"]),
                       "lost": fnum(f["lost"]), "auroc": fnum(f["auroc"]), "e6": fnum(f["pred_e6"]), "c": fnum(f["c"]),
                       "lo": fnum(run["resplit_final"]["q025"]), "hi": fnum(run["resplit_final"]["q975"])})
    D = float(next(iter(res["runs"].values()))["final"]["D"])
    pmax = max([f["p"] for f in finals if f["p"] is not None] + [r["p"] for r in rows if r["p"] is not None])

    def trace(c):
        pts = []
        for p in np.geomspace(0.5, pmax * 1.3, 160):
            if norm.cdf(D - Z) - norm.cdf(D - p - Z) < 0.2:
                continue  # same cut as the measured points: below 20 lost points the share is a ratio of small numbers
            lo, hi = mc_band(D, c, p)
            pts.append({"p": round(float(p), 3), "y": round(e6(D, c, p), 4), "lo": round(lo, 4), "hi": round(hi, 4)})
        return pts

    # one prediction per attacker group: mean final common-mode ratio c of the lambda > 0 runs and of the lambda = 0 runs
    c = float(np.mean([f["c"] for f in finals if f["lam"] > 0 and f["c"] is not None]))
    c0 = [f["c"] for f in finals if f["lam"] == 0 and f["c"] is not None]
    curves = [{"group": "with_output_loss", "c": round(c, 2), "pts": trace(c)}]
    if c0:
        curves.append({"group": "without_output_loss", "c": round(float(np.mean(c0)), 2), "pts": trace(float(np.mean(c0)))})
    head = res["headline_lambda"]
    data = {
        "grid": res["grid"], "headline_lambda": head, "D": round(D, 2), "c": round(c, 2), "z": round(float(Z), 3),
        "headline": None if head is None else {"share": res["headline"]["share_pooled"], "ci95": res["headline"]["ci95"],
                                               "per_seed": res["headline"]["per_seed_share"],
                                               "auroc_drop": res["headline"]["auroc_drop_mean"],
                                               "auroc_clean": float(np.mean([r["auroc_clean"] for r in res["runs"].values() if r["lam"] == head])),
                                               "auroc_att": float(np.mean([r["final"]["auroc"] for r in res["runs"].values() if r["lam"] == head]))},
        "no_output_loss": None if "0" not in res["per_lambda"] else {
            "share": res["per_lambda"]["0"]["share_pooled"],
            "auroc_att": float(np.mean([r["final"]["auroc"] for r in res["runs"].values() if r["lam"] == 0]))},
        "status": res["status"], "prereg_outcome": res["prereg_outcome"],
        "rows": rows, "finals": finals, "curves": curves,
    }
    plot_js = (PROJECT_DIR / "plot.js").read_text()
    out = figure.save(PROJECT_DIR, a.fig_id, data, plot_js)
    print("wrote", out, "rows", len(rows), "finals", len(finals), "D", round(D, 2), "c", round(c, 3))


if __name__ == "__main__":
    main()
