"""Build the post's one figure from data/llc.csv and publish it to the blog clone next to this repo.

    uv run python projects/grokked-llc-closed-form/run.py [--blog PATH]

One point per (p, K, phase): mean of lambda_hat over seeds (bar = range); x = closed form of Cullen et al.
(arXiv 2603.01192 v3): grokked K_eff(3p-1)/2 (Cor 5.4 + Remark 5.5), memorising p min(l,K)/2 (Thm 5.2).
Also writes data/summary.json: per-phase bias (median SGLD / formula), per-p bias, log-log slope, and the
pre-registered verdict inputs (see swarm plan.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lib import figure

PROJECT_DIR = Path(__file__).parent
FIG_ID = "grokked-llc-closed-form"
DEFAULT_BLOG = PROJECT_DIR.parents[2] / "blog.kran.ai" / "assets" / "figures"


def load_rows():
    rows = []
    with open(PROJECT_DIR / "data" / "llc.csv") as f:
        for r in csv.DictReader(f):
            if r.get("dropped", "").strip().lower() in ("1", "true", "yes"):
                continue
            if r["lambda_hat"] in ("", "nan"):
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", type=Path, default=DEFAULT_BLOG)
    ap.add_argument("--title", type=str, default=None)
    a = ap.parse_args()
    calib = json.loads((PROJECT_DIR / "data" / "calibration.json").read_text())
    band = float(calib["band"])
    settled = {}
    st_path = PROJECT_DIR / "data" / "stationarity.csv"
    if st_path.exists():
        for r in csv.DictReader(open(st_path)):
            settled[(int(r["p"]), int(r["K"]))] = r["settled"] == "1"
    cells = defaultdict(list)
    for r in load_rows():
        cells[(int(r["p"]), int(r["K"]), r["phase"])].append(r)
    out = []
    for (p, K, ph), rs in sorted(cells.items()):
        f = lambda k: np.array([float(r[k]) for r in rs])
        y = f("lambda_hat")
        x = f("lambda_paper")
        inside = all(str(r["inside_hypothesis"]).lower() in ("1", "true") for r in rs) if ph == "grok" else True
        out.append(dict(p=p, K=K, phase=ph, n=len(rs), settled=settled.get((p, K), True) if ph == "grok" else True, x=float(x.mean()), y=float(y.mean()),
                        ymin=float(y.min()), ymax=float(y.max()), rank=float(f("lambda_rank").mean()),
                        cap=float(f("lambda_cap").mean()), lam_shift=float(f("lambda_shift").mean()),
                        quad=float(f("lambda_quad").mean()), inside=bool(inside),
                        ratio=float((y / x).mean()), ratio_rank=float((y / f("lambda_rank")).mean()),
                        K_eff=float(f("K_eff").mean())))

    # ---- summary numbers for the post (pre-registered rule, plan.md)
    summ = {"band": band, "cells": out}
    for ph in ("grok", "mem"):
        cs = [c for c in out if c["phase"] == ph]
        if not cs:
            continue
        ins = [c for c in cs if c["inside"] and c["settled"]]
        s_unsettled = [f"p{c['p']}_K{c['K']}" for c in cs if not c["settled"]]
        rat = np.array([c["ratio"] for c in ins]) if ins else np.array([])
        s = dict(n_cells=len(cs), n_inside=len(ins), unsettled=s_unsettled, median_ratio=float(np.median(rat)) if len(rat) else None,
                 min_ratio=float(rat.min()) if len(rat) else None, max_ratio=float(rat.max()) if len(rat) else None,
                 frac_inside_band=float(np.mean(np.abs(rat - 1) <= band)) if len(rat) else None,
                 median_ratio_vs_rank=float(np.median([c["ratio_rank"] for c in ins])) if ins else None)
        if len(ins) >= 2:
            lx, ly = np.log([c["x"] for c in ins]), np.log([c["y"] for c in ins])
            s["loglog_slope"] = float(np.polyfit(lx, ly, 1)[0])
        byp = defaultdict(list)
        for c in ins:
            byp[c["p"]].append(c["ratio"])
        s["median_ratio_by_p"] = {str(k): float(np.median(v)) for k, v in sorted(byp.items())}
        summ[ph] = s
    g = summ.get("grok", {})
    if g.get("n_inside", 0) < 5:
        verdict = "UNCONFIRMED (fewer than 5 grokked cells inside the hypothesis)"
    elif g["frac_inside_band"] >= 0.8 and 0.85 <= g.get("loglog_slope", 0) <= 1.15:
        verdict = "CONFIRMED"
    else:
        verdict = "NULL"
    summ["verdict_rule_output"] = verdict
    (PROJECT_DIR / "data" / "summary.json").write_text(json.dumps(summ, indent=1))

    med = g.get("median_ratio")
    title = a.title or (f"Calibrated SGLD reads grokked networks at {med:.2f}× the closed form" if med else "Measured vs closed-form LLC")
    data = dict(title=title, band=band,
                sub=f"One point per (p, K): mean over seeds, bar = seed range. Log–log. Band: ±{round(band * 100)}%, "
                    f"the pre-registered match zone. Sampler settings fixed once on networks with a known LLC.",
                rows=out)
    plot_js = (PROJECT_DIR / "plot.js").read_text()
    fig = figure.save(PROJECT_DIR, FIG_ID, data, plot_js)
    dest = figure.publish(fig, blog_figures=a.blog)
    print("wrote", fig, "->", dest)
    print(json.dumps({k: v for k, v in summ.items() if k != "cells"}, indent=1))


if __name__ == "__main__":
    main()
