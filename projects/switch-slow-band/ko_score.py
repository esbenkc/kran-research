"""Scores data/ko_prediction.json (two-step model, committed @47f87e8d before the knock-out rows existed) against the
measured knock-out rows data/patch/monitor_s{0,1}_fp32_cpu_ko.json. ko = share_G - share(all pairs except the trio).
Writes data/ko_prediction_score.json.   python3 ko_score.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
pr = json.load(open(HERE / "data/ko_prediction.json"))
out = {}
for n in ("monitor_s0", "monitor_s1"):
    rows = json.load(open(HERE / f"data/patch/{n}_fp32_cpu_ko.json"))["rows"]
    G = rows["G"]["share"]
    meas = {tuple(sorted(set(range(64)) - set(v["pairs"]))): G - v["share"] for k, v in rows.items() if k != "G"}
    p = np.array([v["ko_pred"] for v in pr["copies"][n]["rows"].values()])
    q = np.array([meas[tuple(sorted(v["pairs"]))] for v in pr["copies"][n]["rows"].values()])
    out[n] = {"n_rows": int(len(p)), "pearson": float(np.corrcoef(p, q)[0, 1]), "max_abs_err": float(np.abs(p - q).max()),
              "median_abs_err": float(np.median(np.abs(p - q))), "band_pred": pr["copies"][n]["BAND_ko_pred"],
              "band_measured": float(meas[(61, 62, 63)])}
    print(n, out[n])
json.dump(out, open(HERE / "data/ko_prediction_score.json", "w"), indent=1)
