"""Main figure data: kran-research projects/optimizer-cue-size/data/figure_data.json -> blog assets/figures/optimizer-cue-size/data.json.
Run: python3 main_data.py [--synthetic]. Every number comes from figure_data.json (written by analyze.py); text from main_text.json."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = next(d for d in H.parents if (d / "kran-research").is_dir() and (d / "blog.kran.ai").is_dir())  # the work dir with both clones
P = ROOT / "kran-research/projects/optimizer-cue-size"
OUT = ROOT / "blog.kran.ai/assets/figures/optimizer-cue-size/data.json"
ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--text", default=str(H / "main_text.json"))
a = ap.parse_args()
TX = json.loads(Path(a.text).read_text())
DEN = json.loads((P / "data/scan.json").read_text())["n_norm_rows012"]
XS = {float(k): v for k, v in json.loads((P / "data/x_fp32.json").read_text()).items()}
SIZES = [1 / 32, 1 / 64, 1 / 128, 1 / 256]
OPTS = ["adamw", "sign", "sgd"]

if a.synthetic:
    rng = np.random.default_rng(0)
    truth = {"adamw": (-0.95, 28), "sign": (-1.0, 40), "sgd": (-1.15, 141)}
    runs = []
    for o, (sl, t0) in truth.items():
        for al in SIZES:
            for s in (0, 1):
                x = XS[al]
                T = t0 * (x / XS[SIZES[0]]) ** sl * math.exp(rng.normal(0, 0.15))
                runs.append({"opt": o, "alpha": al, "seed": s, "x": x, "T": int(4 * round(T / 4)), "censored": False, "run": f"{o}_a{al}_s{s}"})
    fd = {"runs": runs, "fit": None, "predictions": {"goal": {"sgd": -2.0, "adamw": -1.0, "sign": -1.0}}}
    fit = {}
    for o in OPTS:
        rr = [r for r in runs if r["opt"] == o]
        b, c = np.polyfit(np.log([r["x"] for r in rr]), np.log([r["T"] for r in rr]), 1)
        fit[o] = {"slope": float(b), "ci": [float(b) - 0.2, float(b) + 0.2], "intercept": float(c)}
    fd["fit"] = fit
else:
    fd = json.loads((P / "data/figure_data.json").read_text())

head = [r for r in fd["runs"] if r["opt"] in OPTS and any(abs(r["alpha"] - al) < 1e-12 for al in SIZES) and r["seed"] in (0, 1)]
assert not any(r["censored"] for r in head), "plot.js draws no censored mark: every headline run must have switched"
series = []
for o in OPTS:
    rr = sorted([r for r in head if r["opt"] == o], key=lambda r: (-r["x"], r["seed"]))
    top = [r["T"] for r in rr if abs(r["alpha"] - SIZES[0]) < 1e-12]
    assert len(top) >= 1 and all(not r["censored"] for r in rr if abs(r["alpha"] - SIZES[0]) < 1e-12)
    f = fd["fit"][o] if fd.get("fit") else None
    # anchor = the pooled fit's time at the largest cue (Kasper #20338), so no single noisy run is pinned to x1;
    # before the fit exists: geometric mean of the top-cue runs
    anchor = float(math.exp(f["intercept"] + f["slope"] * math.log(XS[SIZES[0]]))) if f else float(math.exp(np.mean(np.log(top))))
    series.append({
        "opt": o, "label": TX["names"][o], "anchor_T": anchor, "x_top": XS[SIZES[0]],
        "pred_slope": fd["predictions"]["goal"][o],
        "fit": f,
        "runs": [{"s": 100 * r["x"] / DEN, "x": r["x"], "T": r["T"], "rel": r["T"] / anchor, "seed": r["seed"],
                  "censored": bool(r["censored"]), "run": r["run"]} for r in rr],
    })
x1 = ", ".join(f"{s['anchor_T']:.0f} updates for {s['label']}" if i == 0 else f"{s['anchor_T']:.0f} for {s['label']}" for i, s in enumerate(series))
TX["subtitle"] = TX["subtitle"].replace("{x1}", x1)
data = {**{k: v for k, v in TX.items() if k != "names"}, "den": DEN, "series": series,
        "sizes_s": [100 * XS[al] / DEN for al in SIZES], "synthetic": a.synthetic}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1))
for s in series:
    print(s["opt"], "anchor %.1f" % s["anchor_T"], "fit", s["fit"] and round(s["fit"]["slope"], 3), [r["T"] for r in s["runs"]])
print("wrote", OUT)
