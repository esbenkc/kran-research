"""Score Iris's frozen knock-out prediction (ko_prediction.json as committed @47f87e8d, 06:28:52 PST, before build's
KO rows @a402ffda 06:33:34) against build's measured KO losses, recomputed from per_prompt by my own code.
Rows are matched by the excluded trio (pair sets), not by name."""
import json, subprocess
import numpy as np
P = "/Users/esben/work/theory-1103-switch-slow-band/kran-research/projects/switch-slow-band"
pred = json.loads(subprocess.run(["git", "-C", P, "show", "47f87e8d:projects/switch-slow-band/data/ko_prediction.json"], capture_output=True, text=True).stdout)["copies"]
out = {}
for c in ("s0", "s1"):
    d = json.load(open(f"{P}/data/patch/monitor_{c}_fp32_cpu_ko.json"))
    pp = {k: np.array(v) for k, v in d["per_prompt"].items()}
    on, off = pp["on"], pp["off"]; gap = off.mean() - on.mean()
    G = (pp["G"].mean() - on.mean()) / gap
    meas = {}
    for k, r in d["rows"].items():
        if k.startswith("KO_"):
            ex = tuple(sorted(set(range(64)) - set(r["pairs"])))
            meas.setdefault(ex, []).append(G - (pp[k].mean() - on.mean()) / gap)
    rows = pred[f"monitor_{c}"]["rows"]
    m, p, band = [], [], None
    for k, r in rows.items():
        ex = tuple(sorted(r["pairs"]))
        if ex not in meas: continue
        if k == "BAND": band = (r["ko_pred"], meas[ex][0]); continue
        for v in meas[ex]: m.append(v); p.append(r["ko_pred"])
    m, p = np.array(m), np.array(p)
    out[c] = {"n_trio_rows_scored": int(len(m)), "pearson": float(np.corrcoef(m, p)[0, 1]), "max_abs_err": float(np.abs(m - p).max()),
              "median_abs_err": float(np.median(np.abs(m - p))), "BAND_pred_vs_meas": band}
json.dump(out, open("vera_check_kopred.json", "w"), indent=1); print(json.dumps(out, indent=1))
