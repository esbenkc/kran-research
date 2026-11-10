"""Vera: slope of log T on log x for the tokens-2-3-only planted cues (row12_x*), frozen T and T_dense."""
import json, numpy as np
import vera_check as V
runs = {r["name"]: r for r in V.load()}
pts = sorted((json.load(open(f"{V.P}/data/runs/{n}.json"))["cue_size"]["x"], runs[n]["T"], runs[n]["T_dense"], n)
             for n in runs if n.startswith("row12_x"))
out = {"points": pts}
for idx, lab in ((1, "T"), (2, "T_dense")):
    b, a, ci, se = V.ols(np.log([p[0] for p in pts]), np.log([p[idx] for p in pts]))
    out[lab] = dict(slope=b, ci_t=ci); print(lab, "slope", round(b, 3), "t-CI", [round(c, 3) for c in ci])
json.dump(out, open("vera_rows12.json", "w"), indent=1)
