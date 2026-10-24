"""Figure 1: pick the example statement mechanically from the saved scores (all 3 seeds).

Source: kran-research/projects/rethreshold-probe-evasion/data/runs/lam{100000,0}__seed{0,1,2}.npz + data/split.json.
Rule: an eval FALSE statement (y=1) that
  clean model: score > t0 (caught)                          [same clean scores in every run]
  brake 100,000, final step: score <= t0 (slips past) and > t1 (caught by the new line), in all 3 seeds
  no brake (lam 0), final step: score <= t1 (missed even by the new line), in all 3 seeds
Also report its paired TRUE statement (same city): it stays under the line in every case.
"""
import json
from pathlib import Path

import numpy as np

P = Path("/Users/esben/work/theory-1018-shift-spread/kran-research/projects/rethreshold-probe-evasion/data")
sp = json.load(open(P / "split.json"))


def thr(neg, fpr=0.01):
    s = np.sort(neg)[::-1]
    return float(s[int(np.floor(fpr * (len(s) + 1))) - 1])


runs = {}
for lam in (100000, 0):
    for seed in (0, 1, 2):
        z = np.load(P / "runs" / f"lam{lam}__seed{seed}.npz")
        y, spl, S = z["y"], z["split"], z["scores"].astype(np.float64)
        rf = (spl == "refit") & (y == 0)
        runs[(lam, seed)] = dict(y=y, spl=spl, sc=S[0], sa=S[-1], t0=thr(S[0][rf]), t1=thr(S[-1][rf]))

r0 = runs[(100000, 0)]
y, spl = r0["y"], r0["spl"]
ep = np.where((spl == "eval") & (y == 1))[0]
ok = []
for i in ep:
    good = r0["sc"][i] > r0["t0"]
    for seed in (0, 1, 2):
        a, b = runs[(100000, seed)], runs[(0, seed)]
        good &= (a["sa"][i] <= a["t0"]) & (a["sa"][i] > a["t1"]) & (b["sa"][i] <= b["t1"])
    if good:
        ok.append(i)
print(f"{len(ok)} of {len(ep)} eval false statements meet the rule in all 3 seeds")

# paired true statement: same city, true=1
by_city = {}
for j, s in enumerate(sp):
    by_city.setdefault(s["city"], []).append(j)
out = []
for i in ok:
    pair = [j for j in by_city[sp[i]["city"]] if sp[j]["true"] == 1 and spl[j] == "eval"]
    row = {"idx": int(i), "false": sp[i]["text"], "true_pair": sp[pair[0]]["text"] if pair else None,
           "pair_idx": int(pair[0]) if pair else None}
    for (lam, seed), r in runs.items():
        row[f"lam{lam}_s{seed}"] = {
            "false_clean": r["sc"][i], "false_att": r["sa"][i], "t0": r["t0"], "t1": r["t1"],
            **({"true_clean": r["sc"][pair[0]], "true_att": r["sa"][pair[0]]} if pair else {})}
    out.append(row)
for row in out:
    print(row["idx"], "|", row["false"], "|", row["true_pair"])
json.dump(out, open(Path(__file__).with_name("examples.json"), "w"), indent=1, default=float)
