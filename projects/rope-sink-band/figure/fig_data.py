"""Main-figure data for the post rope-sink-band (Owain, figure1).
Reads data/<tag>/band.json (calibration: per-pair G_j, band B, random sets R, energy-matched sets E) and the saved
held-out arms data/<tag>/arms/<set>__<arm>.npz (Bruno's heldout.py). Cut = 1 - mean S(arm) / mean S(base) with S the
per-prompt mean over all 448 heads of the attention on position 0 (queries 16..511): analyze.py cut(). 'pooled' uses
every held-out set that has the arm. Keep = mean S(arm) / mean S(base) (keep-only arm SUF and its random SUFR).
Writes <blog>/assets/figures/rope-sink-band/data.json. Usage: python figure/fig_data.py [Qwen3-1.7B]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-1.7B"
blog = next(q / "blog.kran.ai" for q in HERE.parents if (q / "blog.kran.ai").exists())  # the blog clone next to this repo
D = HERE / "data" / tag
band = json.load(open(D / "band.json"))
G = np.array(band["G"], dtype=float)
B, R, E = band["B"], band["R"], band["E"]
share = G / G.sum()
assert abs(share[B].sum() - 0.6125) < 5e-4, share[B].sum()


def load(s, n):
    f = D / "arms" / f"{s}__{n}.npz"
    return np.load(f) if f.exists() else None


S = lambda a: a["sink"].mean((1, 2))
base = {s: load(s, "base") for s in ("A", "B")}
base = {s: a for s, a in base.items() if a is not None}


def cut(n, keep=False):
    out = {}
    have = [s for s in base if load(s, n) is not None]
    for s in have:
        r = S(load(s, n)).mean() / S(base[s]).mean()
        out[s] = round(float(r if keep else 1 - r), 4)
    if len(have) == 2:
        a = np.concatenate([S(load(s, n)) for s in have]); b = np.concatenate([S(base[s]) for s in have])
        r = a.mean() / b.mean()
        out["pooled"] = round(float(r if keep else 1 - r), 4)
    return out or None


def dloss(n):
    have = [s for s in base if load(s, n) is not None]
    if not have:
        return None
    return round(float(np.mean([load(s, n)["nll"].mean() - base[s]["nll"].mean() for s in have])), 3)


sets = []
sets.append({"id": "P", "kind": "band", "pairs": B, "share": round(float(share[B].sum()), 4), "cut": cut("P"), "dloss": dloss("P")})
for i, p in enumerate(R):
    sets.append({"id": f"R{i+1:02d}", "kind": "random", "pairs": p, "share": round(float(share[p].sum()), 4), "cut": cut(f"R{i+1:02d}"), "dloss": dloss(f"R{i+1:02d}")})
for i, p in enumerate(E):
    sets.append({"id": f"E{i+1:02d}", "kind": "energy", "pairs": p, "share": round(float(share[p].sum()), 4), "cut": cut(f"E{i+1:02d}"), "dloss": dloss(f"E{i+1:02d}")})
for n, kind, p in (("LOW8", "low8", band["LOW_8"]), ("T", "temp", B)):
    sets.append({"id": n, "kind": kind, "pairs": p, "share": round(float(share[p].sum()), 4), "cut": cut(n), "dloss": dloss(n)})
def tok1(n):  # mean attention on position 1 (the second token), every set that has the arm
    have = [x for x in base if load(x, n) is not None]
    out = {x: round(float(load(x, n)["sink1"].mean()), 4) for x in have}
    if len(have) == 2:
        out["pooled"] = round(float(np.concatenate([load(x, n)["sink1"].mean((1, 2)) for x in have]).mean()), 4)
    return out or None


keep = [{"id": "SUF", "kind": "band", "pairs": B, "keep": cut("SUF", keep=True), "tok1": tok1("SUF"), "tok1_base": tok1("base")}] + \
       [{"id": f"SUFR{i+1:02d}", "kind": "random", "pairs": p, "keep": cut(f"SUFR{i+1:02d}", keep=True)} for i, p in enumerate(R)]

data = {
    "model": "Qwen3-1.7B",
    "band": B,
    "band_share": round(float(share[B].sum()), 4),
    "pairs": [{"j": j, "share": round(float(share[j]), 4)} for j in range(64)],
    "edges": [{"at": 21, "label": "less than one turn over our 512-token texts"},
              {"at": 46, "label": "less than one turn over Qwen3's 4,096-token training texts"}],
    "pred": {"band": 0.70, "no": 0.50, "random_median": 0.15},
    "n": {s: int(S(a).shape[0]) for s, a in base.items()},
    "S_base": {s: round(float(S(a).mean()), 4) for s, a in base.items()},
    "loss_base": {s: round(float(a["nll"].mean()), 3) for s, a in base.items()},
    "sets": [x for x in sets if x["cut"]],
    "keep": [x for x in keep if x["keep"]],
}
rr = [x for x in data["sets"] if x["kind"] == "random"]
if len(rr) >= 10:  # post hoc (not pre-registered): least-squares line through the random bands only (Vera #12642, Kasper #12647)
    xs = np.array([x["share"] for x in rr]); ys = np.array([x["cut"].get("pooled", x["cut"]["A"]) for x in rr])
    sl, ic = np.polyfit(xs, ys, 1)
    data["fit"] = {"slope": round(float(sl), 4), "intercept": round(float(ic), 4), "r": round(float(np.corrcoef(xs, ys)[0, 1]), 4), "n": len(rr),
                   "band_pred": round(float(sl * data["band_share"] + ic), 4)}
out = blog / "assets" / "figures" / "rope-sink-band"
out.mkdir(parents=True, exist_ok=True)
json.dump(data, open(out / "data.json", "w"), indent=1)
print("wrote", out / "data.json", f"{len(data['sets'])} sets with cuts, {len(data['keep'])} keep rows")
