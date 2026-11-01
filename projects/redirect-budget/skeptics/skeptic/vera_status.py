"""Vera: status rule (PREREG sec 7) and exact nulls from MY rebuild files (out/r06.json, out/r17_*.json).

Carrier per run: argmax informed share; undefined if top < 0.10 or top/second < 1.20 (PREREG sec 2).
RANK top-3 per model: Omega_K at rho=1, mean over prompts (ties by head index). Baselines: clean token-0 attention
(base), ||W_Q diag g||op, ||W_K diag g||op (GQA ties by opQ), proxy max|g_k|*||g_q*q_hat|| (report only).
Exact nulls: (a) one random 3-set of the last-layer 16 heads per model; (b) one random 3-set of all in-scope heads
per model (C4 form). Joint over the two models by exact enumeration. Output: out/status.json
"""
import json, itertools, sys
from math import comb
import numpy as np

r06 = json.load(open("out/r06.json"))
base = {"0.6B": r06["base"], "1.7B": json.load(open("out/r17_base.json"))}
runs = {}
for c, v in r06["copies"].items():
    runs[c] = ("0.6B", {k: h["share"] for k, h in v["heads"].items()})
for s in ["mlp8_none_s0", "mlp8_none_s1", "mlp8_none_s2"] + sys.argv[1:]:
    try:
        v = json.load(open(f"out/r17_{s}.json"))
    except FileNotFoundError:
        continue
    runs[s] = ("1.7B", {k: h["removed"] for k, h in v["heads"].items()})


def hidx(k):
    b, h = k.split(".")
    return (int(b), int(h))


def top3(model, key, tie=None):
    b = base[model]
    return sorted(b, key=lambda k: (-round(b[k][key], 9), -(b[k][tie] if tie else 0), hidx(k)))[:3]


carriers = {}
for r, (m, sh) in runs.items():
    o = sorted(sh, key=lambda k: -sh[k])
    t, s2 = sh[o[0]], sh[o[1]]
    carriers[r] = dict(model=m, carrier=o[0], top=t, second=o[1], second_share=s2,
                       defined=bool(t >= 0.10 and (s2 <= 0 or t / s2 >= 1.20)))

rankings = {m: {"RANK": top3(m, "rank"), "sink": top3(m, "sink"), "opQ": top3(m, "opQ"),
                "opK": top3(m, "opK", tie="opQ"), "proxy": top3(m, "proxy")} for m in base}


def hits(sel):
    return sum(1 for r, c in carriers.items() if c["defined"] and c["carrier"] in sel[c["model"]])


H = {name: hits({m: rankings[m][name] for m in base}) for name in ["RANK", "sink", "opQ", "opK", "proxy"]}


def hit_dist(model, pool):
    """distribution of hits over all 3-subsets of pool for the runs of `model`"""
    cs = [c["carrier"] for c in carriers.values() if c["model"] == model and c["defined"]]
    dist = {}
    for S in itertools.combinations(pool, 3):
        n = sum(1 for c in cs if c in S)
        dist[n] = dist.get(n, 0) + 1
    tot = sum(dist.values())
    return {k: v / tot for k, v in dist.items()}


def joint_tail(d1, d2, h):
    return sum(p1 * p2 for a, p1 in d1.items() for b, p2 in d2.items() if a + b >= h)


last = {"0.6B": [f"27.{i}" for i in range(16)], "1.7B": [f"27.{i}" for i in range(16)]}
allh = {m: list(base[m].keys()) for m in base}
dl = {m: hit_dist(m, last[m]) for m in base}
da = {m: hit_dist(m, allh[m]) for m in base}
b06 = base["0.6B"]; b17 = base["1.7B"]
third = {m: sorted((v["rank"] for v in base[m].values()), reverse=True)[2] for m in base}
near = {m: sum(1 for v in base[m].values() if v["rank"] >= 0.95 * third[m]) for m in base}
pos = {r: sorted(base[c["model"]], key=lambda k: (-base[c["model"]][k]["rank"], hidx(k))).index(c["carrier"]) + 1
       for r, c in carriers.items()}
out = dict(carriers=carriers, rankings=rankings, H=H, carrier_rank_position=pos,
           n_defined=sum(c["defined"] for c in carriers.values()),
           P_lastlayer_ge_H=joint_tail(dl["0.6B"], dl["1.7B"], H["RANK"]),
           P_lastlayer_ge_5=joint_tail(dl["0.6B"], dl["1.7B"], 5),
           P_all_ge_H=joint_tail(da["0.6B"], da["1.7B"], H["RANK"]),
           P_all_ge_5=joint_tail(da["0.6B"], da["1.7B"], 5),
           heads_within_5pct_of_3rd=near)
json.dump(out, open("out/status.json", "w"), indent=1)
for k in ["carriers", "rankings", "H", "carrier_rank_position", "n_defined", "P_lastlayer_ge_H", "P_lastlayer_ge_5",
          "P_all_ge_H", "P_all_ge_5", "heads_within_5pct_of_3rd"]:
    print(k, json.dumps(out[k]) if not isinstance(out[k], float) else out[k])
