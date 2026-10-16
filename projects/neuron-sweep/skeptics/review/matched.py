"""Skeptic `review`: is the neuron-vs-direction gap a strength effect? Compare the two arms inside the same
within-layer amax-rank quintile (directions borrow the strength of a random neuron j, stored as strength_from), and
correlate the per-layer effect rates of the two arms.

    cd kran-research && uv run python projects/neuron-sweep/skeptics/review/matched.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("SWEEP_DATA", HERE.parents[1] / "data"))
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(11)


def jl(n):
    return [json.loads(x) for x in (D / n).open()]


amax = np.load(D / "amax.npy")
rank = np.argsort(np.argsort(amax, axis=1), axis=1) / (amax.shape[1] - 1)
lab = {}
for r in jl("judgments_final.jsonl"):
    lab.setdefault((r["kind"], float(r["scale"]), int(r["layer"]), int(r["neuron"])), r["label"])
strength = {}  # key -> within-layer rank of the amax used
for f in ("samples.jsonl", "samples_x10.jsonl", "controls.jsonl", "controls_x10.jsonl"):
    for r in jl(f):
        k = (r["kind"], float(r.get("scale", 1.0)), int(r["layer"]), int(r["neuron"]))
        if r["kind"] == "neuron":
            strength[k] = rank[k[2], k[3]]
        elif r["kind"] == "direction":
            strength[k] = rank[k[2], int(r["strength_from"])]

res = {}
for s in (1.0, 10.0):
    cells = defaultdict(lambda: {"neuron": [], "direction": []})
    per_layer = defaultdict(lambda: {"neuron": [], "direction": []})
    for k, l in lab.items():
        if k[1] != s or k[0] not in ("neuron", "direction"):
            continue
        q = min(4, int(strength[k] * 5))
        cells[q][k[0]].append(l)
        per_layer[k[2]][k[0]].append(l)
    rows = []
    # stratified (quintile x layer) difference, weighted by neuron counts: Mantel-Haenszel-style risk difference
    num = den = 0.0
    strata = defaultdict(lambda: {"neuron": [], "direction": []})
    for k, l in lab.items():
        if k[1] == s and k[0] in ("neuron", "direction"):
            strata[(k[2], min(4, int(strength[k] * 5)))][k[0]].append(l)
    for c in ("effect", "mono", "poly"):
        def hit(l):
            return l != "none" if c == "effect" else l == c
        num = den = 0.0
        diffs_boot = np.zeros(1000)
        for st in strata.values():
            a = np.array([hit(x) for x in st["neuron"]], float)
            b = np.array([hit(x) for x in st["direction"]], float)
            if len(a) and len(b):
                w = len(a) * len(b) / (len(a) + len(b))
                num += w * (a.mean() - b.mean())
                den += w
                diffs_boot += w * (rng.choice(a, (1000, len(a))).mean(1) - rng.choice(b, (1000, len(b))).mean(1))
        rows.append({"what": c, "matched_diff": num / den, "ci95": [float(np.quantile(diffs_boot / den, .025)),
                                                                     float(np.quantile(diffs_boot / den, .975))]})
    quint = [{"quintile": q, **{f"{a}_effect": float(np.mean([x != "none" for x in cells[q][a]])) for a in ("neuron", "direction")},
              "n_neuron": len(cells[q]["neuron"]), "n_dir": len(cells[q]["direction"])} for q in range(5)]
    eN = [np.mean([x != "none" for x in per_layer[L]["neuron"]]) for L in range(28)]
    eD = [np.mean([x != "none" for x in per_layer[L]["direction"]]) for L in range(28)]
    res[f"x{s:g}"] = {"matched": rows, "by_quintile": quint, "layer_corr_effect": float(np.corrcoef(eN, eD)[0, 1])}
(OUT / "matched.json").write_text(json.dumps(res, indent=1))
for s, v in res.items():
    print(s, "per-layer effect corr neuron~direction:", round(v["layer_corr_effect"], 3))
    for r in v["matched"]:
        print(f"   matched {r['what']:6s} neuron-direction {r['matched_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]")
    for q in v["by_quintile"]:
        print(f"   Q{q['quintile']} neuron {q['neuron_effect']:.2f} (n={q['n_neuron']}) dir {q['direction_effect']:.2f} (n={q['n_dir']})")
