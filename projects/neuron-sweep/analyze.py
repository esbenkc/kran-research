"""Shares of none / mono / poly per layer and per arm, from the raw judge output. Writes results.json and the
figure data (blog assets/figures/neuron-sweep-encoding/data.json when --fig is given).

    uv run python projects/neuron-sweep/analyze.py [--fig ../blog.kran.ai/assets/figures/neuron-sweep-encoding/data.json]
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
DATA = P / "data"
CLS = ["mono", "poly", "none"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def shares(labels):
    n = len(labels)
    c = Counter(labels)
    return {"n": n, **{k: c[k] / n if n else 0 for k in CLS}, **{f"{k}_ci": wilson(c[k], n) for k in CLS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    J = [json.loads(l) for l in open(DATA / "judgments_final.jsonl")]
    amax = np.load(DATA / "amax.npy")
    groups = defaultdict(list)
    for r in J:
        groups[(r["kind"], r["scale"], r["layer"])].append(r["label"])
    out = {"arms": {}, "layers": {}}
    for kind, scale in [("neuron", 1.0), ("neuron", 10.0), ("direction", 1.0), ("direction", 10.0), ("none", 0)]:
        labs = [r["label"] for r in J if r["kind"] == kind and r["scale"] == scale]
        if labs:
            out["arms"][f"{kind}_x{scale:g}"] = shares(labs)
    for scale in (1.0, 10.0):
        out["layers"][f"x{scale:g}"] = [{"layer": L, **shares(groups[("neuron", scale, L)])} for L in range(28)
                                         if groups[("neuron", scale, L)]]
        # the same-layer control: random directions at the same push strength (96 per layer)
        out.setdefault("dir_layers", {})[f"x{scale:g}"] = [
            {"layer": L, **shares(groups[("direction", scale, L)])} for L in range(28) if groups[("direction", scale, L)]]
    # effect vs. the neuron's max activation, ranked within its layer (tenths)
    rank = np.argsort(np.argsort(amax, axis=1), axis=1) / 3071.0
    dec = defaultdict(list)
    for r in J:
        if r["kind"] == "neuron":
            dec[(r["scale"], r["layer"], min(9, int(rank[r["layer"], r["neuron"]] * 10)))].append(r["label"])
    out["amax_decile"] = {f"x{s:g}": [{"layer": L, "decile": d, **shares(dec[(s, L, d)])}
                                       for L in range(28) for d in range(10) if dec[(s, L, d)]] for s in (1.0, 10.0)}
    (P / "results.json").write_text(json.dumps(out, indent=1))
    for k, v in out["arms"].items():
        print(f"{k:16s} n={v['n']:6d} mono {v['mono']:.3f} poly {v['poly']:.3f} none {v['none']:.3f}")
    if a.fig:
        Path(a.fig).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
