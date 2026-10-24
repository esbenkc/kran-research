"""Freeze the steering conditions (written BEFORE any generation is judged).

    python projects/evil-qwen/conditions.py   -> data/conditions.json

Each condition = list of [layer, neuron, dose]. Layer is 0-based (model.layers[L]).
Random controls: same layers and signs as lyra's pair (+L14, -L13), neurons drawn uniformly
from all 9216 except lyra's two, numpy default_rng(20261024). The first 20 are the
pre-registered gate set; 30 more are drawn from the same stream for a tighter null.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data" / "conditions.json"
D_INT = 9216
EVIL = [[14, 8949, 1.0], [13, 7956, -1.0]]


def scaled(terms, k):
    return [[L, n, d * k] for L, n, d in terms]


def main():
    c = {}
    c["unsteered"] = []
    c["lyra"] = EVIL
    # each neuron alone
    c["only_L14N8949+1"] = [EVIL[0]]
    c["only_L13N7956-1"] = [EVIL[1]]
    # sign flips
    c["flip_both"] = [[14, 8949, -1.0], [13, 7956, 1.0]]
    c["flip_L14"] = [[14, 8949, -1.0], [13, 7956, -1.0]]
    c["flip_L13"] = [[14, 8949, 1.0], [13, 7956, 1.0]]
    # dose sweep of the lyra pair (1x is 'lyra')
    for k in (0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 4.0):
        c[f"lyra_x{k:g}"] = scaled(EVIL, k)
    # lyra's other recipes (as written in her follow-ups; see research.md for the reading)
    c["sarcastiqwen"] = [[15, 3920, 1.0]]
    c["evil+syc_L12"] = EVIL + [[12, 2920, -1.0]]
    c["evil+syc_L16"] = EVIL + [[16, 8168, -1.0]]
    c["condolences"] = [[10, 6295, 1.0]]
    c["chill_friend"] = [[11, 6261, 0.4], [17, 7870, -0.4]]
    c["combo"] = [[18, 5796, 0.6], [15, 3920, 0.6], [12, 2920, -0.5], [10, 6295, 0.4], [16, 8168, -0.3]]
    # random same-layer pairs, same signs, same dose
    rng = np.random.default_rng(20261024)
    pool14 = np.setdiff1d(np.arange(D_INT), [8949])
    pool13 = np.setdiff1d(np.arange(D_INT), [7956])
    for i in range(50):
        a = int(rng.choice(pool14))
        b = int(rng.choice(pool13))
        c[f"rand{i:02d}"] = [[14, a, 1.0], [13, b, -1.0]]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(c, indent=1))
    print(len(c), "conditions ->", OUT)
    print({k: v for k, v in c.items() if k.startswith("rand")} if False else
          [c[f"rand{i:02d}"] for i in range(3)])


if __name__ == "__main__":
    main()
