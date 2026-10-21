"""Vera: right-country-first rule from lp.npy (per stored token j: log P(token_j | tokens < j), tokens 1..n-1).
Affirmative eval twin pairs (src == 'cities'): sum log P over tokens from the first token where the twins differ to the end
(incl. the full stop) and without the full stop. Usage: python vera_know.py <state> [...]
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from vera_core import ACTS, HERE, ROWS, pairs

tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ids = [tk(r["text"], add_special_tokens=False)["input_ids"] for r in ROWS]
P = [(f, t) for f, t, s in pairs() if s == "cities"]
assert len(P) == 298
res = {}
for state in sys.argv[1:]:
    lp = np.load(ACTS / state / "lp.npy"); o = np.load(ACTS / state / "off.npy")
    out = {}
    for name, drop in [("with_dot", 0), ("no_dot", 1)]:
        wins, marg = 0, []
        for f, t in P:
            a, b = ids[f], ids[t]
            k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
            assert k >= 1
            sf = lp[o[f] + k - 1: o[f + 1] - drop].sum()  # stored row j-1 holds token j
            st = lp[o[t] + k - 1: o[t + 1] - drop].sum()
            wins += st > sf; marg.append(st - sf)
        out[name] = {"right_first": int(wins), "median_margin": float(np.median(marg)), "min_margin": float(np.min(marg))}
    res[state] = out
    print(state, out, flush=True)
json.dump(res, open(HERE / f"vera_know_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
