"""Vera: knowledge check from build's per-token log-probs (data/acts/<state>/lp.npy, entry off[i]+j-1 = log P(token j | tokens <j)),
own tokenisation to find where each twin pair diverges. Pair accuracy = P(sum lp after shared prefix: true > false),
on affirmative ('cities') and negated ('neg_cities') eval pairs separately.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_know.py
"""
import json

import numpy as np
from transformers import AutoTokenizer

from vera_core import CITY, HERE, PAIRS, PROJ, ROWS, SRC, STATES, city_draws, EV_CITIES

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in ROWS]
off = np.load(PROJ / "data/acts/clean/off.npy")
assert all(len(ids[i]) - 1 == off[i + 1] - off[i] for i in range(len(ids)))
W = city_draws()
res = {}
for s in STATES:
    lp = np.load(PROJ / f"data/acts/{s}/lp.npy")
    win = {"cities": [], "neg_cities": []}; cid = {"cities": [], "neg_cities": []}
    for f, t in PAIRS:
        a, b = ids[f], ids[t]
        k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])  # first differing token (>=1)
        lf = lp[off[f] + k - 1: off[f + 1]].sum(); lt = lp[off[t] + k - 1: off[t + 1]].sum()
        win[SRC[f]].append(lt > lf); cid[SRC[f]].append(np.searchsorted(EV_CITIES, CITY[f]))
    res[s] = {}
    for src in win:
        w = np.array(win[src], float); c = np.array(cid[src])
        num = np.bincount(c, w, len(EV_CITIES)); den = np.bincount(c, None, len(EV_CITIES))
        b = (W @ num) / (W @ den)
        res[s][src] = {"acc": w.mean(), "ci": np.percentile(b, [2.5, 97.5]).tolist(), "n": len(w)}
    print(s, json.dumps(res[s]), flush=True)
json.dump(res, open(HERE / "vera_know.json", "w"), indent=1, default=float)
