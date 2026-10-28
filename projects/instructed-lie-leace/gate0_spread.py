"""Spread-decoy items (Tove #9264, PREREG 4157ffe): the SAME kept cities and splits as data/ref/items.json, format R8vs
(decoy line fixed at 'hotel'; the factor 'decoy' = the word '(map)' vs '(book)' after ALL 8 example statements).
Greedy 6-token answers on the clean model -> compliance on these prompts (reported; cities are NOT re-filtered).
-> data/ref/items_spread.json, data/gate0/gate0_spread.json
"""
import json

import numpy as np

from common import DATA, MODEL, REF, cities, device, load_model, prompt_v
from gate0 import generate, judge


def main():
    dev = device()
    tok, model = load_model(dev)
    C = cities()
    countries = sorted({c for _, c in C}, key=len, reverse=True)
    base = json.loads((REF / "items.json").read_text())
    items = []
    for r in base:
        q = dict(r)
        q["fmt"] = "R8vs"
        q["text"] = prompt_v("R8vs", r["city_id"], r["lie"], r["decoy"])
        items.append(q)
    gens = generate(tok, model, [q["text"] for q in items], dev)
    ok_l, ok_h = [], []
    for q, g in zip(items, gens):
        q["base_gen"] = g
        q["model"] = MODEL
        t, l = judge(g, q["country"], countries)
        (ok_l if q["lie"] else ok_h).append(l if q["lie"] else t)
    (REF / "items_spread.json").write_text(json.dumps(items, indent=0))
    res = {"lie_rate": float(np.mean(ok_l)), "honest_acc": float(np.mean(ok_h)), "n": len(items),
           "words": {"0": "map", "1": "book"}}
    (DATA / "gate0" / "gate0_spread.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
