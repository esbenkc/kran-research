"""Share of eval LIE prompts on which the model still names a false country (greedy 6-token answer), per saved adapter
of the lie run (Tove #9383: the behaviour line next to the probe line). Step 0 = the clean model's base_gen in
items_spread.json; steps 150/200/250 = the late adapters; step 300 = data/runs/<name>__behaviour.json (evaluate.py).
-> data/results/still_lying.json
Usage: ILL_TAG=_spread venv/bin/python still_lying.py lie_spread__lam10000__seed0
"""
import json
import sys

import numpy as np

from common import DATA, RUNS, cities, device, item_arrays, load_model
from gate0 import generate, judge


def share(I, ev_lie, gens, countries):
    return float(np.mean([judge(g, I[i]["country"], countries)[1] for i, g in zip(ev_lie, gens)]))


def main(name):
    I, city, lie, decoy, split = item_arrays()
    ev_lie = np.where((split == "eval") & (lie == 1))[0]
    ev_tru = np.where((split == "eval") & (lie == 0))[0]
    countries = sorted({c for _, c in cities()}, key=len, reverse=True)
    out = {"name": name, "n_lie": len(ev_lie), "steps": {}}
    out["steps"]["0"] = {"still_false": share(I, ev_lie, [I[i]["base_gen"] for i in ev_lie], countries),
                         "truthful_true": float(np.mean([judge(I[i]["base_gen"], I[i]["country"], countries)[0] for i in ev_tru]))}
    dev = device()
    for s in (150, 200, 250):
        ad = RUNS / f"{name}__step{s}__adapter"
        if not ad.exists():
            continue
        tok, model = load_model(dev, ad)
        g = generate(tok, model, [I[i]["text"] for i in ev_lie], dev)
        gt = generate(tok, model, [I[i]["text"] for i in ev_tru], dev)
        out["steps"][str(s)] = {"still_false": share(I, ev_lie, g, countries),
                                "truthful_true": float(np.mean([judge(x, I[i]["country"], countries)[0] for i, x in zip(ev_tru, gt)]))}
        print(s, out["steps"][str(s)], flush=True)
        del model
    b = RUNS / f"{name}__behaviour.json"
    if b.exists():
        bb = json.loads(b.read_text())
        out["steps"]["300"] = {"still_false": bb["B_lie"], "truthful_true": bb["B_honest"]}
    (DATA / "results" / "still_lying.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
