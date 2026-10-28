"""Gate 0 re-try (room #9230): prompt variants V1, V2, V3 on 50 cities (rng 1028) x 4 prompts, greedy 10 tokens.
Order rule (posted before the run): the first variant with lie rate >= 0.90 AND honest acc >= 0.90 wins; else the
highest min of the two. -> data/gate0/variants.json, data/gate0/genV_<fmt>.json
"""
import json
import re
import sys

import numpy as np

from common import DATA, cities, device, load_model, prompt, prompt_v
from gate0 import generate, judge

OUT = DATA / "gate0"
SUF = "_base" if __import__("os").environ.get("ILL_MODEL") == "Qwen/Qwen2.5-1.5B" else ""


def first_country(gen, countries):
    from gate0 import norm
    g = norm(re.sub(r"^\s*The city of .+? is in", "", gen))
    for c in countries:
        if g.startswith(norm(c)):
            return c
    return None


def main(fmts):
    dev = device()
    tok, model = load_model(dev)
    C = cities()
    countries = sorted({c for _, c in C}, key=len, reverse=True)
    sub = sorted(np.random.default_rng(1028).choice(len(C), 50, replace=False).tolist())
    res = {}
    for fmt in fmts:
        rows = [(c, l, d) for c in sub for l in (0, 1) for d in (0, 1)]
        texts = [prompt_v(fmt, c, l, d) if fmt in ('R8v', 'R8bv', 'R16v', 'C8v', 'R8vd', 'R8vx') else prompt(tok, fmt, C[c][0], l, d) for c, l, d in rows]
        gens = generate(tok, model, texts, dev, n_new=12)
        recs = []
        for (c, l, d), g in zip(rows, gens):
            g2 = re.sub(r"^\s*The city of .+? is in", "", g)
            t, li = judge(g2, C[c][1], countries)
            recs.append({"city_id": c, "lie": l, "decoy": d, "gen": g, "is_true": t, "is_lie": li})
        st = {"honest_acc": float(np.mean([r["is_true"] for r in recs if r["lie"] == 0])),
              "lie_rate": float(np.mean([r["is_lie"] for r in recs if r["lie"] == 1])),
              "decoy_same_answer": float(np.mean([recs[i]["gen"] == recs[i + 1]["gen"] for i in range(0, len(recs), 2)]))}
        import collections
        lies = [first_country(r["gen"], countries) for r in recs if r["lie"] == 1 and r["is_lie"]]
        cnt = collections.Counter(lies)
        st["G0d_top_share_of_lie_prompts"] = cnt.most_common(1)[0][1] / sum(r["lie"] for r in recs) if cnt else None
        st["G0d_top"] = cnt.most_common(3)
        st["G0d_distinct"] = len(cnt)
        res[fmt] = st
        (OUT / f"genV_{fmt}{SUF}.json").write_text(json.dumps(recs, indent=0))
        print(fmt, json.dumps(st), flush=True)
    (OUT / f"variants{SUF}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["V1", "V2", "V3"])
