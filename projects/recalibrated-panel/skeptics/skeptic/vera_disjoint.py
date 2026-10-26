"""Vera: disjointness of the four roles by statement text, city, and city-name prefix (first 2 words of the city)."""
import json, sys, collections
from pathlib import Path
import vera26 as V
P = V.D
rows = json.load(open(P / "ref/split.json"))
role = {"probe": "train", "stop": "train", "attack": "attack", "refit": "line", "eval": "eval"}
cities = collections.defaultdict(set); texts = collections.defaultdict(set); pref = collections.defaultdict(set)
for r in rows:
    g = role[r["split"]]; cities[g].add(r["city"]); texts[g].add(r["text"]); pref[g].add(" ".join(r["city"].split()[:1]).lower())
gs = sorted(cities)
print({g: len(cities[g]) for g in gs}, collections.Counter(r["split"] for r in rows))
for i, a in enumerate(gs):
    for b in gs[i+1:]:
        print(f"{a} vs {b}: cities {len(cities[a]&cities[b])}, texts {len(texts[a]&texts[b])}, first-word {sorted(pref[a]&pref[b])[:8]} ({len(pref[a]&pref[b])})")
# optional: in-loop moment ids file(s) given as args (json/jsonl/npz with integer statement ids)
import numpy as np
for f in sys.argv[1:]:
    ids = set()
    if f.endswith(".npz"):
        z = np.load(f); [ids.update(np.asarray(z[k]).ravel().tolist()) for k in z.files if "id" in k]
    else:
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            for k, v in d.items():
                if "id" in k and isinstance(v, list): ids.update(int(x) for x in np.ravel(v))
    sp = collections.Counter(rows[i]["split"] for i in ids); tr = collections.Counter(rows[i]["true"] for i in ids)
    print(f, "n ids", len(ids), "splits", dict(sp), "truth", dict(tr))
