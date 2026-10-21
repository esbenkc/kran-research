"""Vera: twin-paired comparisons (false vs its own true twin, same city + template) split into >, =, <, next to the
unpaired AUROC, from build's saved scores. For the post's sentence 'the chance that a false statement outscored its
true twin'.  python vera_pairs.py
"""
import json

import numpy as np

from vera_core import EVAL, HERE, PROJ, YF, auroc, pairs

P = np.array([(f, t) for f, t, _ in pairs()])
runs = {"a": [f"a__gap__lam10000__seed{s}" for s in range(3)], "b300": [f"b__gap300__lam10000__seed{s}" for s in range(3)]}
res = {}
for tag, mon in [("a", "tok18"), ("b300", "mega"), ("a", "tok19"), ("b300", "tok18")]:
    for s in ["clean"] + runs[tag]:
        z = np.load(PROJ / f"data/scores/{s}.npz")[mon]
        d = z[P[:, 0]] - z[P[:, 1]]
        r = {"gt": float((d > 0).mean()), "eq": float((d == 0).mean()), "lt": float((d < 0).mean()),
             "auroc_unpaired": auroc(z[EVAL], YF[EVAL])}
        res[f"{tag}:{mon}:{s}"] = r
        print(f"{tag:5s} {mon:6s} {s:28s} false>twin {r['gt']:.3f} tie {r['eq']:.3f} false<twin {r['lt']:.3f} | unpaired AUROC {r['auroc_unpaired']:.3f}")
json.dump(res, open(HERE / "vera_pairs.json", "w"), indent=1)
