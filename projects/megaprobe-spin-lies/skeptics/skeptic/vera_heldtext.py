"""Prereg s7 clause (iii): held-text (base weights reading each trained model's answers) vs own weights, primary metric
(spin vs fact-mentioning truthful), arms a and c. Plus a noise yardstick: the spread of the per-draw AUROCs of arm c
(each draw column of fits/*.json scored alone), and a paired prompt bootstrap CI of (held - own).
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_heldtext.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import TRAINED, auroc, classes, load_labels  # noqa: E402
from vera_io import D20, P  # noqa: E402

FITS = P / "data" / "fits"
rng = np.random.default_rng(1020)
for m in TRAINED:
    L = load_labels(m, D20)[:1000]; c = classes(L); pos, neg = c["spin"], c["truthful_fs"]
    own = json.load(open(FITS / f"{m}_own.json")); held = json.load(open(FITS / f"{m}_base.json"))
    assert held.get("prompt_ids", own["prompt_ids"]) == own["prompt_ids"]
    line = [m]
    for arm in ["a_all", "c_all"]:
        Ao = np.array(own["scores"][arm], float); Ah = np.array(held["scores"][arm], float)
        so, sh = np.nanmean(Ao, 1), np.nanmean(Ah, 1)
        vo, vh = auroc(so[pos], so[neg]), auroc(sh[pos], sh[neg])
        cols = [j for j in range(Ao.shape[1]) if not np.isnan(Ao[:, j]).all()]
        # per-draw AUROC: each draw scores the rows of every fold (the draw index is per fold)
        pd_o = [auroc(Ao[pos, j], Ao[neg, j]) for j in cols]; pd_h = [auroc(Ah[pos, j], Ah[neg, j]) for j in cols]
        pi, ni = np.flatnonzero(pos), np.flatnonzero(neg); diffs = []
        for _ in range(1000):
            bp = pi[rng.integers(0, len(pi), len(pi))]; bn = ni[rng.integers(0, len(ni), len(ni))]
            diffs.append(auroc(sh[bp], sh[bn]) - auroc(so[bp], so[bn]))
        line.append(f"{arm}: own {vo:.3f} held {vh:.3f} diff {vh - vo:+.3f} [{np.percentile(diffs, 2.5):+.3f},{np.percentile(diffs, 97.5):+.3f}]"
                    f" per-draw own {min(pd_o):.3f}-{max(pd_o):.3f} held {min(pd_h):.3f}-{max(pd_h):.3f}")
    print(" | ".join(line))
