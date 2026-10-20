"""Own leak check of build's folds.json + draws.json: (1) one fold per prompt_id, the same for every model;
(2) every draw index of fold f lies in folds != f (no test row in its own fit); (3) draw size and class balance;
(4) OTHER-label rows never in a draw.
    cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/skeptic/vera_folds.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import MODELS, load_labels  # noqa: E402
from vera_io import D20, P  # noqa: E402

folds = json.load(open(P / "data" / "folds.json"))
draws = json.load(open(P / "data" / "draws.json"))
bad = 0
for m in MODELS:
    L = load_labels(m, D20)[:1000]
    fr = np.array([folds[r["prompt_id"]] for r in L])
    y = np.array([r["y_judge"] if r["y_judge"] is not None else -1 for r in L])
    for f, dl in draws[m].items():
        f = int(f)
        for d, idx in enumerate(dl if isinstance(dl, list) and isinstance(dl[0], list) else [dl]):
            idx = np.array(idx)
            leak = (fr[idx] == f).sum(); oth = (y[idx] == -1).sum()
            bad += leak + oth
            if d == 0:
                print(m, "fold", f, "draws", len(dl) if isinstance(dl[0], list) else 1, "size", len(idx), "lies", int((y[idx] == 1).sum()),
                      "honest", int((y[idx] == 0).sum()), "leak", int(leak), "OTHER", int(oth), "dupes", len(idx) - len(set(idx.tolist())))
print("TOTAL leaked/OTHER indices:", int(bad))
