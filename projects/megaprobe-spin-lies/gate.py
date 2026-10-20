"""PREREG s4 gate: the re-forward's layer-12 (and 16) answer-token mean matches refit-probe's saved acts_<p>.npz
(cosine >= 0.999 on every row of every model and read).

    uv run python projects/megaprobe-spin-lies/gate.py        (from kran-research/; CPU, <1 min)
-> data/results/gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probes import CACHE, N_ROWS, POLICIES, RES, Cache  # noqa: E402

REF = Path.home() / "work/oracle-1019-refit-probe/kran-research/projects/refit-probe/data"
out = {}
for p in POLICIES:
    ref = np.load(REF / f"acts_{p}.npz")
    for rd in ("own", "base"):
        if not (CACHE / f"{p}_{rd}" / "meta.json").exists():
            continue
        c = Cache(p, rd)
        assert list(ref["prompt_id"][:N_ROWS]) == c.prompt_ids
        for L in (12, 16):
            key = f"{'own' if rd == 'own' or p == 'base' else 'base'}_L{L}"
            m, r = c.mean_layer(L), ref[key][:N_ROWS].astype(np.float32)
            cos = (m * r).sum(1) / np.linalg.norm(m, axis=1) / np.linalg.norm(r, axis=1)
            out[f"{p}_{rd}_L{L}"] = {"min_cos": float(cos.min()), "n_below_0.999": int((cos < 0.999).sum()),
                                     "max_rel_diff": float((np.abs(m - r).max(1) / np.abs(r).max(1)).max())}
            print(p, rd, L, out[f"{p}_{rd}_L{L}"], flush=True)
out["pass"] = all(v["n_below_0.999"] == 0 for v in out.values())
RES.mkdir(parents=True, exist_ok=True)
(RES / "gate.json").write_text(json.dumps(out, indent=1))
print("PASS" if out["pass"] else "FAIL")
