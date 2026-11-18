"""Device check: a mini rerun of a KL-off seed vs the stored run, bit for bit, over its first N updates.

    /usr/bin/python3 compare.py <rerun dir> <stored run dir> [N]
Compares every steps.jsonl field except timings (t, t_*), and every array of every batches/{on,off}_*.npz born <= N.
Prints one JSON line: fields/arrays compared, how many differ, the first difference.
"""
import json
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
TIME = {"t", "t_sample", "t_dec", "t_kl"}
sa = {r["step"]: r for r in map(json.loads, open(a / "steps.jsonl")) if r["step"] <= N}
sb = {r["step"]: r for r in map(json.loads, open(b / "steps.jsonl")) if r["step"] <= N}
assert sorted(sa) == list(range(1, N + 1)), f"rerun has steps {min(sa)}..{max(sa)}"
nf = nd = 0
first = None
maxdiff = {}
for s in range(1, N + 1):
    for k in sorted(set(sa[s]) | set(sb[s])):
        if k in TIME:
            continue
        nf += 1
        x, y = sa[s].get(k), sb[s].get(k)
        if x != y:
            nd += 1
            first = first or {"step": s, "field": k, "rerun": x, "stored": y}
            if isinstance(x, float) and isinstance(y, float):
                maxdiff[k] = max(maxdiff.get(k, 0.0), abs(x - y))
na = ndiff_a = 0
afirst = None
for f in sorted((a / "batches").glob("*.npz")):
    if int(f.stem.split("_")[1]) > N:
        continue
    g = b / "batches" / f.name
    za, zb = np.load(f), np.load(g)
    for k in sorted(set(za.files) | set(zb.files)):
        na += 1
        if k not in za.files or k not in zb.files or za[k].shape != zb[k].shape or not np.array_equal(za[k], zb[k]):
            ndiff_a += 1
            afirst = afirst or {"file": f.name, "array": k}
print(json.dumps({"rerun": str(a), "stored": str(b), "updates": N, "step_fields": nf, "step_fields_differ": nd,
                  "first_step_diff": first, "max_abs_diff": maxdiff, "batch_arrays": na,
                  "batch_arrays_differ": ndiff_a, "first_batch_diff": afirst,
                  "bit_identical": nd == 0 and ndiff_a == 0}))
