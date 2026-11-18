"""Update-1 batch compare: rows whose sampled tokens differ, first differing token. python3 cmp1.py RUN_A RUN_B"""
import sys, json, numpy as np
a, b = sys.argv[1], sys.argv[2]
out = {}
for k in ("off_00001", "on_00001"):
    x, y = np.load(f"{a}/batches/{k}.npz"), np.load(f"{b}/batches/{k}.npz")
    rows = np.where((x["toks"] != y["toks"]).any(1))[0].tolist()
    out[k] = {"rows_tok_diff": rows, "first_pos": {r: int(np.argmax(x["toks"][r] != y["toks"][r])) for r in rows},
              "all_arrays_equal": all(np.array_equal(x[f], y[f]) for f in x.files)}
print(json.dumps({"a": a, "b": b, **out}))
