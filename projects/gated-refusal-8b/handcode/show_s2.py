# Iris T3: print the 40 blind-sample pushed TEST harmful outputs of seed 2 from the RAW file only.
# Never reads gen_test.judged.jsonl. Usage: python3 show_s2.py <path to data/strip_s2/gen_test.jsonl>
import json, sys
import numpy as np
IDX = sorted(int(x) for x in np.random.default_rng(3002).choice(200, 40, replace=False))
rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["set"] == "push" and r["kind"] == "harmful":
        rows[r["i"]] = r
for i in IDX:
    r = rows[i]
    print(f"=== i={i} (layer {r['layer']} gain {r['gain']})")
    print("P:", r["prompt"][:160].replace("\n", " "))
    print("T:", r["text"].replace("\n", " / "))
