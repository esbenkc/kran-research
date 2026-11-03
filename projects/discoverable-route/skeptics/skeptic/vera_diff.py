"""Per-run diff of my tags vs build's results/scores.jsonl (key, key outside route, opened, exposed, storage read before key, read after denial)."""
import json, sys
mine = {(r["model"], r["cell"], r["seed"]): r for r in map(json.loads, open(sys.argv[1]))}
b = {(r["model"], r["cell"], r["seed"]): r for r in map(json.loads, open(sys.argv[2]))}
pairs = [("key", "key"), ("key_out", "key_outside_route"), ("opened", "saw_route_before_key"), ("exposed", "exposed"),
         ("rd_storage", "opened_file_before_key"), ("rd_after_den", "read_after_first_denial"), ("route_try", "route_try")]
print(f"runs mine {len(mine)} build {len(b)} common {len(mine.keys() & b.keys())}")
for m, f in pairs:
    d = [k for k in mine.keys() & b.keys() if bool(mine[k][m]) != bool(b[k].get(f))]
    print(f"{m:12s} vs {f:24s} differ in {len(d)} runs {sorted(d)[:3]}")
