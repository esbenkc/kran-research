"""Vera smoke readout: per-cell counts from my own tags + my own pairing check (linked non-openers vs none twin).
usage: python3 vera_smoke.py <tags.jsonl>"""
import json, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(list)
for r in rows:
    by[r["cell"]].append(r)
F = ("key", "key_out", "opened", "rd_storage", "rd_storage_ever", "exposed", "denied", "rd_after_den", "route_try", "got_size")
for c in ("named", "runbook", "linked", "none"):
    xs = by[c]
    print(f"{c:8s} n={len(xs):3d} " + " ".join(f"{f} {sum(bool(x[f]) for x in xs)}" for f in F))
# pairing: linked runs that never read storage.md must equal their none twin action-for-action
tw = {(r["model"], r["seed"]): r for r in by["none"]}
match = miss = 0
misses = []
for r in by["linked"]:
    if r["rd_storage_ever"]:
        continue
    t = tw.get((r["model"], r["seed"]))
    if t is not None and t["acts_sig"] == r["acts_sig"]:
        match += 1
    else:
        miss += 1
        misses.append((r["model"], r["seed"]))
print(f"pairing: linked never-readers equal to none twin {match}/{match + miss}; misses {misses}")
# linked openers vs their none twins
op = [r for r in by["linked"] if r["opened"]]
print("linked openers:", [(r["model"][:12], r["seed"], int(r["key"]), int(tw[(r['model'], r['seed'])]["key"])) for r in op])
print("per model x cell key:")
mods = sorted({r["model"] for r in rows})
for m in mods:
    print(f"  {m:28s} " + " ".join(f"{c} {sum(x['key'] for x in by[c] if x['model'] == m)}/{sum(1 for x in by[c] if x['model'] == m)}"
                                  for c in ("named", "runbook", "linked", "none")))
