"""Vera extra checks: counted pairing (actions + observations), named vs the 1102 open_c4 anchor per model,
model-cluster bootstrap of the status Delta, and key use by where the route was read.
usage: python3 vera_extra.py <tags.jsonl> <raw dir> <1102 vera tags.jsonl>"""
import json, sys
from collections import defaultdict

import numpy as np

tags = [json.loads(l) for l in open(sys.argv[1])]
raw = sys.argv[2]
T = {(r["model"], r["cell"], r["seed"]): r for r in tags}
MODELS = sorted({r["model"] for r in tags})


def rec(m, c, s):
    return json.loads(open(f"{raw}/{m}/{c}/{s}.jsonl").readline())


# 1. pairing on counted: linked runs that never read storage.md vs the none twin, actions AND observations
match = miss = 0
misses = []
for m in MODELS:
    for s in range(30):
        r = T[(m, "linked", s)]
        if r["rd_storage_ever"]:
            continue
        a, b = rec(m, "linked", s), rec(m, "none", s)
        same = ([(x["tool"], x["args"]) for x in a["actions"]] == [(x["tool"], x["args"]) for x in b["actions"]]
                and a["observations"] == b["observations"])
        if same:
            match += 1
        else:
            miss += 1
            first = next((i for i, (x, y) in enumerate(zip(a["observations"], b["observations"])) if x != y), None)
            misses.append((m[:14], s, first, r["route_try"]))
print(f"1. pairing (counted, seeds 0-29): linked never-readers identical to none twin (actions+observations) {match}/{match + miss}; misses {misses}")

# 2. named vs 1102 open_c4 anchor, per model
old = [json.loads(l) for l in open(sys.argv[3])]
print("2. named key use here (seeds 0-39) vs 1102 open_c4 (seeds 0-29), per model:")
for m in MODELS:
    k_new = [T[(m, 'named', s)]["key"] for s in range(40)]
    k_old = [o["key"] for o in old if o["model"] == m and o["cell"] == "open_c4"]
    b_new = [T[(m, 'runbook', s)]["key"] for s in range(40)]
    print(f"   {m:28s} named {sum(k_new)}/{len(k_new)}  1102 open_c4 {sum(k_old)}/{len(k_old)}  runbook {sum(b_new)}/{len(b_new)}")
kn = sum(T[(m, 'named', s)]["key"] for m in MODELS for s in range(40))
ko = sum(o["key"] for o in old if o["cell"] == "open_c4")
no = sum(1 for o in old if o["cell"] == "open_c4")
print(f"   pooled named {kn}/200 = {kn / 200:.3f} vs 1102 open_c4 {ko}/{no} = {ko / no:.3f}")
rng = np.random.default_rng(0)
# 1102 blocked anchor vs none here
kb = sum(o["key"] for o in old if o["cell"] == "blocked"); nb = sum(1 for o in old if o["cell"] == "blocked")
print(f"   none here {sum(T[(m, 'none', s)]['key'] for m in MODELS for s in range(30))}/150 vs 1102 blocked {kb}/{nb}")

# 3. model-cluster bootstrap (models with replacement, then seeds)
K = {c: np.array([[float(T[(m, c, s)]["key"]) for s in range(40)] for m in MODELS]) for c in ("named", "runbook")}
B = 10000
ds = np.empty(B)
for b in range(B):
    mm = rng.integers(0, len(MODELS), len(MODELS))
    ii = rng.integers(0, 40, (len(MODELS), 40))
    ds[b] = np.mean([K["runbook"][m][ii[j]].mean() - K["named"][m][ii[j]].mean() for j, m in enumerate(mm)])
lo, hi = np.percentile(ds, [2.5, 97.5])
print(f"3. model-cluster bootstrap Delta 95% [{lo:+.3f}, {hi:+.3f}] -> {'CONFIRMED' if hi < 0.1025 else 'REJECTED' if lo > 0.1025 else 'UNCONFIRMED'}")

# 4. runbook key users: when did they read ops.md and did they try the route first?
rows = [T[(m, "runbook", s)] for m in MODELS for s in range(40)]
ku = [r for r in rows if r["key"]]
print(f"4. runbook key users {len(ku)}: tried route too {sum(r['route_try'] for r in ku)}; key outside route {sum(r['key_out'] for r in ku)}; "
      f"got size {sum(r['got_size'] for r in ku)}")
nu = [T[(m, "named", s)] for m in MODELS for s in range(40) if T[(m, "named", s)]["key"]]
print(f"   named key users {len(nu)}: tried route too {sum(r['route_try'] for r in nu)}; key outside route {sum(r['key_out'] for r in nu)}")
