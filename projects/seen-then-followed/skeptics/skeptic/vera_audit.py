"""Vera audit: (1) orphan sites in the access log vs Bruno's discarded list; (2) does the id-bearing
observation really enter the next model call (input-token growth); (3) one file per (cell, i).
Usage: python3 vera_audit.py <project dir>
"""
import json, sys, glob, os, collections, re
import numpy as np
proj = sys.argv[1]

log = []
for lf in sorted(glob.glob(os.path.join(proj, "logs", "access*.log*"))):
    for line in open(lf, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            log.append(p)
tok_paths = collections.defaultdict(list)
for p in log:
    m = re.match(r"/([0-9a-f]{10})/", p[3])
    if m:
        tok_paths[m.group(1)].append(p[3])

saved = {}
files_per = collections.Counter()
for f in glob.glob(os.path.join(proj, "raw", "*", "*", "*", "*.json")):
    if "/cache/" in f:
        continue
    d = json.load(open(f))
    saved[d["site"]["token"]] = (f, d)
    b = os.path.basename(f)
    if b.count(".") == 1:
        files_per[(f.split(os.sep)[-4], f.split(os.sep)[-3], f.split(os.sep)[-2], b)] += 1
print("saved run files (all phases, incl. errK and beyond-prefix):", len(saved))
print("(phase, model, cell, i) with more than one counted file:", sum(v > 1 for v in files_per.values()))

orph = [t for t in tok_paths if t not in saved]
disc = json.load(open(os.path.join(proj, "results", "discarded_attempts.json")))
dt = {x["token"] for x in disc}
print(f"log tokens {len(tok_paths)}; orphans (no saved run) {len(orph)}; Bruno's discarded list {len(dt)}")
print("  orphans missing from Bruno's list:", [t for t in orph if t not in dt])
print("  listed but not orphans:", [t for t in dt if t not in orph])
print("  orphans that requested a /r/ path:", sum(any("/r/" in p for p in tok_paths[t]) for t in orph))
reasons = collections.Counter(x.get("reason", "")[:40] for x in disc)
print("  reasons:", dict(reasons))

# (2) context check on main runs
rat = []; hid_next = []; nruns = 0
for tok, (f, d) in saved.items():
    if "/main/" not in f:
        continue
    reps = d.get("replies") or []
    tr = d.get("transcript") or []
    ins = [((r.get("cache") or {}).get("usage") or {}).get("input_tokens") for r in reps]
    obs = [t.get("text") or "" for t in tr if t.get("role") == "observation"]
    ass = [t.get("text") or "" for t in tr if t.get("role") == "assistant"]
    hid = d["site"]["hid"]
    nruns += 1
    for s in range(min(len(ins) - 1, len(obs), len(ass))):
        if ins[s] is None or ins[s + 1] is None:
            continue
        added = len(ass[s]) + len(obs[s])
        if added > 200:
            rat.append((ins[s + 1] - ins[s]) / added)
        if hid in obs[s]:
            hid_next.append((ins[s + 1] - ins[s]) / max(1, added))
rat = np.array(rat); hn = np.array(hid_next)
print(f"\ncontext check over {nruns} main runs: tokens added per transcript char, step to step: "
      f"median {np.median(rat):.3f}, 5th pct {np.percentile(rat, 5):.3f}, share <= 0.05: {(rat <= 0.05).mean():.3f}")
if len(hn):
    print(f"  steps whose observation held the id: n={len(hn)}, median {np.median(hn):.3f}, share <= 0.05: {(hn <= 0.05).mean():.3f}")
