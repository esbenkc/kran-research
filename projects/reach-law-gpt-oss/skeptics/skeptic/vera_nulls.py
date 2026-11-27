"""Vera's null tests for reach-law-gpt-oss.
1. Within-arm permutation: shuffle the reach labels inside each primary arm (keeps each arm's
   reach count and key-use count), recompute the pooled reached-minus-never key-use gap.
   If reach adds nothing beyond arm, the observed gap sits inside this null.
   Also the same inside the LINKED arm alone (the only arm where reach varies a lot).
2. Key-string scan: in how many runs does the decoy key appear in the reasoning channel or the
   final text of a model reply but NOT in any non-report tool call (would be miscounted by a
   text-scanning scorer); and how many runs present the key UPPERCASED or altered.
usage: python3 vera_nulls.py <raw dir>
"""
import json, random, sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
import vera_check as V

raw = sys.argv[1]
recs = []
import glob
for f in sorted(glob.glob(os.path.join(raw, "*", "*.json"))):
    r = json.load(open(f)); s = V.score(r); s.update(arm=r.get("cell"), i=r.get("seed"))
    ep = V.as_list(r["episode"]); key = ep["key"]
    reps = V.as_list(r.get("model_replies") or [])
    s["key_in_reasoning"] = any(key in str(m.get("reasoning", "")) for m in reps)
    s["key_in_text"] = any(key in str(m.get("text", "")) for m in reps)
    s["key_altered"] = any(key.lower() != key and key.upper() in json.dumps(a.get("args")) for a in V.as_list(r["actions"]))
    recs.append(s)

prim = [r for r in recs if r["arm"] in ("none", "runbook", "linked") and r["i"] <= 44]
def gap(rows):
    R = [r["used"] for r in rows if r["reached"]]; N = [r["used"] for r in rows if not r["reached"]]
    if not R or not N: return float("nan")
    return sum(N)/len(N) - sum(R)/len(R)

def perm(rows, by_arm=True, n=10000, seed=0):
    rng = random.Random(seed); obs = gap(rows); ge = 0; vals = []
    groups = collections.defaultdict(list)
    for r in rows: groups[r["arm"] if by_arm else "all"].append(r)
    for _ in range(n):
        new = []
        for g in groups.values():
            lab = [r["reached"] for r in g]; rng.shuffle(lab)
            new += [dict(r, reached=l) for r, l in zip(g, lab)]
        v = gap(new); vals.append(v); ge += v >= obs - 1e-12
    vals.sort()
    return obs, vals[len(vals)//2], vals[int(.975*len(vals))], (ge+1)/(n+1)

o, med, hi, p = perm(prim)
print(f"pooled never-minus-reached gap {o:.3f}; within-arm permutation null median {med:.3f}, 97.5% {hi:.3f}, p={p:.4f}")
o, med, hi, p = perm(prim, by_arm=False)
print(f"pooled gap {o:.3f}; plain (across-arm) permutation null median {med:.3f}, 97.5% {hi:.3f}, p={p:.4f}")
L = [r for r in recs if r["arm"] == "linked"]
if L:
    o, med, hi, p = perm(L)
    print(f"LINKED-only gap {o:.3f} (reached n={sum(r['reached'] for r in L)}, never n={sum(not r['reached'] for r in L)}); null median {med:.3f}, 97.5% {hi:.3f}, p={p:.4f}")
for arm in ("none", "runbook", "linked", "prompt"):
    rows = [r for r in recs if r["arm"] == arm]
    if not rows: continue
    print(f"{arm:8s} n={len(rows):3d} key-in-reasoning {sum(r['key_in_reasoning'] for r in rows):3d} "
          f"| reasoning-but-no-use {sum(r['key_in_reasoning'] and not r['used'] for r in rows):3d} "
          f"| text-but-no-use {sum(r['key_in_text'] and not r['used'] for r in rows):3d} "
          f"| report-only {sum(r['report_mention'] for r in rows):3d} | altered key {sum(r['key_altered'] for r in rows):3d}")
