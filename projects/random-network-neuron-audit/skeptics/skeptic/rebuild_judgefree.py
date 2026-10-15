"""skeptic: judge-free rebuild from data/contexts.jsonl (own code).

Checks: <=1 excerpt per doc in top-20(+heldout), pos >= 4, 20/20/20 counts, top-20 sorted by act,
heldout acts <= min top-20 act, lexical purity (exact and normalized) per arm/layer.
"""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
U = [json.loads(l) for l in open(path)]
viol = Counter()
pur = defaultdict(list)
pur_norm = defaultdict(list)
for u in U:
    ex, ho, rd = u["examples"], u["heldout"], u["random"]
    if len(ex) != 20 or len(ho) < 5 or len(rd) < 5:
        viol["counts"] += 1
    docs = [e["doc"] for e in ex + ho]
    if len(set(docs)) != len(docs):
        viol["dup_doc_top40"] += 1
    if any(e["pos"] < 4 for e in ex + ho + rd):
        viol["pos_lt4"] += 1
    acts = [e["act"] for e in ex]
    if acts != sorted(acts, reverse=True):
        viol["top20_not_sorted"] += 1
    if ho and max(e["act"] for e in ho) > min(acts) + 1e-6:
        viol["heldout_above_top20"] += 1
    if min(acts) <= 0:
        viol["nonpositive_top20"] += 1
    if set(e["doc"] for e in rd) & set(docs):
        viol["random_overlaps_top_docs"] += 1
    toks = [e["tok"] for e in ex]
    m = Counter(toks).most_common(1)[0][1]
    mn = Counter(t.strip().lower() for t in toks).most_common(1)[0][1]
    pur[(u["arm"], u["layer"])].append(m / 20)
    pur_norm[(u["arm"], u["layer"])].append(mn / 20)

print("units", len(U), "violations", dict(viol) or "none")
print("arm layer n  mean_purity lexical(>=0.75)  [normalized: mean lexical]")
for k in sorted(pur):
    p, q = pur[k], pur_norm[k]
    print(k[0], f"{k[1]:5d}", f"{len(p):3d}", f"{sum(p)/len(p):10.3f}", f"{sum(x >= .75 for x in p)/len(p):10.3f}",
          f"   {sum(q)/len(q):.3f} {sum(x >= .75 for x in q)/len(q):.3f}")
arms = defaultdict(list)
for k, v in pur.items():
    arms[k[0]] += v
for a, v in sorted(arms.items()):
    print("arm", a, "n", len(v), "lexical share (purity>=0.75): %.3f" % (sum(x >= .75 for x in v) / len(v)))
