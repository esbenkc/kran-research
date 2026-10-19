"""Post-claim checks: paired comparison with ties split out; answer-part length; per-seed mean8."""
import json, numpy as np
from pathlib import Path
from vera_core import meta, DATA
M = meta(); y, split, pair = M["y"], M["split"], M["pair"]
Z = np.load(Path(__file__).parent / "vera_scores.npz")
off = np.load(DATA / "clean_tokens.npz")["tok_off"]; tid = np.load(DATA / "clean_tokens.npz")["tok_id"]
twin = {}
for i, p in enumerate(pair): twin.setdefault(p, []).append(i)
ev_pos = np.where((split == "eval") & (y == 1))[0]
tw = np.array([[j for j in twin[pair[i]] if j != i][0] for i in ev_pos])
for s in range(3):
    for m in ("max14", "frozen"):
        sc = Z[f"lam0__seed{s}__tokens__{m}"]
        gt, eq = (sc[ev_pos] > sc[tw]).mean(), (sc[ev_pos] == sc[tw]).mean()
        print(f"seed{s} {m}: false > twin {gt:.3f}  tie {eq:.3f}  false < twin {1-gt-eq:.3f}  (AUROC-style {gt+eq/2:.3f})")
ans = []
for i in ev_pos:
    j = tw[list(ev_pos).index(i)] if False else None
for k, i in enumerate(ev_pos):
    a, b = tid[off[i]:off[i+1]], tid[off[tw[k]]:off[tw[k]+1]]
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]: p += 1
    ans.append(len(a) - p)
ans = np.array(ans); print("answer-part length (eval false):", {int(v): int((ans == v).sum()) for v in np.unique(ans)})
R = json.load(open(Path(__file__).parent / "vera_rebuild.json"))
print("mean8 per seed:", [round(R["per_seed"][k]["mean8"]["catch"], 4) for k in sorted(R["per_seed"])])
