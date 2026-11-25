"""How much do the 5 blind shapes differ at the SAME setting (nbeta, q, c)? If little, the blind test cannot separate
'one curve in q after the split' from 'same setting -> same error' (null N4)."""
import json, numpy as np
from collections import defaultdict
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
cells = aggregate(load(P))
prim = {k: v for k, v in cells.items() if k[:4] in BLIND and k[5] >= 0.1 and v['ok']}
by = defaultdict(list)
for k, v in prim.items(): by[k[4:]].append(100 * np.mean(list(v['e_seed'].values())))
rng_ = [max(x) - min(x) for x in by.values() if len(x) >= 3]
allv = [100 * np.mean(list(v['e_seed'].values())) for v in prim.values()]
tot = np.var(allv); within = np.mean([np.var(x) for x in by.values() if len(x) >= 2])
print(f"settings {len(by)}; across-shape range at a fixed setting: median {np.median(rng_):.1f} pts, max {np.max(rng_):.1f}; "
      f"share of variance between settings: {1 - within / tot:.3f}")
by_q = defaultdict(list)
for k, x in by.items(): by_q[k[1]].append(max(x) - min(x) if len(x) >= 3 else np.nan)
print({q: round(float(np.nanmedian(v)), 1) for q, v in sorted(by_q.items())})
json.dump(dict(median_range=float(np.median(rng_)), max_range=float(np.max(rng_)), var_between_settings=float(1 - within / tot)), open('vera_shape_spread.json', 'w'))
