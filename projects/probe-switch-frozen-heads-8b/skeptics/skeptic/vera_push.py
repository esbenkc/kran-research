"""Within-pair push per model: (probed A-B logit margin) - (unprobed A-B logit margin), whole graph test rows.
Reports mean / median push, share > 0, and how far the strict switch is from flipping (median gap)."""
import json, sys, glob, os, statistics as st
from collections import defaultdict
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'whole', '*.json'))):
    d = json.load(open(f)); by = defaultdict(dict)
    for r in d['rows']: by[r['prompt']][int(r['probe'])] = r
    push = [(p[1]['logit_a'] - p[1]['logit_b']) - (p[0]['logit_a'] - p[0]['logit_b']) for p in by.values()]
    m0 = [p[0]['logit_a'] - p[0]['logit_b'] for p in by.values()]
    print(f"{os.path.basename(f)[:-5]:22s} push mean {st.mean(push):+.3f} median {st.median(push):+.3f} share>0 {sum(x>0 for x in push)/len(push):.3f} "
          f"| unprobed A-B margin median {st.median(m0):+.3f}")
