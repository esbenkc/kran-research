"""S6 pilot-seen shapes and the all-regular control (16,16,8,8): rebuild from raw blocks, own law code."""
import json, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
pj = json.load(open(P + '/predictions.json'))['cells']
cells = aggregate(load(P))
SEEN = [(32, 32, 16, 2), (32, 32, 16, 4), (32, 32, 16, 12), (16, 16, 8, 1), (16, 16, 8, 2), (16, 16, 8, 4)]
def sc(shapes):
    m, ms = [], []
    for k, v in cells.items():
        if k[:4] not in shapes or k[5] < 0.1: continue
        kk = f"{k[0]}x{k[1]}x{k[2]}r{k[3]}|nb{int(k[4])}_q{k[5]:g}_c{k[6]:g}"
        mm = abs(pj[kk]['e_L'] - np.mean(list(v['e_seed'].values()))) * 100
        m.append(100.0 if not v['ok'] else mm)
        if v['ok']: ms.append(mm)
    return len(m), np.mean(np.array(m) <= 5), np.median(m), (sum(np.array(ms) <= 5), len(ms))
print('S6 all seen', sc(SEEN))
for s in SEEN + [(16, 16, 8, 8)]: print(s, sc([s]))
