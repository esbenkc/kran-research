"""Recompute every number in the post's results + foldout from raw blocks / raw Gibbs / raw arm-P blocks, own code."""
import json, glob, numpy as np
from collections import defaultdict
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
pj = json.load(open(P + '/predictions.json'))['cells']
def key(k): return f"{k[0]}x{k[1]}x{k[2]}r{k[3]}|nb{int(k[4])}_q{k[5]:g}_c{k[6]:g}"
def table(cells, sel, label, gibbs_sub=None):
    preds = ['L', 'Lz', 'N4', 'N2', 'N3', 'N1']
    ms = {p: [] for p in preds}; ms_ok = {p: [] for p in preds}
    for k, v in cells.items():
        if not sel(k): continue
        em = np.mean(list(v['e_seed'].values())) if gibbs_sub is None else gibbs_sub[k]
        for p in preds:
            m = abs(pj[key(k)]['e_' + p] - em) * 100
            ms[p].append(100.0 if not v['ok'] and gibbs_sub is None else m)
            if v['ok']: ms_ok[p].append(m)
    print(label, 'n', len(ms['L']))
    for p in preds:
        print(f"   {p:3s} share {np.mean(np.array(ms[p]) <= 5):.3f} median {np.median(ms[p]):.2f} | settled-only share {np.mean(np.array(ms_ok[p]) <= 5):.3f} (n {len(ms_ok[p])})")
    return ms
cells = aggregate(load(P))
prim = lambda k: k[:4] in BLIND and k[5] >= 0.1
table(cells, prim, 'PRIMARY arm S')
table(cells, lambda k: prim(k) and k[:4] != (24, 24, 12, 3), 'without 24x24x12r3')
table(cells, lambda k: k[:4] in BLIND and k[5] < 0.1, 'S4 q<0.1')
# per shape
for s in sorted(BLIND):
    sub = {k: v for k, v in cells.items() if prim(k) and k[:4] == s}
    m = [100.0 if not v['ok'] else abs(pj[key(k)]['e_L'] - np.mean(list(v['e_seed'].values()))) * 100 for k, v in sub.items()]
    print('shape', s, 'share', round(float(np.mean(np.array(m) <= 5)), 3))
# Gibbs
gb = {}
for f in glob.glob(P + '/data/gibbs_S/*.json'):
    g = json.load(open(f)); c = g['case']; M, N, H, r = [int(t) for t in c.replace('r', 'x').split('x')]
    for x in g['rows']: gb.setdefault((M, N, H, r, float(x['nbeta']), float(x['q'])), []).append(x['e_gibbs'])
tgt, smp, gsub = [], [], {}
for k, v in cells.items():
    if not prim(k): continue
    g = np.mean(gb[k[:6]]); gsub[k] = g
    tgt.append(abs(g) * 100); smp.append(abs(np.mean(list(v['e_seed'].values())) - g) * 100)
print(f"Gibbs target |e| median {np.median(tgt):.2f}; sampler |SGLD-Gibbs| median {np.median(smp):.2f} max {np.max(smp):.1f}")
mg = [abs(pj[key(k)]['e_L'] - gsub[k]) * 100 for k in gsub]
print(f"law vs Gibbs: share {np.mean(np.array(mg) <= 5):.3f} (n {len(mg)})")
# halves of failed cells
raw = load(P); hd = []
for x in raw:
    k = cell_key(x)
    if prim(k) and not cells[k]['ok']:
        e1, e2 = x['halves'][0]; hd.append(abs(e1 - e2) * 100)
fails = [abs(a) for a in hd if a > 2]
print(f"failed-cell seed halves > 2 pts: min {min(fails):.2f} max {max(fails):.2f} median {np.median(fails):.2f}")
# arm P
try:
    cP = aggregate(load(P, 'P'))
    table(cP, prim, 'arm P primary')
except Exception as e:
    print('arm P', repr(e))
