"""Vera's rebuild of the primary status from RAW blocks (frozen 2(6)), plus anatomy and robustness."""
import json, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
out, cells, prim, train = score(P, verbose=True)
print('rebuild', REBUILD)
# anatomy
a, b = out['a'], out['b']
nok = [k for k, v in prim.items() if not v['ok']]
print('not-ok primary', len(nok), 'by c', {c: sum(k[6] == c for k in nok) for c in (0.01, 0.03, 0.1)}, 'by q', {q: sum(k[5] == q for k in nok) for q in (0.1, 1, 10, 100)})
conv = [(k, v) for k, v in prim.items() if v['ok']]
m_conv = [abs(law_pred(k[:4], k[4], k[5], v['eps'], a, b) - np.mean(list(v['e_seed'].values()))) * 100 for k, v in conv]
print('converged cells', len(conv), 'share<=5', np.mean(np.array(m_conv) <= 5), 'median', np.median(m_conv))
for k, v in conv:
    m = abs(law_pred(k[:4], k[4], k[5], v['eps'], a, b) - np.mean(list(v['e_seed'].values()))) * 100
    if m > 5: print('  law miss', k, round(m, 1), 'meas', round(100 * np.mean(list(v['e_seed'].values())), 1))
# without (24,24,12,3)
sub = {k: v for k, v in prim.items() if k[:4] != (24, 24, 12, 3)}
ms = [100.0 if not v['ok'] else abs(law_pred(k[:4], k[4], k[5], v['eps'], a, b) - np.mean(list(v['e_seed'].values()))) * 100 for k, v in sub.items()]
print('without 24x24x12r3: n', len(ms), 'share', np.mean(np.array(ms) <= 5))
json.dump(dict(out=out, rebuild=REBUILD), open('vera_blind_score.json', 'w'), indent=1, default=float)
