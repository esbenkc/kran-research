"""Cell-by-cell: my independent SGLD/Gibbs (vera_blind_sgld.json) vs Bruno's raw-block e and Gibbs on the same settings."""
import json, glob, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
rows = load(P); cells = aggregate(rows)
gb = {}
for f in glob.glob(P + '/data/gibbs_S/*.json'):
    g = json.load(open(f))
    for x in g['rows']: gb.setdefault((g['case'], float(x['nbeta']), float(x['q'])), []).append(x['e_gibbs'])
v = json.load(open('vera_blind_sgld.json')); d = []; dg = []
for o in v:
    M, N, H, r = o['shape']; nb, q, c = [float(t[2:]) if i == 0 else float(t[1:]) for i, t in enumerate(o['key'].split('_'))]
    k = (M, N, H, r, nb, q, c); em = np.mean(list(cells[k]['e_seed'].values())); g = np.mean(gb[(f"{M}x{N}x{H}r{r}", nb, q)])
    d.append(100 * (o['e_sgld'] - em)); dg.append(100 * (o['e_gibbs'] - g))
    print(f"{M}x{N}x{H}r{r} {o['key']:20s} SGLD Bruno {100*em:+5.1f} (ok={cells[k]['ok']}) mine {100*o['e_sgld']:+5.1f} | Gibbs Bruno {100*g:+5.1f} mine {100*o['e_gibbs']:+5.1f}")
print('SGLD mine-Bruno: median %+.2f, max |.| %.2f; Gibbs mine-Bruno: median %+.2f, max |.| %.2f' % (np.median(d), np.max(np.abs(d)), np.median(dg), np.max(np.abs(dg))))
