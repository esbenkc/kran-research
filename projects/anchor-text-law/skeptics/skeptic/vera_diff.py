# Vera: field-by-field diff of my exact Fractions (vera_check.py on raw rows) vs the build's results/analysis.json.
import json, sys, os
from fractions import Fraction as F
import vera_check as V
proj = sys.argv[1]
J = json.load(open(os.path.join(proj, 'results', 'analysis.json')))
a, b = V.model_dirs()
evA, _, _ = V.build(a[0]); evB, _, _ = V.build(b[0])
nA, nB = V.scored_n(evA), V.scored_n(evB)
R = V.compute(evA, evB, nA, nB)
n = 0; bad = []
def chk(name, mine, theirs):
    global n
    n += 1
    if F(mine) != F(theirs):
        bad.append((name, mine, theirs))
for c in J['cells']:
    if not c.get('scored'):
        continue
    m = c['slot']; f = c['cell'].replace('-free', '')
    if f not in V.MAIN:
        continue
    k = R['C'][(m, f)]
    chk(f'{m}-{f} n', k['n'], c['n']); chk(f'{m}-{f} reached', k['reached'], c['reached'])
    chk(f'{m}-{f} followed', k['follow'], c['followed']); chk(f'{m}-{f} followed_reached', k['follow_reached'], c['followed_reached'])
    chk(f'{m}-{f} fr', R['fr'][(m, f)], c['fr']['frac']); chk(f'{m}-{f} catch', R['catch'][(m, f)], c['catch']['frac'])
    for bl in range(3):
        mb = [evA, evB][m == 'B'][f]; nn = [nA, nB][m == 'B']
        es = [mb[i] for i in range(nn) if i % 3 == bl]
        pb = c['per_block'][str(bl)]
        chk(f'{m}-{f} b{bl} n', len(es), pb['n']); chk(f'{m}-{f} b{bl} followed', sum(e['follow'] for e in es), pb['followed'])
for key, (m, f) in {'A-md': ('A', 'md'), 'A-anchor': ('A', 'anchor'), 'B-raw': ('B', 'raw'), 'B-bare': ('B', 'bare')}.items():
    h = J['heldout'][key]
    chk(key + ' pred', R['pred'][(m, f)], h['pred']['frac']); chk(key + ' catch', R['catch'][(m, f)], h['catch']['frac'])
    chk(key + ' err', R['err'][(m, f)], h['err']['frac'])
    for bl in range(3):
        Rb = V.compute(evA, evB, nA, nB, block=bl)
        chk(f'{key} seed b{bl} err', Rb['err'][(m, f)], h['err_seed']['blocks'][bl]['frac'])
    if key in J['nulls']:
        chk(key + ' format-blind err', R['null_fb'][(m, f)], J['nulls'][key]['format_blind_err']['frac'])
        chk(key + ' constant err', R['null_const'][(m, f)], J['nulls'][key]['constant_err']['frac'])
    print(f"{key}: my CI vs json CI: [{h['err_ci'][0]['value']:+.4f}, {h['err_ci'][1]['value']:+.4f}] (json, seed 20261123)")
for m in 'AB':
    chk(m + ' anchor-md', R['fr'][(m, 'anchor')] - R['fr'][(m, 'md')], J['fr_diffs'][m]['anchor_md']['value']['frac'])
    chk(m + ' bare-md', R['fr'][(m, 'bare')] - R['fr'][(m, 'md')], J['fr_diffs'][m]['bare_md']['value']['frac'])
chk('gate A', 30, J['gate']['A']['success']); chk('gate B', 22, J['gate']['B']['success'])
G = {m: R['fr'][(m, 'bare')] <= R['fr'][(m, 'md')] - F(15, 100) for m in 'AB'}
print('G mine', G, 'json', J['status']['G_fired'], '| status json', J['status']['status'], J['status']['prediction'])
print(f'{n} fields compared, {len(bad)} mismatches'); [print('  MISMATCH', x) for x in bad]
