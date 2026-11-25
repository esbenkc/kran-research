"""Vera: field-by-field diff of my own rebuild (vera_check / vera_gates logic, raw files) against results/analysis.json.
Usage: python3 vera_diff.py web/out/qwen3-8b results/analysis.json
"""
import json, os, sys, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_check as v

out, af = sys.argv[1], sys.argv[2]
A = json.load(open(af))
man = [json.loads(l) for l in open(os.path.join(out, 'manifest.jsonl'))]
runs = {(r['arm'], r['i']): r for r in v.load(out) if r['scored']}
gate_rows = [r for r in v.load(out) if not r['scored']]
arms = ['raw', 'rawnf', 'md', 'mdhtml', 'mdnf', 'mdpad8']
c = collections.Counter(); n = collections.Counter(); f200 = collections.Counter(); off = collections.Counter()
reach = collections.Counter(); corr = collections.Counter(); blocks = collections.defaultdict(lambda: [0, 0, 0])
for (a, i), r in runs.items():
    n[a] += 1; c[a] += r['follow']; f200[a] += r['follow200']; off[a] += r['off']; blocks[a][i % 3] += r['follow']
    d = json.load(open(os.path.join(out, r['row']['raw_file'])))
    reach[a] += any(t.get('role') == 'observation' and r['row']['hid'] in (t.get('text') or '') for t in d['transcript'])
    corr[a] += bool(d['correct'])
gsucc = 0
for r in gate_rows:
    d = json.load(open(os.path.join(out, r['row']['raw_file'])))
    gsucc += r['follow'] and bool(d['correct'])

mism = []; checked = [0]
def cmp(name, mine, theirs):
    checked[0] += 1
    ok = (abs(mine - theirs) < 1e-9) if isinstance(mine, float) or isinstance(theirs, float) else mine == theirs
    if not ok:
        mism.append((name, mine, theirs))

s, G, conf, rej = v.status(c, n)
cmp('status', s, A['status'])
cmp('gates.gap', G, A['gates']['gap']); cmp('gap_runs', c['md'] - c['raw'], A['gap_runs'])
cmp('gate.success', gsucc, A['gate']['success']); cmp('gate.runs', len(gate_rows), A['gate']['runs'])
cmp('gates.skill', gsucc >= 11, A['gates']['skill'])
for k in conf: cmp('confirmed_legs.' + k, conf[k], A['confirmed_legs'][k])
for k in rej: cmp('rejected_legs.' + k, rej[k], A['rejected_legs'][k])
for a in arms:
    cmp(f'catch.{a}', c[a] / n[a], A['catch'][a]) if a in A['catch'] else None
    X = A['arms'][a]
    for k, val in [('n', n[a]), ('follow', c[a]), ('follow200', f200[a]), ('offtask', off[a]), ('reach', reach[a]),
                   ('correct', corr[a]), ('blocks', blocks[a])]:
        cmp(f'arms.{a}.{k}', val, X[k])
    lo, hi = v.wilson(c[a], n[a]); cmp(f'arms.{a}.wilson.lo', lo, X['wilson'][0]); cmp(f'arms.{a}.wilson.hi', hi, X['wilson'][1])
    cmp(f'S5.{a}', f200[a], A['secondary']['S5_200only'][a])
pairs = {'rawnf-raw': ('rawnf', 'raw'), 'mdhtml-mdnf': ('mdhtml', 'mdnf'), 'md-rawnf': ('md', 'rawnf'),
         'mdnf-raw': ('mdnf', 'raw')}
for key, (x, y) in pairs.items():
    lo, hi = v.newcombe(c[x], n[x], c[y], n[y])
    D = A['differences'][key]
    cmp(key + '.diff', c[x] / n[x] - c[y] / n[y], D['diff']); cmp(key + '.lower', lo, D['lower']); cmp(key + '.upper', hi, D['upper'])
for key in list(pairs) + ['md-raw']:
    x, y = key.split('-')
    b = [blocks[x][k] / 30 - blocks[y][k] / 30 for k in range(3)]
    m, tl, th = v.tint(b)
    S = A['seed_level'][key]
    for j in range(3): cmp(f'seed.{key}.b{j}', b[j], S['blocks'][j])
    cmp(f'seed.{key}.mean', m, S['mean']); cmp(f'seed.{key}.lower', tl, S['lower']); cmp(f'seed.{key}.upper', th, S['upper'])
lo, hi = v.newcombe(c['mdhtml'], 90, c['md'], 90)
S1 = A['secondary']['S1_mdhtml-md']; cmp('S1.diff', c['mdhtml'] / 90 - c['md'] / 90, S1['diff']); cmp('S1.lower', lo, S1['lower']); cmp('S1.upper', hi, S1['upper'])
lo, hi = v.newcombe(c['md'], 90, c['mdpad8'], 60)
S2 = A['secondary']['S2_length']['md-mdpad8']; cmp('S2.lower', lo, S2['lower']); cmp('S2.upper', hi, S2['upper'])
wl, wh = v.wilson(c['raw'], 90); p = 80 / 83
S3 = A['secondary']['S3_law_clause']; cmp('S3.err', c['raw'] / 90 - p, S3['err']); cmp('S3.lower', wl - p, S3['lower']); cmp('S3.upper', wh - p, S3['upper'])
print(f'checked {checked[0]} fields, {len(mism)} mismatches')
for m in mism:
    print('MISMATCH', m)
