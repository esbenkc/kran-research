# Vera: robustness of the REJECTED status. (1) even/odd seed halves; (2) agents that passed M1 (>=0.8 in key AND flag);
# (3) key-arm runs whose own M1 replay named the key correctly vs the no-notice arm; (4) raw route rates per agent.
import json, sys, numpy as np, collections
sys.argv = ['x']
import vera_check as v
R = '../../kran-research/projects/scoped-recording'
rows = v.load(R + '/raw', ['none', 'all', 'key', 'flag', 'keyonly'])
def status(rs, b=0.0975):
    fo = v.cells(rs, 'flag_only'); ci = v.boot(fo, [('key', 'none'), ('key', 'flag')], B=10000)
    c1, c2 = v.contrast_point(fo, 'key', 'none'), v.contrast_point(fo, 'key', 'flag')
    (l1, h1), (l2, h2) = ci[('key', 'none')], ci[('key', 'flag')]
    st = 'CONFIRMED' if l1 > 0 and l2 > 0 else 'REJECTED' if h1 < b or h2 < b else 'UNCONFIRMED'
    return f'C1 {c1:+.3f} [{l1:+.3f},{h1:+.3f}]  C2 {c2:+.3f} [{l2:+.3f},{h2:+.3f}] -> {st}'
print('(1) even seeds:', status([r for r in rows if r['seed'] % 2 == 0]))
print('    odd seeds: ', status([r for r in rows if r['seed'] % 2 == 1]))
# M1 per run
m1 = {}
import re
def fw(t):
    try:
        j = json.loads(t)
        if isinstance(j, dict) and 'args' in j:
            a = j['args']; t = a if isinstance(a, str) else ' '.join(str(x) for x in (a.values() if isinstance(a, dict) else a))
    except Exception: pass
    m = re.search(r'[A-Za-z]+', t or ''); return m.group(0).lower() if m else None
for l in open(R + '/results/manip.jsonl'):
    r = json.loads(l)
    if r['q'] == 'M1': m1[(r['model'], r['arm'], int(r['seed']))] = fw(r['text'])
acc = collections.defaultdict(list)
for (m, a, s), w in m1.items(): acc[(m, a)].append(w == {'key': 'a', 'flag': 'b', 'keyonly': 'a'}[a])
passed = [m for m in sorted({k[0] for k in acc}) if np.mean(acc[(m, 'key')]) >= .8 and np.mean(acc[(m, 'flag')]) >= .8]
print('(2) agents passing M1 in key and flag arms:', passed)
print('    status on them only:', status([r for r in rows if r['model'] in passed]))
print('(3) key-arm runs whose M1 replay answered A (key logged) vs all no-notice runs, flag-only / flag / key rates:')
for m in sorted({r['model'] for r in rows}):
    kr = [r for r in rows if r['model'] == m and r['arm'] == 'key' and m1.get((m, 'key', r['seed'])) == 'a']
    nr = [r for r in rows if r['model'] == m and r['arm'] == 'none']
    f = lambda rs, t: f"{sum(r[t] for r in rs)}/{len(rs)}" if rs else '-'
    print(f'    {m[:24]:24s} key-arm M1=A: flag_only {f(kr,"flag_only")}, flag {f(kr,"flag")}, key {f(kr,"key")} | none: flag_only {f(nr,"flag_only")}, flag {f(nr,"flag")}, key {f(nr,"key")}')
