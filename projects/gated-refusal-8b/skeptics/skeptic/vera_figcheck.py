"""Vera: recount the main figure's data (fig_counts.json) from per-prompt judge rows + gate.json.
Usage: python vera_figcheck.py <project dir>"""
import json, sys, collections

P = sys.argv[1]
L = lambda p: [json.loads(l) for l in open(p)]
f = json.load(open(f'{P}/data/fig_counts.json'))
base = {(r['kind'], r['i']): r['judge'] for r in L(f'{P}/data/base/gen_test.judged.jsonl')}
n = bad = 0
for srow in f['seeds']:
    s = srow['seed']
    lab = {(r['set'], r['kind'], r['i']): r['judge'] for r in L(f'{P}/data/strip_s{s}/gen_test.judged.jsonl')}
    fl = json.load(open(f'{P}/data/strip_s{s}/gate.json'))['test_flags']
    srcs = {'S': lambda k, i: 'none', 'G': lambda k, i: 'push' if fl[k][i] else 'none',
            'RG0': lambda k, i: 'rand0' if fl[k][i] else 'none', 'RG1': lambda k, i: 'rand1' if fl[k][i] else 'none',
            'U': lambda k, i: 'push', 'O': lambda k, i: 'push' if k == 'harmful' else 'none'}
    for a, src in srcs.items():
        if a not in srow['arms']:
            continue
        for k in ('harmful', 'harmless'):
            mine = collections.Counter(lab[(src(k, i), k, i)] for i in range(200)
                                       if k == 'harmful' or base[('harmless', i)] == 'compliance')
            n += 1
            if dict(mine) != srow['arms'][a][k]:
                bad += 1
                print('MISMATCH', s, a, k, dict(mine), srow['arms'][a][k])
        new = sum(base[('harmless', i)] == 'compliance' and lab[(src('harmless', i), 'harmless', i)] != 'compliance'
                  for i in range(200))
        n += 1
        if new != srow['arms'][a]['new_harmless']:
            bad += 1
            print('MISMATCH new', s, a, new, srow['arms'][a]['new_harmless'])
    for k, key in (('harmful', 'flags_harmful'), ('harmless', 'flags_harmless')):
        n += 1
        if sum(fl[k]) != srow[key]:
            bad += 1
            print('MISMATCH flags', s, k)
print(f'{n} figure cells recounted, {bad} mismatches; arms per seed: {[list(x["arms"]) for x in f["seeds"]][0]}')
