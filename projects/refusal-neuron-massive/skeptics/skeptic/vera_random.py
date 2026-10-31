"""Vera: first-token null families from Bruno's raw score files (data/runs/logit/test_harmful/*.json).
score = log-odds that the first reply token is a refusal start ({'I', 'Sorry'}); drop = mean_prompts(base - arm).
Families are grouped by the file-name prefix before '_<draw>'. F (random neurons): per neuron keep the larger drop
of its pos/neg sign (D4/D6). Prints gate A drop vs each family's distribution.
"""
import json, glob, os, re
import numpy as np

D = '/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/runs/logit/test_harmful'
S = lambda a: np.array(json.load(open(f'{D}/{a}.json'))['score'])
T = lambda a: np.array(json.load(open(f'{D}/{a}.json'))['top_is_refusal'], float)
base, tbase = S('base'), T('base')
drop = lambda a: float((base - S(a)).mean())
fam = {}
for p in glob.glob(f'{D}/*.json'):
    n = os.path.basename(p)[:-5]
    m = re.match(r'(.+?)_(\d+)$', n)
    if m and '.slow' not in n:
        fam.setdefault(m.group(1), {})[int(m.group(2))] = n
A = drop('A')
out = {'A_drop': A, 'A_top_refusal': float(T('A').mean()), 'base_top_refusal': float(tbase.mean())}
print(f'base top-is-refusal {tbase.mean():.3f}; A drop {A:.3f} (top-is-refusal {T("A").mean():.3f})')
for a in ['B', 'C', 'A_tok0', 'A_skip0', 'A_zero', 'A_hm', 'N5441pos', 'N5441neg', 'N4475pos', 'N4475neg']:
    if os.path.exists(f'{D}/{a}.json'):
        out[a] = drop(a)
        print(f'{a:10s} drop {drop(a):+.3f} = {drop(a)/A:+.3f} of A; top-is-refusal {T(a).mean():.3f}')
if 'Fpos' in fam or 'Fneg' in fam:
    ks = sorted(set(fam.get('Fpos', {})) | set(fam.get('Fneg', {})))
    best = []
    for k in ks:
        c = [drop(fam[f][k]) for f in ('Fpos', 'Fneg') if k in fam.get(f, {})]
        best.append(max(c))
    best = np.array(best)
    out['F'] = {'n': len(best), 'p50': float(np.median(best)), 'p99': float(np.percentile(best, 99)),
                'max': float(best.max()), 'A_rank_above': int((best >= A).sum())}
    print(f'F random neurons (n={len(best)}, larger sign): median {np.median(best):.3f}, p99 {np.percentile(best,99):.3f}, '
          f'max {best.max():.3f}; #draws >= A: {(best >= A).sum()}; gate/p99 = {A/np.percentile(best,99):.1f}x')
for f, d in sorted(fam.items()):
    if f in ('Fpos', 'Fneg'):
        continue
    v = np.array([drop(d[k]) for k in sorted(d)])
    out[f] = {'n': len(v), 'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(v.max()),
              'min': float(v.min()), 'median_over_A': float(np.median(v) / A)}
    print(f'{f:10s} n={len(v)} drop median {np.median(v):+.3f} ({np.median(v)/A:+.3f} of A) p95 {np.percentile(v,95):+.3f} '
          f'min {v.min():+.3f} max {v.max():+.3f} ({v.max()/A:+.3f} of A)')
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_random.json'), 'w'), indent=1)
