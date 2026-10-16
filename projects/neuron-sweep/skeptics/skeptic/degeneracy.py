"""How much of 'mono' at 10x is a repetition loop rather than a theme?

Judge-free measure: zlib compression ratio of each text (compressed bytes / raw bytes).
A row is 'degenerate' if >= 2 of its 3 texts compress below the 1st percentile of the no-push arm.
Reports degenerate share per arm x final label. Reads only data/*.jsonl.
"""
import json, os, zlib, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data') + '/'


def rows(f):
    return [json.loads(l) for l in open(D + f) if l.strip()]


def nk(kind, scale, layer, neuron):
    return f"{kind}|{0 if kind == 'none' else float(scale):g}|{int(layer)}|{int(neuron)}"


def cr(t):
    b = t.encode()
    return len(zlib.compress(b, 9)) / max(len(b), 1)


lab = {nk(r['kind'], r['scale'], r['layer'], r['neuron']): r['label'] for r in rows('judgments_final.jsonl')}
gen = []
for f in ('samples.jsonl', 'samples_x10.jsonl', 'controls.jsonl', 'controls_x10.jsonl'):
    gen += rows(f)
none_cr = [cr(t) for r in gen if r['kind'] == 'none' for t in r['texts']]
thr = float(np.percentile(none_cr, 1))
tab = collections.defaultdict(lambda: collections.Counter())
for r in gen:
    k = nk(r['kind'], r['scale'], r['layer'], r['neuron'])
    a = 'none' if r['kind'] == 'none' else f"{r['kind']}_x{float(r['scale']):g}"
    deg = sum(cr(t) < thr for t in r['texts']) >= 2
    tab[a][(lab[k], deg)] += 1
out = {'threshold_cr': thr}
for a in sorted(tab):
    c = tab[a]
    n = sum(c.values())
    line = {}
    for l in ('none', 'mono', 'poly'):
        nl = c[(l, True)] + c[(l, False)]
        line[l] = dict(n=nl, degenerate_pct=100 * c[(l, True)] / nl if nl else None)
    out[a] = dict(n=n, degenerate_pct=100 * sum(v for (l, d), v in c.items() if d) / n, by_label=line)
    print(f"{a:14s} n={n:6d} degenerate {out[a]['degenerate_pct']:5.1f}%  | " + '  '.join(
        f"{l}: {line[l]['degenerate_pct'] if line[l]['degenerate_pct'] is None else round(line[l]['degenerate_pct'],1)}% of {line[l]['n']}"
        for l in line))
print('threshold (1st pct of no-push compression ratio):', round(thr, 3))
json.dump(out, open(os.path.join(HERE, 'degeneracy_out.json'), 'w'), indent=1)
