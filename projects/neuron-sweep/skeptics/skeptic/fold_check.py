"""Check the fold's extra numbers (blog 1bc8488) from raw data: judge-choice receipts, push-size quintiles,
per-layer correlation. Reads only data/. Run: uv run --no-project python fold_check.py
"""
import json, os, collections
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data') + '/'


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nk(kind, scale, layer, neuron):
    return f"{kind}|{0 if kind == 'none' else float(scale):g}|{int(layer)}|{int(neuron)}"


hand = json.load(open(D + 'hand_labels.json')); hand.update(json.load(open(D + 'dev_labels.json')))
out = {}
V = D + 'validation/'
for f in sorted(os.listdir(V)):
    if f.endswith('.jsonl'):
        R = rows(V + f)
        pairs = [(hand[r['id']], r['label']) for r in R]
        hn = [b for a, b in pairs if a == 'none']
        out[f] = dict(n=len(R), acc=mean(a == b for a, b in pairs), ordinary_called_changed=mean(b != 'none' for b in hn))
        print(f"{f:58s} n={len(R)} acc {out[f]['acc']:.3f}  ordinary->changed {out[f]['ordinary_called_changed']:.3f}")

lab = {nk(r['kind'], r['scale'], r['layer'], r['neuron']): r['label'] for r in rows(D + 'judgments_final.jsonl')}


def quint(src, kind):
    byL = collections.defaultdict(list)
    for r in rows(D + src):
        if r['kind'] == kind:
            byL[r['layer']].append((r['amax'], lab[nk(r['kind'], r['scale'], r['layer'], r['neuron'])] != 'none'))
    top, rest = [], []
    for v in byL.values():
        v.sort(); q = len(v) // 5
        top += [e for _, e in v[-q:]]; rest += [e for _, e in v[:-q]]
    return 100 * mean(top), 100 * mean(rest)


for name, src, kind in (('neuron_x1', 'samples.jsonl', 'neuron'), ('direction_x1', 'controls.jsonl', 'direction'),
                        ('neuron_x10', 'samples_x10.jsonl', 'neuron'), ('direction_x10', 'controls_x10.jsonl', 'direction')):
    t, r = quint(src, kind)
    out[f'top_fifth_{name}'] = dict(top=t, rest=r)
    print(f"{name:14s} effect, top fifth of amax per layer {t:.1f}%  rest {r:.1f}%")


def per_layer(kind, scale):
    c = collections.defaultdict(collections.Counter)
    for k, l in lab.items():
        kd, s, L, _ = k.split('|')
        if kd == kind and float(s) == scale:
            c[int(L)][l != 'none'] += 1
    return [c[L][True] / (c[L][True] + c[L][False]) for L in range(28)]


def corr(x, y):
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5


for s in (1.0, 10.0):
    r = corr(per_layer('neuron', s), per_layer('direction', s))
    out[f'layer_corr_x{s:g}'] = r
    print(f"per-layer effect-rate correlation neuron vs direction at x{s:g}: r = {r:.3f}")
json.dump(out, open(os.path.join(HERE, 'fold_check_out.json'), 'w'), indent=1)
