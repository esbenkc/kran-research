"""Check the figure's per-layer x amax-fifth panel (blog data.json) against raw data.
Fifths = fifths of amax over ALL 3,072 neurons of the layer (data/amax.npy). amax is bf16, so ties at
the edges are common; several binning conventions are tried and the best one reported.
Usage: uv run --no-project --with numpy python figure_check.py /path/to/data.json
"""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data') + '/'
fig = json.load(open(sys.argv[1]))
amax = np.load(D + 'amax.npy')


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nk(kind, scale, layer, neuron):
    return f"{kind}|{0 if kind == 'none' else float(scale):g}|{int(layer)}|{int(neuron)}"


lab = {nk(r['kind'], r['scale'], r['layer'], r['neuron']): r['label'] for r in rows(D + 'judgments_final.jsonl')}
R = {'x1': rows(D + 'samples.jsonl'), 'x10': rows(D + 'samples_x10.jsonl')}
rank = {L: np.argsort(np.argsort(amax[L], kind='stable'), kind='stable') for L in range(28)}


def binner(name):
    if name == 'stable_rank':
        return lambda L, i: int(rank[L][i] * 5 // amax.shape[1])
    if name == 'rank_below':
        return lambda L, i: min(4, int((amax[L] < amax[L, i]).sum() * 5 // amax.shape[1]))
    side, method = name.split('/')
    E = {L: np.quantile(amax[L], [.2, .4, .6, .8], method=method) for L in range(28)}
    return lambda L, i: int(np.searchsorted(E[L], amax[L, i], side=side))


best = None
for name in ('right/linear', 'left/linear', 'right/lower', 'left/higher', 'stable_rank', 'rank_below'):
    b = binner(name)
    res = {}
    for s in ('x1', 'x10'):
        cells = collections.defaultdict(collections.Counter)
        for r in R[s]:
            cells[(r['layer'], b(r['layer'], r['neuron']))][lab[nk('neuron', r['scale'], r['layer'], r['neuron'])]] += 1
        bad = 0
        for e in fig['amax_fifth'][s]:
            c = cells[(e['layer'], e['fifth'])]; n = sum(c.values())
            bad += (n != e['n']) or (n > 0 and max(abs(c[l] / n - e[l]) for l in ('none', 'mono', 'poly')) > 1e-9)
        tn = sum(sum(v for k, v in cells[(L, 4)].items() if k != 'none') for L in range(28))
        td = sum(sum(cells[(L, 4)].values()) for L in range(28))
        res[s] = (bad, 100 * tn / td)
    print(f"{name:13s} mismatching cells x1 {res['x1'][0]:3d}/140, x10 {res['x10'][0]:3d}/140; "
          f"pooled top-fifth effect x1 {res['x1'][1]:.1f}%, x10 {res['x10'][1]:.1f}%")
