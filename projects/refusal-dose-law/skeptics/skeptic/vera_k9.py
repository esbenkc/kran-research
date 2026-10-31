"""Vera: D2/D4 rebuild. (1) build's G_offK / G_offK9 vectors (d2.pt, d4.pt) checked against MY directions: my r_13 and
my own DoM at the last 6 template tokens (recomputed here from vera_offaxis' recipe would cost a forward; instead I use
the energy check against my span from vera_offaxis.json is not stored -> recompute the 9-position DoM with my forward).
(2) ratio EFFECT(G_offK*@30)/EFFECT(G@30) with my paired bootstrap (2,000, seed 0), both labelings.
"""
import json, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F
from vera_rebuild import load_cells, y_of

D = '/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/'
cells = load_cells()
rng = np.random.default_rng(0)
ids = cells['base']['ids']
for k in ['G_offK', 'G_offK9']:
    for mode in ['comply', 'nonrefuse']:
        yb, yg, yk = (y_of(cells[c]['labs'], mode) for c in ['base', 'G@30', f'{k}@30'])
        assert (cells[f'{k}@30']['ids'] == ids).all() and (cells['G@30']['ids'] == ids).all()
        pt = (yk.mean() - yb.mean()) / (yg.mean() - yb.mean())
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(ids), len(ids))
            bs.append((yk[i].mean() - yb[i].mean()) / (yg[i].mean() - yb[i].mean()))
        print(k, mode, 'comp@30 %.3f' % yk.mean(), 'ratio %.3f' % pt, 'CI', np.round(np.quantile(bs, [.025, .975]), 3))
# vector check with my own 9-position DoM at L13
torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
tok = AutoTokenizer.from_pretrained(F.SNAP)


def last9(prompts):
    ids_ = [F.encode(tok, p) for p in prompts]
    xs = [F.T_rows('model.embed_tokens.weight', i) for i in ids_]
    pos = [torch.arange(len(i)) for i in ids_]
    for l in range(14):
        W = F.load_layer(l)
        for b in range(len(ids_)):
            xs[b], _, _ = F.layer_step(W, xs[b], pos[b], None, None)
        del W
    return torch.stack([x[-9:] for x in xs])


H, N = last9(S['dev_harmful'][:100]), last9(S['test_harmless'][:100])
R = (H.mean(0) - N.mean(0))  # [9, D]
Q, _ = torch.linalg.qr(R.T)
for name, f in [('G_offK', 'd2.pt'), ('G_offK9', 'd4.pt')]:
    obj = torch.load(D + 'phase0/' + f, weights_only=False)
    vec = None
    if isinstance(obj, dict):
        for kk, v in obj.items():
            if hasattr(v, 'shape') and tuple(v.shape) == (2048,) and 'off' in kk.lower():
                vec = v.float(); print(name, 'vector key', kk)
    if vec is None:
        print(name, 'no 2048 vector found; keys', list(obj)[:10] if isinstance(obj, dict) else type(obj)); continue
    e = float((Q.T @ vec).pow(2).sum() / vec.pow(2).sum())
    cs = [float(vec @ R[k] / vec.norm() / R[k].norm()) for k in range(9)]
    print(name, 'energy in MY 9-position span %.5f' % e, 'max |cos| with my position r %.4f' % max(map(abs, cs)), 'norm/||w|| %.3f' % float(vec.norm() / 1.10845))
