"""First-order screen (derived from vera_fixdir_screen.py). Pre-data screen (old screen docs pile-10k 9200-9209 only, never the held-out set).
Q: is the frozen-LN error carried by a near-fixed direction?
Modes (everything else REAL: real MLPs, live attention):
  real   : true backward
  lnfroz : all LN scales detached (drop radial term x_hat x_hat^T/d at every norm)
  lnfix  : LN scales detached, but each norm instance (layer, ln_1/ln_2/ln_f) projects out ONE fixed
           unit direction v (mean of x_hat/|x_hat| over calibration tokens, docs 9300-9309)
Reports median cos to real per layer, and mean |cos(x_hat, v)| per norm."""
import torch, numpy as np, json
from datasets import load_dataset
from vera_engine_old import Engine

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)


class E2(Engine):
    V = None  # dict key -> unit direction (centred)
    acc = None  # calibration accumulator

    def fwd(self, ids, L, mode, delta):
        T = len(ids)
        x = self.wte[ids] + self.wpe[:T]
        for l, B in enumerate(self.blocks):
            h = self.nrm(x, mode, (l, 1)) * B['g1'] + B['b1']
            x = x + self.attn(h, B, False)
            z = self.nrm(x, mode, (l, 2))
            mo = gelu_new((z * B['g2'] + B['b2']) @ B['Wi'] + B['bi']) @ B['Wm'] + B['bm']
            if l == L and delta is not None: mo = mo + delta
            x = x + mo
        hf = self.nrm(x, mode, ('f',)) * self.gf + self.bf
        return hf @ self.WU.T

    def nrm(self, x, mode, key):
        c = x - x.mean(-1, keepdim=True)
        sd = torch.sqrt((c ** 2).mean(-1, keepdim=True) + 1e-5)
        if mode == 'calib':
            u = (c / c.norm(dim=-1, keepdim=True))[1:]  # skip BOS position
            self.acc.setdefault(key, []).append(u.sum(0))
            return c / sd
        if mode == 'align':
            u = (c / c.norm(dim=-1, keepdim=True))[1:]
            self.acc.setdefault(key, []).extend((u @ self.V[key]).tolist())
            return c / sd
        if mode == 'real': return c / sd
        sdd = sd.detach()
        if mode == 'lnfroz': return c / sdd
        if isinstance(mode, float):  # eps-blend: radial term kept with weight mode (0=real,1=frozen)
            return c / (mode * sdd + (1 - mode) * sd)
        v = self.V[key]
        pc = c @ v
        return (c - v * pc[:, None] + v * pc.detach()[:, None]) / sdd


from vera_lib import gelu_new

E = E2()
E.blocks = [{k: v.double() for k, v in B.items()} for B in E.blocks]
E.wte, E.wpe, E.gf, E.bf, E.WU = (t.double() for t in (E.wte, E.wpe, E.gf, E.bf, E.WU))
ds = load_dataset('NeelNanda/pile-10k', split='train')


def ids_of(i):
    ids = E.tok(ds[i]['text'], return_tensors='pt').input_ids[0][:127]
    return torch.cat([torch.tensor([50256]), ids])


E.acc = {}; ntok = 0
for i in range(9300, 9310):
    ids = ids_of(i); E.fwd(ids, None, 'calib', None); ntok += len(ids) - 1
E.V = {}
for k, lst in E.acc.items():
    v = torch.stack(lst).sum(0); v = v - v.mean(); E.V[k] = v / v.norm()
# alignment of x_hat with v on screen docs
align = {}
E.acc = {}
for i in range(9200, 9210): E.fwd(ids_of(i), None, 'align', None)
align={}


def grad(ids, L, t, tok, mode):
    delta = torch.zeros(len(ids), 768, requires_grad=True)
    with torch.enable_grad():
        lo = E.fwd(ids, L, mode, delta)
        y = lo[t, tok] - lo[t].mean()
        g, = torch.autograd.grad(y, delta)
    return g[t]


rng = np.random.default_rng(0); out = {}
for L in [8, 9, 10, 11]:
    cs = {'lnfroz': [], 'first': [], 'second': []}
    for i in range(9200, 9210):
        ids = ids_of(i)
        if len(ids) < 40: continue
        for t in rng.choice(np.arange(1, len(ids) - 1), 3, replace=False):
            t = int(t); tok = int(ids[t + 1])
            gr = grad(ids[:t + 1], L, t, tok, 'real')
            gf = grad(ids[:t + 1], L, t, tok, 'lnfroz')
            h = 1e-2
            gp = grad(ids[:t + 1], L, t, tok, h); gm = grad(ids[:t + 1], L, t, tok, -h)
            d1 = (gp - gm) / (2 * h); d2 = (gp - 2 * gr + gm) / h ** 2
            cosf = lambda a: float(torch.nn.functional.cosine_similarity(a, gr, 0))
            cs['lnfroz'].append(cosf(gf)); cs['first'].append(cosf(gr + d1)); cs['second'].append(cosf(gr + d1 + d2 / 2))
    out[L] = {m: round(float(np.median(v)), 3) for m, v in cs.items()}
    from scipy.stats import spearmanr
    out[L]['spear_first'] = round(float(spearmanr(cs['lnfroz'], cs['first'])[0]), 3)
    out[L]['mad_first'] = round(float(np.median(np.abs(np.array(cs['lnfroz']) - np.array(cs['first'])))), 3)
    out[L]['n'] = len(cs['lnfroz'])
    print(L, out[L], flush=True)
json.dump({'cos': out, 'align': align}, open('vera_first_screen.json', 'w'), indent=1)
