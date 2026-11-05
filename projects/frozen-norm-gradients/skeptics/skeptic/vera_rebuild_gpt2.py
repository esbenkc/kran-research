"""Vera's GPT-2 reading cells for PREREG 0b4e178b s6, own engine (vera_engine_old.py = my theory-1104 engine).
Set A: first 200 usable docs from pile-10k row 8000 (>=128 tokens), context BOS + first 127 tokens,
t = default_rng(1105+doc).integers(4,127) with the massive-activation redraw over read layers 8-11.
Cells: 2^3 freeze factorial over {ln, attn, tc} (tc = downstream transcoders with frozen error node),
LN-only tick (exact first order from the frozen end via JVP), |g_frz|/|g_real| in the LN-only cell."""
import json, time, itertools, torch, numpy as np
from datasets import load_dataset
import vera_engine_old as VE
from vera_engine_old import Engine

W = {'w': 1.0}


def norm(x, detach_scale):
    mu = x.mean(-1, keepdim=True)
    sd = torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True) + 1e-5)
    if detach_scale:
        w = W['w']; sd = w * sd.detach() + (1 - w) * sd
    return (x - mu) / sd


Engine.norm = staticmethod(norm)
torch.set_grad_enabled(False)
E = Engine(); ds = load_dataset('NeelNanda/pile-10k', split='train')
read_L = [8, 9, 10, 11]
EXCL = set(range(9200, 9210)) | set(range(9300, 9310)) | set(range(100, 140)) | set(range(5000, 7213))


def resid_norms(ids):
    T = len(ids); x = E.wte[ids] + E.wpe[:T]; out = []
    for l, B in enumerate(E.blocks):
        out.append(x.norm(dim=-1))
        h = E.norm(x, False) * B['g1'] + B['b1']; x = x + E.attn(h, B, False)
        z = E.norm(x, False)
        x = x + VE.gelu_new((z * B['g2'] + B['b2']) @ B['Wi'] + B['bi']) @ B['Wm'] + B['bm']
    return out


docs = []; skipped = []; i = 8000
while len(docs) < 200:
    if i in EXCL: i += 1; continue
    raw = E.tok(ds[i]['text'], return_tensors='pt').input_ids[0]
    if len(raw) < 128: i += 1; continue
    ids = torch.cat([torch.tensor([50256]), raw[:127]])
    nr = resid_norms(ids); rng = np.random.default_rng(1105 + i); ok = None
    for d in range(10):
        t = int(rng.integers(4, 127))
        if not any(float(nr[L][t]) > 10 * float(nr[L][1:128].median()) for L in read_L): ok = (i, t, d); break
    (docs.append(ok) if ok else skipped.append(i)); i += 1
print('docs', docs[0], docs[-1], 'skipped', skipped, flush=True)

cells = [p for r in range(4) for p in itertools.combinations(('ln', 'attn', 'tc'), r)]
rows = []; t0 = time.time()
for k, (doc, t, nd) in enumerate(docs):
    raw = E.tok(ds[doc]['text'], return_tensors='pt').input_ids[0]
    ids = torch.cat([torch.tensor([50256]), raw[:127]])[:t + 1]
    tok = int(torch.cat([torch.tensor([50256]), raw[:127]])[t + 1])
    r = {'doc': doc, 't': t, 'redraws': nd}
    for L in read_L:
        gr, _ = E.grad_g(ids, L, t, tok, graph=False)
        for p in cells[1:]:
            g, _ = E.grad_g(ids, L, t, tok, graph=True, parts=p)
            r[f'c_{L}_{"+".join(p)}'] = float(torch.nn.functional.cosine_similarity(g, gr, 0))
            if p == ('ln',): r[f'ratio_{L}'] = float(g.norm() / gr.norm()); gf = g
        # tick: g_pred = g_frz - dG/dw at w=1 (LN-only cell), exact JVP by double backward
        w = torch.tensor(1.0, requires_grad=True); W['w'] = w
        with torch.enable_grad():
            delta = torch.zeros(len(ids), 768, requires_grad=True)
            lo = E.forward(ids, L=L, graph=True, delta=delta, parts=('ln',))
            y = lo[t, tok] - lo[t].mean()
            g, = torch.autograd.grad(y, delta, create_graph=True); g = g[t]
            u = torch.zeros_like(g, requires_grad=True)
            s, = torch.autograd.grad(g, w, grad_outputs=u, create_graph=True)
            dg, = torch.autograd.grad(s, u, allow_unused=True)
        W['w'] = 1.0
        dg = torch.zeros_like(gf) if dg is None else dg.detach()
        gp = g.detach() - dg
        r[f'cpred_{L}'] = float(torch.nn.functional.cosine_similarity(gf, gp, 0))
        r[f'cpredreal_{L}'] = float(torch.nn.functional.cosine_similarity(gp, gr, 0))
    rows.append(r)
    if k % 20 == 0: print(k, round(time.time() - t0), {L: round(r[f'c_{L}_ln'], 3) for L in read_L}, flush=True)
with open('vera_rows_gpt2_A.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
json.dump({'skipped': skipped, 'sec': time.time() - t0}, open('vera_meta_gpt2_A.json', 'w'))
print('done', time.time() - t0)
