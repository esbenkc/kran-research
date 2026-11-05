"""Vera's independent rebuild of the PREREG (kran-research 0b4e178b, projects/frozen-norm-gradients/PREREG.md)
status cells for an RMSNorm model, from scratch: doc selection, read positions, massive-activation redraws,
g_real, g_frz (residual norms frozen, q/k-norm live), first-order tick, |g_frz|/|g_real|, derangement null.
Usage: python vera_rebuild_rms.py qwen3|smol SET(A|B|C) [--qk]
Writes vera_rows_<model>_<set>.jsonl (one line per token, incl. vectors as float lists? no: scalars only)
and vera_vec_<model>_<set>.pt (g_real, g_frz, g_pred per layer)."""
import sys, json, time, torch, numpy as np
from datasets import load_dataset
import vera_hf as V

model = sys.argv[1]; SET = sys.argv[2]; QK = '--qk' in sys.argv
NL = {'qwen3': 28, 'smol': 30}[model]
LG = [8, 9, 10]
status_L = [int(np.floor(NL * l / 12 + 0.5)) for l in LG]   # round half up
read_L = status_L + [NL - 1]
EXCL = set(range(9200, 9210)) | set(range(9300, 9310)) | set(range(100, 140)) | set(range(5000, 7213))
torch.set_grad_enabled(False)
E = V.HF(model)
ds = load_dataset('NeelNanda/pile-10k', split='train')

# usable docs from 8000 upward; sets A,B,C = consecutive blocks of 200 (skipped all-flagged docs replaced)
picked = []  # (doc, t, n_redraws)
skipped = []
need = {'A': 200, 'B': 400, 'C': 600}[SET]
i = 8000
while len(picked) < need:
    if i in EXCL: i += 1; continue
    ids = E.ids(ds[i]['text'])
    if ids is None: i += 1; continue
    norms = E.resid_norms(ids)  # list over blocks: [128]
    rng = np.random.default_rng(1105 + i)
    ok = None
    for draw in range(10):
        t = int(rng.integers(4, 127))
        flag = any(float(norms[L][t]) > 10 * float(norms[L][1:128].median()) for L in read_L)
        if not flag: ok = (i, t, draw); break
    if ok is None: skipped.append(i)
    else: picked.append(ok)
    i += 1
lo_ = {'A': 0, 'B': 200, 'C': 400}[SET]
docs = picked[lo_:lo_ + 200]
print('read layers', read_L, 'docs', docs[0], docs[-1], 'skipped', skipped, flush=True)

rows = []; vec = {L: {'real': [], 'frz': [], 'pred': []} for L in read_L}
t0 = time.time()
for k, (doc, t, nd) in enumerate(docs):
    ids = E.ids(ds[doc]['text'])
    gr = E.grads(ids, t, read_L)
    if QK:
        gf = E.grads(ids, t, read_L, w=1.0, wqk=1.0); gp = {L: torch.full_like(gf[L], float('nan')) for L in read_L}
    else:
        gf, gp = E.frozen_and_tick(ids, t, read_L)
    r = {'doc': doc, 't': t, 'redraws': nd, 'tok_next': int(ids[t + 1])}
    for L in read_L:
        r[f'c_{L}'] = V.cos(gf[L], gr[L]); r[f'cpred_{L}'] = V.cos(gf[L], gp[L]) if not QK else None
        r[f'cpredreal_{L}'] = V.cos(gp[L], gr[L]) if not QK else None
        r[f'ratio_{L}'] = float(gf[L].norm() / gr[L].norm())
        for nm, g in (('real', gr), ('frz', gf), ('pred', gp)): vec[L][nm].append(g[L].float())
    rows.append(r)
    if k % 20 == 0: print(k, round(time.time() - t0), {L: round(r[f'c_{L}'], 3) for L in read_L}, flush=True)
tag = f'{model}_{SET}' + ('_qk' if QK else '')
with open(f'vera_rows_{tag}.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
torch.save({L: {nm: torch.stack(v) for nm, v in d.items()} for L, d in vec.items()}, f'vera_vec_{tag}.pt')
json.dump({'read_L': read_L, 'skipped': skipped, 'n': len(rows), 'sec': time.time() - t0}, open(f'vera_meta_{tag}.json', 'w'))
print('done', time.time() - t0)
