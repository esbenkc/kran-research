"""Physics rebuild (own engine): recompute sampled pushes from the model and compare with Bruno's raw CSV columns.
Also re-derives the windows from pile-10k and the unit cosines from the weights."""
import json, sys, numpy as np, pandas as pd, torch
from datasets import load_dataset
from vera_engine import Engine
from vera_lib import tc_forward

torch.set_grad_enabled(False)
D = '/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron/data'
K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LAYERS = [int(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [8, 9, 10, 11]
rng = np.random.default_rng(7)
E = Engine()
W = np.load(f'{D}/windows.npy')
meta = json.load(open(f'{D}/windows_meta.json'))
ds = load_dataset('NeelNanda/pile-10k', split='train')
# window check
wok = 0
chk = rng.choice(len(W), 50, replace=False)
for i in chk:
    ids = E.tok(ds[meta['doc_index'][i]]['text'][:2000]).input_ids[:127]
    wok += int(list(W[i]) == [50256] + ids)
out = {'windows_match': f'{wok}/50'}
print(out, flush=True)


def cosines(L):
    tc = E.tc[L]; Win, _, Wout, _ = E.mz[L]
    e = tc['W_enc']; d = tc['W_dec']
    en = e / e.norm(dim=0, keepdim=True); wn = Win / Win.norm(dim=0, keepdim=True)
    dn = d / d.norm(dim=1, keepdim=True); on = Wout / Wout.norm(dim=1, keepdim=True)
    return en, wn, dn, on


rows = []
for L in LAYERS:
    U = pd.read_csv(f'{D}/pushes/L{L}_units.csv'); P = pd.read_csv(f'{D}/pushes/L{L}_pushes.csv')
    tc = E.tc[L]; Win, bin_, Wout, _ = E.mz[L]
    en, wn, dn, on = cosines(L)
    # unit cosines for all non-ORACLE units
    uf = U[U.group != 'ORACLE']
    idx = torch.tensor(uf.uid.values)
    ce = (en[:, idx].T @ wn).abs(); nstar = ce.argmax(1)
    enc = ce.max(1).values; dec = (dn[idx] * on[nstar]).sum(1).abs()
    cos_err = float(max((enc - torch.tensor(uf.enc_cos.values)).abs().max(), (dec - torch.tensor(uf.dec_cos.values)).abs().max()))
    nstar_eq = float((nstar.numpy() == uf.same_neuron.values).mean())
    # class membership check
    joint = torch.minimum(enc, dec).numpy()
    grp_ok = float(np.mean([(g == 'NEURON') == (j >= 0.9) for g, j in zip(uf.group, joint)]))
    out[f'L{L}_units'] = dict(max_abs_cos_err=cos_err, nstar_match=nstar_eq, neuron_rule_ok=grp_ok, n=len(uf))
    print(L, out[f'L{L}_units'], flush=True)
    for g in ['NEURON', 'GAP', 'REST', 'ORACLE']:
        d = P[(P.group == g) & (P.pre_ok != 0)]
        for _, r in d.iloc[rng.choice(len(d), min(K, len(d)), replace=False)].iterrows():
            w, t, u = int(r.window), int(r.t), int(r.uid)
            ids = torch.tensor(W[w][:t + 1].astype(np.int64)); tok = int(W[w][t + 1])
            lo, cap = E.forward(ids, L=L, capture=True)
            z = cap['z'][t]
            if g == 'ORACLE':
                ef = Win[:, u]; pre = float(z @ ef + bin_[u])
            else:
                ef = tc['W_enc'][:, u]; pre = float((z - tc['b_dec']) @ ef + tc['b_enc'][u])
            du = -pre * ef / (ef @ ef)
            rv = E.mlp_local(L, z + du) - E.mlp_local(L, z)
            _, t1 = tc_forward(tc, z + du); _, t0 = tc_forward(tc, z); tv = t1 - t0
            gr, y0 = E.grad_g(ids, L, t, tok, graph=False)
            gg, _ = E.grad_g(ids, L, t, tok, graph=True)
            lo2 = E.forward(ids, L=L, zpush=(t, du))
            e2e = float(E.y(lo2, t, tok)) - y0
            fonly = (-pre * float(tc['W_dec'][u] @ gr)) if g != 'ORACLE' else np.nan
            rows.append(dict(L=L, group=g, uid=u, window=w, t=t,
                             pre_v=pre, pre_b=r.pre, du_v=float(du.norm()), du_b=r.du_norm,
                             ml_r_v=float(gr @ rv), ml_r_b=r.meas_local_r, tot_r_v=float(gr @ tv), tot_r_b=r.total_r,
                             ml_g_v=float(gg @ rv), ml_g_b=r.meas_local_g, tot_g_v=float(gg @ tv), tot_g_b=r.total_g,
                             e2e_v=e2e, e2e_b=r.e2e_real, y_v=y0, y_b=r.y_clean, tr2_v=float(((tv - rv) ** 2).sum()), tr2_b=r.tr2,
                             r2_v=float((rv ** 2).sum()), r2_b=r.r2, fonly_v=fonly, fonly_b=r.fonly_r))
        print(L, g, 'done', flush=True)
R = pd.DataFrame(rows)
R.to_csv('vera_rebuild_rows.csv', index=False)
summ = {}
for c in ['pre', 'du', 'ml_r', 'tot_r', 'ml_g', 'tot_g', 'e2e', 'y', 'tr2', 'r2', 'fonly']:
    a, b = R[c + '_v'].values, R[c + '_b'].values
    m = ~np.isnan(a)
    rel = np.abs(a[m] - b[m]) / (np.abs(b[m]) + 1e-6)
    summ[c] = dict(med_rel=float(np.median(rel)), p95_rel=float(np.quantile(rel, .95)), max_abs=float(np.abs(a[m] - b[m]).max()),
                   corr=float(np.corrcoef(a[m], b[m])[0, 1]))
out['columns'] = summ
print(json.dumps(summ, indent=1))
json.dump(out, open('vera_rebuild.json', 'w'), indent=1)
