"""Screen v2 (own code): linear (eps) vs finite, f-only vs TOTAL, robust stats, random-direction control,
and the paper's rank-1 alignment. Scalar = direct-path logit proxy of clean top-1 token."""
import torch, json, numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from vera_lib import *

torch.set_grad_enabled(False)
torch.manual_seed(1)
m, tok = load_gpt2()
ds = load_dataset('NeelNanda/pile-10k', split='train')
caps = {}
for L in range(8, 12):
    m.transformer.h[L].ln_2.register_forward_pre_hook(lambda mod, inp, L=L: caps.__setitem__(L, inp[0].detach()))
m.transformer.ln_f.register_forward_pre_hook(lambda mod, inp: caps.__setitem__('f', inp[0].detach()))
Zs = {L: [] for L in range(8, 12)}; TOP = []; SD = []
for i in range(9100, 9160):
    ids = tok(ds[i]['text'], return_tensors='pt').input_ids[:, :128]
    if ids.shape[1] < 16: continue
    lo = m(ids).logits[0]
    TOP.append(lo.argmax(-1)[1:]); xf = caps['f'][0]
    SD.append(torch.sqrt(xf.var(-1, unbiased=False) + 1e-5)[1:])
    for L in range(8, 12):
        x = caps[L][0]
        Zs[L].append(((x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5))[1:])
TOP = torch.cat(TOP); SD = torch.cat(SD)
WU = m.lm_head.weight.detach(); gf = m.transformer.ln_f.weight.detach()
V = WU[TOP] * gf; V = V - V.mean(1, keepdim=True); V = V / SD[:, None]


def dgelu(x):
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        y = gelu_new(x).sum(); g, = torch.autograd.grad(y, x)
    return g


def stats(p, y):
    p, y = np.asarray(p), np.asarray(y)
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rel = np.median(np.abs(p - y) / (np.abs(y) + 1e-9))
    return dict(R2=round(float(r2), 3), sp=round(float(spearmanr(p, y)[0]), 3), medrel=round(float(rel), 3), n=len(p))


res = {}
for L in range(8, 12):
    Z = torch.cat(Zs[L])
    tc = load_tc(L)
    Win, bin_, Wout, bout = mlp_in_z(m, L)
    e, d = tc['W_enc'], tc['W_dec']
    en = e / e.norm(dim=0, keepdim=True); wn = Win / Win.norm(dim=0, keepdim=True)
    dn = d / d.norm(dim=1, keepdim=True); on = Wout / Wout.norm(dim=1, keepdim=True)
    G = (Win.T @ Win) * (Wout @ Wout.T)
    jc = torch.cat([torch.minimum(en[:, i:i + 4096].T @ wn, dn[i:i + 4096] @ on.T).max(1).values for i in range(0, 24576, 4096)])
    A, _ = tc_forward(tc, Z); act = A > 0
    pre = Z @ Win + bin_; gp = dgelu(pre)
    mlp = lambda z: gelu_new(z @ Win + bin_) @ Wout
    base = mlp(Z); _, tcb = tc_forward(tc, Z)

    def run(kind, ids):
        acc = {k: ([], []) for k in ['fonly_lin', 'total_lin', 'fonly_fin', 'total_fin', 'rand_total_lin']}
        al = []
        for f in ids.tolist():
            if kind == 'oracle':
                ts = torch.where(pre[:, f] > 0)[0]; ts = ts[torch.randperm(len(ts))[:8]]
                ef, df, a = Win[:, f], Wout[f], pre[ts, f]
                fpred = -gelu_new(a)[:, None] * df[None]
            else:
                ts = torch.where(act[:, f])[0]; ts = ts[torch.randperm(len(ts))[:8]]
                ef, df, a = e[:, f], d[f], A[ts, f]
                fpred = -a[:, None] * df[None]
            du = -a[:, None] * ef[None] / (ef @ ef)
            vv = V[ts]
            rl = ((gp[ts] * (du @ Win)) @ Wout)  # J_MLP du
            tl = ((act[ts].float() * (du @ e)) @ d)  # J_TC du
            rf = mlp(Z[ts] + du) - base[ts]
            _, t1 = tc_forward(tc, Z[ts] + du); tf = t1 - tcb[ts]
            r = torch.randn_like(du); r = r / r.norm(dim=1, keepdim=True) * du.norm(dim=1, keepdim=True)
            rrl = (gp[ts] * (r @ Win)) @ Wout; rtl = (act[ts].float() * (r @ e)) @ d
            fl = -(a * gp[ts, f])[:, None] * df[None] if kind == 'oracle' else fpred
            s = lambda X: (X * vv).sum(1).tolist()
            for k, (p, y) in dict(fonly_lin=(fl, rl), total_lin=(tl, rl), fonly_fin=(fpred, rf), total_fin=(tf, rf), rand_total_lin=(rtl, rrl)).items():
                acc[k][0].extend(s(p)); acc[k][1].extend(s(y))
            # paper rank-1 alignment, averaged over contexts: d^T J e/(|d||e||J|_F), J = Wout^T diag(gp) Win^T
            g_ = gp[ts]  # [k,3072]
            num = (g_ * (ef @ Win)[None] * (Wout @ df)[None]).sum(1)
            jf = torch.sqrt(((g_ @ G) * g_).sum(1))
            al.extend((num / (ef.norm() * df.norm() * jf)).tolist())
        o = {k: stats(*v) for k, v in acc.items()}
        o['rank1_align_mean'] = round(float(np.mean(al)), 4)
        return o

    out = {}
    for name, mask in [('neuron', jc >= 0.9), ('rest', jc < 0.9)]:
        cand = torch.where(mask & act.any(0))[0]; out[name] = run('feat', cand[torch.randperm(len(cand))[:200]])
    cand = torch.where((pre > 0).any(0))[0]; out['oracle'] = run('oracle', cand[torch.randperm(len(cand))[:200]])
    res[L] = out
    print(L, json.dumps(out), flush=True)
json.dump(res, open('vera_screen2.json', 'w'), indent=1)
