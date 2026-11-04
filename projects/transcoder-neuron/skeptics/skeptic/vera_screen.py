"""Pre-data READ-arm screen (own code). Finite turn-off dose at the MLP input, f-only prediction.
Scalar = dMLP_out . v_t, v_t = centered direct-path unembed of the clean top-1 token (no later layers):
a direct-logit proxy for g. Reports R^2 about y=x per class, plus oracle-neuron ceiling and shuffled-decoder floor."""
import torch, json, numpy as np
from datasets import load_dataset
from vera_lib import *

torch.set_grad_enabled(False)
torch.manual_seed(0)
m, tok = load_gpt2()
ds = load_dataset('NeelNanda/pile-10k', split='train')
caps = {}
for L in range(8, 12):
    m.transformer.h[L].ln_2.register_forward_pre_hook(lambda mod, inp, L=L: caps.__setitem__(L, inp[0].detach()))
caps['f'] = None
m.transformer.ln_f.register_forward_pre_hook(lambda mod, inp: caps.__setitem__('f', inp[0].detach()))
Zs = {L: [] for L in range(8, 12)}; TOP = []; SD = []
for i in range(9000, 9030):
    ids = tok(ds[i]['text'], return_tensors='pt').input_ids[:, :128]
    if ids.shape[1] < 16: continue
    lo = m(ids).logits[0]
    TOP.append(lo.argmax(-1)[1:])
    xf = caps['f'][0]
    SD.append(torch.sqrt(xf.var(-1, unbiased=False) + 1e-5)[1:])
    for L in range(8, 12):
        x = caps[L][0]
        Zs[L].append(((x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5))[1:])
TOP = torch.cat(TOP); SD = torch.cat(SD)
WU = m.lm_head.weight.detach()  # [V,768]
gf = m.transformer.ln_f.weight.detach()
V = (WU[TOP] * gf) ; V = V - V.mean(1, keepdim=True); V = V / SD[:, None]  # [T,768]


def r2yx(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


res = {}
for L in range(8, 12):
    Z = torch.cat(Zs[L]); T = Z.shape[0]
    tc = load_tc(L)
    Win, bin_, Wout, bout = mlp_in_z(m, L)
    mlp = lambda z: gelu_new(z @ Win + bin_) @ Wout
    e, d = tc['W_enc'], tc['W_dec']
    en = e / e.norm(dim=0, keepdim=True); wn = Win / Win.norm(dim=0, keepdim=True)
    dn = d / d.norm(dim=1, keepdim=True); on = Wout / Wout.norm(dim=1, keepdim=True)
    jc = torch.minimum(en.T @ wn, dn @ on.T).max(1).values
    A, _ = tc_forward(tc, Z)
    act = (A > 0)
    cls = {'neuron': (jc >= 0.9), 'rest': (jc < 0.9)}
    base = mlp(Z)
    out = {}
    for name, mask in cls.items():
        cand = torch.where(mask & act.any(0))[0]
        feats = cand[torch.randperm(len(cand))[:300]]
        P, Y, Pshuf, Ytot, jcs = [], [], [], [], []
        for f in feats.tolist():
            ts = torch.where(act[:, f])[0]
            ts = ts[torch.randperm(len(ts))[:8]]
            ef = e[:, f]
            du = -A[ts, f][:, None] * ef[None] / (ef @ ef)
            dmlp = mlp(Z[ts] + du) - base[ts]
            pred = -A[ts, f][:, None] * d[f][None]
            fs = int(torch.randint(0, d.shape[0], (1,)))
            preds = -A[ts, f][:, None] * d[fs][None]
            vv = V[ts]
            P += (pred * vv).sum(1).tolist(); Y += (dmlp * vv).sum(1).tolist()
            Pshuf += (preds * vv).sum(1).tolist()
            _, r1 = tc_forward(tc, Z[ts] + du); _, r0 = tc_forward(tc, Z[ts])
            Ytot += ((r1 - r0) * vv).sum(1).tolist()
        out[name] = dict(n_feat=len(feats), n_pts=len(P), R2=r2yx(P, Y), R2_shufdec=r2yx(Pshuf, Y),
                         R2_TCtotal_vs_real=r2yx(Ytot, Y), slope=float(np.polyfit(P, Y, 1)[0]))
    # oracle neuron ceiling
    pre = Z @ Win + bin_
    nact = pre > 0
    cand = torch.where(nact.any(0))[0]; ns = cand[torch.randperm(len(cand))[:300]]
    P, Y = [], []
    for n in ns.tolist():
        ts = torch.where(nact[:, n])[0]; ts = ts[torch.randperm(len(ts))[:8]]
        w = Win[:, n]
        du = -pre[ts, n][:, None] * w[None] / (w @ w)
        dmlp = mlp(Z[ts] + du) - base[ts]
        pred = -gelu_new(pre[ts, n])[:, None] * Wout[n][None]
        vv = V[ts]
        P += (pred * vv).sum(1).tolist(); Y += (dmlp * vv).sum(1).tolist()
    out['oracle_neuron'] = dict(n_feat=len(ns), n_pts=len(P), R2=r2yx(P, Y), slope=float(np.polyfit(P, Y, 1)[0]))
    res[L] = out
    print(L, json.dumps(out), flush=True)
json.dump(res, open('vera_screen.json', 'w'), indent=1)
