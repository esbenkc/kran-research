"""Engine gates (run before any arm): python gate.py -> data/gate.json
G1 own forward == HF GPT2LMHeadModel logits (max abs diff).
G2 linearised forward (any L0) has the same VALUE as the real forward.
G3 transcoder loader: fraction of variance unexplained (FVU) of TC_L(x_n) vs the centred MLP output,
   and next-token loss with MLP_L replaced by TC_L (vs clean), on held-out pile-10k text.
G4 a mlp_in_delta of zero and a mlp_out_delta of zero change nothing; an mlp_out_delta of c*ones changes
   nothing (LN mean invariance), so a centred TC output is a valid stand-in.
"""
import json
import os

import torch

from common import DEV, GPT2, centred
from data import heldout_batch

torch.manual_seed(0)
os.makedirs("data", exist_ok=True)
m = GPT2()
ids = heldout_batch(8, 64, seed=0).to(DEV)
res = {}
with torch.no_grad():
    lg, cap = m.forward(ids, capture={f"xn{L}" for L in range(8, 12)} | {f"mlp{L}" for L in range(8, 12)})
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(DEV).eval()
    lh = hf(ids).logits
    res["G1_maxabs_logit_diff_vs_HF"] = (lg - lh).abs().max().item()
    del hf
    for L0 in [7, 8, 10, 11]:
        ll, _ = m.forward(ids, linearise=L0)
        res[f"G2_linearise{L0}_maxabs"] = (ll - lg).abs().max().item()
    ce = torch.nn.functional.cross_entropy
    base_loss = ce(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item()
    res["clean_loss"] = base_loss
    for L in range(8, 12):
        tc = m.tc(L)
        xn, mo = cap[f"xn{L}"][:, 1:], cap[f"mlp{L}"][:, 1:]  # position 0 (BOS) excluded from FVU
        moc = mo - mo.mean(-1, keepdim=True)
        rec = tc(xn)
        fvu = ((rec - moc) ** 2).sum() / ((moc - moc.mean((0, 1))) ** 2).sum()
        res[f"G3_L{L}_fvu"] = fvu.item()
        res[f"G3_L{L}_L0"] = (tc.acts(xn) > 0).float().sum(-1).mean().item()
        rows = torch.arange(ids.shape[0], device=DEV).repeat_interleave(ids.shape[1] - 1)
        pos = torch.arange(1, ids.shape[1], device=DEV).repeat(ids.shape[0])  # splice all but BOS
        dv = (rec - mo).reshape(-1, 768)
        ls, _ = m.forward(ids, mlp_out_delta={L: (rows, pos, dv)})
        res[f"G3_L{L}_spliced_loss"] = ce(ls[:, :-1].reshape(-1, ls.shape[-1]), ids[:, 1:].reshape(-1)).item()
    r = torch.arange(8, device=DEV); p = torch.full((8,), 40, device=DEV)
    l0, _ = m.forward(ids, mlp_in_delta={9: (r, p, torch.zeros(8, 768, device=DEV))})
    res["G4_zero_in_delta"] = (l0 - lg).abs().max().item()
    l1, _ = m.forward(ids, mlp_out_delta={9: (r, p, 3.0 * torch.ones(8, 768, device=DEV))})
    res["G4_const_out_delta"] = (l1 - lg).abs().max().item()
# G5: the chunked frozen-error node (_TCLin) gives the same gradient as TC(x) + (MLP(x) - TC(x)).detach()
from common import _TCLin
ids2 = ids[:4, :48]
rows4 = torch.arange(4, device=DEV); pos4 = torch.full((4,), 40, device=DEV)
nxt4 = ids2[:, 41]
def gfun(explicit):
    dv = torch.zeros(4, 768, device=DEV, requires_grad=True)
    if explicit:
        orig = _TCLin.apply
        def expl(xn, mr, We, be, bd, Wd):
            t = torch.relu((xn - bd) @ We + be) @ Wd + m.tcs[11].b_dec_out * 0 + 0
            return t + (mr - t).detach()
        _TCLin.apply = staticmethod(expl)
    try:
        lg, _ = m.forward(ids2, mlp_out_delta={8: (rows4, pos4, dv)}, linearise=8, read_pos=(rows4, pos4))
        centred(lg, nxt4).sum().backward()
    finally:
        if explicit:
            _TCLin.apply = orig
    return dv.grad
ga, gb = gfun(False), gfun(True)
res["G5_TCLin_vs_explicit_relerr"] = ((ga - gb).norm() / gb.norm()).item()
# G6: real-model gradient vs central finite difference along a random direction
dv0 = torch.zeros(4, 768, device=DEV, requires_grad=True)
lg, _ = m.forward(ids2, mlp_out_delta={8: (rows4, pos4, dv0)}, read_pos=(rows4, pos4))
centred(lg, nxt4).sum().backward()
v = torch.randn(4, 768, device=DEV); v = v / v.norm(dim=-1, keepdim=True)
with torch.no_grad():
    h = 0.05
    yp = centred(m.forward(ids2, mlp_out_delta={8: (rows4, pos4, h * v)}, read_pos=(rows4, pos4))[0], nxt4)
    ym = centred(m.forward(ids2, mlp_out_delta={8: (rows4, pos4, -h * v)}, read_pos=(rows4, pos4))[0], nxt4)
fd = (yp - ym) / (2 * h); an = (dv0.grad * v).sum(-1)
res["G6_grad_vs_fd_maxrel"] = ((fd - an).abs() / an.abs().clamp(min=1e-3)).max().item()
for k, v in res.items():
    print(f"{k:32s} {v:.3e}")
json.dump(res, open("data/gate.json", "w"), indent=1)
