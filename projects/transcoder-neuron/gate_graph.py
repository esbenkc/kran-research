"""G7: is g_graph really the circuit-tracer graph's gradient? Independent check by finite differences
through an EXPLICITLY frozen replacement model (stored clean attention patterns, LN scales and error nodes;
downstream MLP_l := TC_l(x_n) + err_l). The frozen model is piecewise linear in the push, so a small central
difference must equal g_graph . v.

  env/.venv/bin/python gate_graph.py data/pushes/L8_smoke_pushes.csv   -> data/gate_graph_L8.json
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

from common import DEV, GPT2, centred

path = sys.argv[1]
P = pd.read_csv(path)
L = int(path.split("/")[-1].split("_")[0][1:])
P = P[P.group != "ORACLE"].head(48).reset_index(drop=True)
W = torch.tensor(np.load("data/windows.npy"), dtype=torch.long)
m = GPT2()
tc = m.tc(L)
for l in range(L + 1, 12):
    m.tc(l)
B = len(P)
Tm = int(P.t.max()) + 1
ids = W[P.window.values, :Tm].to(DEV)
rows = torch.arange(B, device=DEV)
tpos = torch.tensor(P.t.values, device=DEV)
nxt = W[P.window.values, P.t.values + 1].to(DEV)
EPS = 1e-5


def stats(x):
    mu = x.mean(-1, keepdim=True)
    sd = ((x - mu).pow(2).mean(-1, keepdim=True) + EPS).sqrt()
    return mu, sd


with torch.no_grad():
    _, cap = m.forward(ids, capture={f"xn{L}"}, stop_after=L)
    R = cap[f"resid_pre{L}"]
    # clean pass from L, storing the frozen quantities
    fr = {}
    x = R
    for l in range(L, 12):
        b = m.blocks[l]
        mu, sd1 = stats(x)
        h = (x - mu) / sd1 * b["ln_1.weight"] + b["ln_1.bias"]
        q, k, v = (h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]).split(768, -1)
        sh = lambda z: z.view(B, Tm, 12, 64).transpose(1, 2)
        s = (sh(q) @ sh(k).transpose(-1, -2)) / 8.0
        s = s.masked_fill(~torch.ones(Tm, Tm, dtype=torch.bool, device=DEV).tril(), float("-inf"))
        pat = torch.softmax(s, -1)
        x = x + ((pat @ sh(v)).transpose(1, 2).reshape(B, Tm, 768) @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"])
        mu2, sd2 = stats(x)
        xn = (x - mu2) / sd2
        mo = m.mlp(b, xn)
        fr[l] = dict(sd1=sd1, pat=pat, sd2=sd2, err=mo - m.tc(l)(xn) if l > L else None, xn=xn, mo=mo)
        x = x + mo
    _, sdf = stats(x)
    xL = fr[L]["xn"][rows, tpos]

    def frozen_y(dv):
        """replacement model with everything downstream of mlp_out_L frozen; dv added to mlp_out_L at t."""
        x = R + 0
        b = m.blocks[L]
        # block L attention is unchanged by a push at its MLP output: reuse the clean residual after it
        mo = fr[L]["mo"].index_put((rows, tpos), fr[L]["mo"][rows, tpos] + dv)
        mu, _ = stats(x)
        h = (x - mu) / fr[L]["sd1"] * b["ln_1.weight"] + b["ln_1.bias"]
        q, k, v = (h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]).split(768, -1)
        sh = lambda z: z.view(B, Tm, 12, 64).transpose(1, 2)
        x = x + ((fr[L]["pat"] @ sh(v)).transpose(1, 2).reshape(B, Tm, 768) @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"])
        x = x + mo
        for l in range(L + 1, 12):
            b = m.blocks[l]
            mu = x.mean(-1, keepdim=True)
            h = (x - mu) / fr[l]["sd1"] * b["ln_1.weight"] + b["ln_1.bias"]
            q, k, v = (h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]).split(768, -1)
            x = x + ((fr[l]["pat"] @ sh(v)).transpose(1, 2).reshape(B, Tm, 768) @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"])
            xn = (x - x.mean(-1, keepdim=True)) / fr[l]["sd2"]
            x = x + m.tc(l)(xn) + fr[l]["err"]
        xr = x[rows, tpos]
        h = (xr - xr.mean(-1, keepdim=True)) / sdf[rows, tpos] * m.lnf_w + m.lnf_b
        return centred(h @ m.wte.T, nxt)

    y0 = frozen_y(torch.zeros(B, 768, device=DEV))
    yreal0 = centred(m.forward(ids, read_pos=(rows, tpos))[0], nxt)
    e = tc.W_enc[:, torch.tensor(P.uid.values, device=DEV)].T
    pre = ((xL - tc.b_dec) * e).sum(-1) + tc.b_enc[torch.tensor(P.uid.values, device=DEV)]
    du = -pre[:, None] * e / (e * e).sum(-1, keepdim=True)
    dt = tc(xL + du) - tc(xL)

dv = torch.zeros(B, 768, device=DEV, requires_grad=True)
lg, _ = m.forward(ids, start=(L, R), mlp_out_delta={L: (rows, tpos, dv)}, linearise=L, read_pos=(rows, tpos))
centred(lg, nxt).sum().backward()
gg = dv.grad
with torch.no_grad():
    out = {"frozen_clean_minus_real_maxabs": (y0 - yreal0).abs().max().item()}
    for hname, hh in [("h1e-3", 1e-3), ("h1e-2", 1e-2), ("h1", 1.0)]:
        fd = (frozen_y(hh * dt) - frozen_y(-hh * dt)) / (2 * hh)
        an = (gg * dt).sum(-1)
        out[f"fd_{hname}_vs_ggraph_dt_maxrel"] = ((fd - an).abs() / an.abs().clamp(min=1e-4)).max().item()
        out[f"fd_{hname}_vs_ggraph_dt_median_rel"] = ((fd - an).abs() / an.abs().clamp(min=1e-4)).median().item()
    out["one_sided_finite_dt_vs_pred_R2"] = None
    yf = frozen_y(dt) - y0
    an = (gg * dt).sum(-1)
    out["frozen_model_finite_push_vs_ggraph_R2"] = (1 - ((yf - an) ** 2).sum() / ((yf - yf.mean()) ** 2).sum()).item()
    out["rows"] = [{"uid": int(P.uid[i]), "t": int(P.t[i]), "ggraph_dt": float(an[i]), "frozen_fd": float(yf[i]),
                    "csv_total_g": float(P.total_g[i])} for i in range(B)]
    out["csv_total_g_vs_recomputed_maxabs"] = float(np.abs(P.total_g.values - an.cpu().numpy()).max())
for k, v in out.items():
    if k != "rows":
        print(k, v)
json.dump(out, open(f"data/gate_graph_L{L}.json", "w"), indent=1)
