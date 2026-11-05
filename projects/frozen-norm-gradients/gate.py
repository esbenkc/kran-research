"""Engine gates on OLD data only (parent windows, docs 5000-7212; never the new calibration/test docs).
E1 real-cell logits == HF GPT2 logits at the read positions.
E2 all-frozen cell == parent common.py linearise=L gradient (independent code path).
E3 real-cell gradient == central finite difference along 3 random directions.
E4 ln='token' (u = own x_hat) == ln='live' (cos, all else real).
E5 ln='fixed' with u=0 == ln='frozen'.
-> gate.json"""
import json
import sys

import numpy as np
import torch

from engine import DEV, GPT2, GRAPH, REAL, NORMS

PARENT = "/Users/esben/work/theory-1104-transcoder-neuron/kran-research/projects/transcoder-neuron"
W = torch.tensor(np.load(f"{PARENT}/data/windows.npy")[:8]).long().to(DEV)
rng = np.random.default_rng(0)
pos = torch.tensor(rng.integers(10, 126, size=8), device=DEV)
tok = W[torch.arange(8), pos + 1]
m = GPT2()
res = {}
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)

from transformers import GPT2LMHeadModel
hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(DEV).eval()
with torch.no_grad():
    lh = hf(W).logits[torch.arange(8), pos]
    lo = m.forward(W, 8, REAL, torch.arange(8, device=DEV), pos)
res["E1_maxabs_logit_vs_HF"] = (lo - lh).abs().max().item()
del hf

sys.path.insert(0, PARENT)
import common as pc  # parent engine
pm = pc.GPT2()
for L in (8, 9, 10, 11):
    g_me, _, _ = m.readout(W, L, GRAPH, pos, tok)
    rows = torch.arange(8, device=DEV)
    dv = torch.zeros(8, 768, device=DEV, requires_grad=True)
    with torch.enable_grad():
        lg, _ = pm.forward(W, mlp_out_delta={L: (rows, pos, dv)}, linearise=L, read_pos=(rows, pos))
        y = pc.centred(lg, tok)
        g_p, = torch.autograd.grad(y.sum(), dv)
    res[f"E2_L{L}_maxrel_vs_parent"] = ((g_me - g_p).norm(dim=-1) / g_p.norm(dim=-1)).max().item()
    g_r, _, _ = m.readout(W, L, REAL, pos, tok)
    res[f"E2b_L{L}_median_cos_graph_vs_real_n8"] = cos(g_me, g_r).median().item()
    # E3 finite difference of the real model
    fd = []
    for k in range(3):
        v = torch.tensor(rng.standard_normal((8, 768)), dtype=torch.float32, device=DEV)
        v = v / v.norm(dim=-1, keepdim=True)
        h = 1e-2
        with torch.no_grad():
            lp = m.forward(W, L, REAL, rows, pos, h * v)
            lm = m.forward(W, L, REAL, rows, pos, -h * v)
        yp = lp.gather(-1, tok[:, None])[:, 0] - lp.mean(-1)
        ym = lm.gather(-1, tok[:, None])[:, 0] - lm.mean(-1)
        num = (yp - ym) / (2 * h)
        ana = (g_r * v).sum(-1)
        fd.append(((num - ana).abs() / g_r.norm(dim=-1)).max().item())  # relative to |g|
    res[f"E3_L{L}_max_fd_err_over_gnorm"] = max(fd)
    tokcfg = dict(REAL, ln="token")
    g_t, _, _ = m.readout(W, L, tokcfg, pos, tok)
    g_l, _, _ = m.readout(W, L, REAL, pos, tok)
    res[f"E4_L{L}_min_cos_token_vs_live"] = cos(g_t, g_l).min().item()
    z = {n: torch.zeros(768, device=DEV) for n in NORMS}
    g_f0, _, _ = m.readout(W, L, dict(GRAPH, ln="fixed"), pos, tok, u=z)
    res[f"E5_L{L}_maxabs_fixed0_vs_frozen"] = (g_f0 - g_me).abs().max().item()
    print(L, {k: round(v, 6) for k, v in res.items() if f"L{L}_" in k}, flush=True)
json.dump(res, open("data/gate_gpt2.json", "w"), indent=1)
print(json.dumps(res, indent=1))
