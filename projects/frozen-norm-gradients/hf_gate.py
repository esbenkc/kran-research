"""Gates for hf_engine (no status quantity is computed; no pile text is used).
H1 patched live RMSNorm == the library RMSNorm forward (random input, every norm).
H2 real gradient == central finite difference along 3 random directions (one fixed sentence, 2 positions).
H3 with ONLY the final norm downstream (L = n-1), frozen-final-norm first-order is exact: g_live - g_frz equals the
   rank-1 term computed by hand from the frozen gradient at the final norm (checks the frozen switch itself).
-> hf_gate_<name>.json"""
import json
import os
import sys
import time

import torch

from hf_engine import DEV, HFEngine

repo = sys.argv[1]
name = repo.split("/")[-1]
t0 = time.time()
E = HFEngine(repo, dtype=torch.float64 if os.environ.get("GATE_FP64") else torch.float32)
res = {"repo": repo, "device": str(DEV), "n_layers": E.n, "n_norms": len(E.norms)}
g = torch.Generator().manual_seed(0)
err = 0.0
for n, mod in E.norms.items():
    d = mod.weight.shape[0]
    x = (torch.randn(2, 5, d, generator=g) * 3).to(DEV, E.m.dtype)
    ref = type(mod).forward(mod, x)
    err = max(err, (mod.forward(x) - ref).abs().max().item())
res["H1_maxabs_patched_vs_library"] = err

text = "The cat sat on the mat because it was tired after a long day of chasing mice around the old barn."
ids = torch.tensor([E.tok(text)["input_ids"][:24]], device=DEV)
for L in (E.n // 2, E.n - 1):
    for t in (10, 20):
        pos = torch.tensor([t], device=DEV)
        tok = ids[0, t + 1:t + 2]
        gr, _ = E.readout(ids, L, pos, tok)
        rel = []
        for k in range(3):
            v = torch.randn(1, gr.shape[1], generator=g).to(DEV, E.m.dtype)
            v = v / v.norm()
            h = 1e-2  # eager attention runs softmax in float32 even for fp64 weights: smaller h hits round-off
            ys = []
            for s in (1, -1):
                E._delta, E._hL = (torch.arange(1, device=DEV), pos, s * h * v), L
                with torch.no_grad():
                    hh = E.m.model(input_ids=ids, use_cache=False).last_hidden_state
                    lg = E.m.lm_head(hh[0, t][None])
                E._delta, E._hL = None, None
                ys.append((lg[0, tok[0]] - lg[0].mean()).item())
            num = (ys[0] - ys[1]) / (2 * h)
            ana = (gr * v).sum().item()
            rel.append(abs(num - ana) / gr.norm().item())  # error per unit direction, relative to |g|
        res[f"H2_L{L}_t{t}_max_fd_err_over_gnorm"] = max(rel)
L = E.n - 1
pos = torch.tensor([20], device=DEV)
tok = ids[0, 21:22]
gl, _ = E.readout(ids, L, pos, tok)
gf, _ = E.readout(ids, L, pos, tok, frozen=E.downstream(L))
# by hand: x = residual at the final norm input (y depends on mlp_out_L only through x). Frozen: grad_x = r gamma*w,
# w = dy/d(final norm output). Live adds -(r^3/d) x (x . gamma*w) = -(r^2/d) x (x . grad_x_frozen).
cap = {}
hk = E.m.model.norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0][0, 20].detach()))
with torch.no_grad():
    E.m.model(input_ids=ids, use_cache=False)
hk.remove()
x = cap["x"]
d = x.shape[0]
r = torch.rsqrt(x.pow(2).mean() + E.m.model.norm.variance_epsilon)
gam = E.m.model.norm.weight
pred = gf - (r ** 2 / d) * x * (x * gf).sum()
res["H3_maxrel_live_vs_byhand"] = ((gl - pred).norm() / gl.norm()).item()
res["seconds"] = time.time() - t0
res["dtype"] = str(E.m.dtype)
json.dump(res, open(f"data/hf_gate_{name}{'_fp64' if os.environ.get('GATE_FP64') else ''}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
