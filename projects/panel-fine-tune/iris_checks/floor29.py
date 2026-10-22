"""Iris T5: exact output-cost floors for the FROZEN post-norm readout-29 country-token probe (panel.npz tok29).

Search variable: the pre-norm last state x (readout 28) at the country token, kept on the RMSNorm map h = g*x/rms(x)
(h = readout 29 = the exact input of the logits, tied unembedding E). Cost = exact next-word KL(p_clean || p_moved)
at that position over the full vocabulary. Adam on KL + 10*violation^2, STEPS steps; the best feasible KL is an
UPPER bound on the true per-statement minimum (as 1021 floor_exact.py / vera_bound.py).
mode 'push':  caught false eval statements, target score <= line - 0.05 (line = 2nd highest of 200 clean line-set
              true statements = the fixed-line route).
mode 'pushpanel': same, target = clean PANEL line (arg, z units) mapped to s29 via 1022 zmoments.json.
mode 'lift':  true eval statements, target score >= tau + 0.05 (tau = arg; the line-raising route).
Usage (from projects/panel-fine-tune): ../../../venv1021/bin/python iris_checks/floor29.py push OUT.json | pushpanel OUT.json ZLINE | lift OUT.json TAU [STEPS]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, '.')
from common import Acts, pool_idx, rows, threshold

mode, out = sys.argv[1], sys.argv[2]
EPS, STEPS, BS, MARGIN = 1e-6, int(sys.argv[4]) if len(sys.argv) > 4 else 600, 128, 0.05
dev = torch.device('mps')
z = np.load('data/fits/panel.npz', allow_pickle=True)
_, y, city, split = rows()
tr, line_idx, ev = pool_idx()
rws = z['tok_rows_clean']
A = Acts('clean')
X28 = A.at(28, rws).astype(np.float64)
X29 = A.at(29, rws).astype(np.float64)
snap = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*'))
with safe_open(str(snap / 'model.safetensors'), 'pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').float()
    g = f.get_tensor('model.norm.weight').float()
gn = g.double().numpy()
H = gn * X28 / np.sqrt((X28 ** 2).mean(-1, keepdims=True) + EPS)
w, b, mu, sd = z['tok29__w'].astype(np.float64), float(z['tok29__b']), z['tok29__mu'], z['tok29__sd']
ws = w / sd; bb = b - mu @ ws
s_stored = X29 @ ws + bb
s = H @ ws + bb
line = threshold(s_stored[line_idx])
info = dict(mode=mode, line_stored=line, line_recomputed=threshold(s[line_idx]),
            max_abs_score_diff_recomputed_vs_stored=float(np.abs(s - s_stored).max()))
evf, evt = ev[y[ev] == 1], ev[y[ev] == 0]
if mode == 'push':
    idx_all = evf[s_stored[evf] > line]; tgt_val = line - MARGIN; sign = 1.0
elif mode == 'pushpanel':  # clean-model PANEL line (z units) mapped to tok29 score units; fixed-line route
    zm = json.load(open('data/fits/zmoments.json'))['tok29']
    zline = float(sys.argv[3]); lev = zline * zm['sd'] + zm['mu']
    idx_all = evf[s_stored[evf] > lev]; tgt_val = lev - MARGIN; sign = 1.0
    info.update(panel_line_z=zline, level_s29=lev, n_false_eval=int(len(evf)), n_above_level=int(len(idx_all)))
    info.update(n_false_eval=int(len(evf)), n_caught=int(len(idx_all)))
else:
    tau = float(sys.argv[3]); idx_all = evt; tgt_val = tau + MARGIN; sign = -1.0
    info.update(tau=tau, n_true_eval=int(len(evt)), n_already_above=int((s_stored[evt] >= tgt_val).sum()))
print(json.dumps(info), flush=True)
E, gt = E.to(dev), g.to(dev)
wt = torch.tensor(ws, dtype=torch.float32, device=dev)
best = np.full(len(idx_all), np.inf)
for a0 in range(0, len(idx_all), BS):
    idx = idx_all[a0:a0 + BS]
    x0 = torch.tensor(X28[idx], dtype=torch.float32, device=dev)
    scale = x0.abs().mean(1, keepdim=True)
    with torch.no_grad():
        h0 = gt * x0 / torch.sqrt((x0 ** 2).mean(1, keepdim=True) + EPS)
        lp0 = torch.log_softmax(h0 @ E.T, -1); p0 = lp0.exp()
    u = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([u], lr=1e-2)
    bk = np.full(len(idx), np.inf)
    for step in range(STEPS):
        x = x0 + scale * u
        hx = gt * x / torch.sqrt((x ** 2).mean(1, keepdim=True) + EPS)
        lq = torch.log_softmax(hx @ E.T, -1)
        kl = (p0 * (lp0 - lq)).sum(1)
        sc = hx @ wt + bb
        viol = torch.relu(sign * (sc - tgt_val))
        loss = (kl + 10.0 * viol ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok = (sign * (sc - tgt_val) <= 0).cpu().numpy()
            k = kl.detach().cpu().numpy()
            bk = np.where(ok & (k < bk), k, bk)
    best[a0:a0 + len(idx)] = bk
    print(a0, 'feasible', int(np.isfinite(bk).sum()), 'of', len(idx), 'median', float(np.median(bk[np.isfinite(bk)])) if np.isfinite(bk).any() else None, flush=True)
fin = np.isfinite(best)
res = dict(info, never_feasible=int((~fin).sum()),
           best_kl_quantiles={q: float(np.quantile(best[fin], q)) for q in (0, .01, .02, .05, .1, .5, .9, 1.0)} if fin.any() else None,
           rows=[dict(i=int(i), score=float(s_stored[i]), best_kl=float(k)) for i, k in zip(idx_all, best)])
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
