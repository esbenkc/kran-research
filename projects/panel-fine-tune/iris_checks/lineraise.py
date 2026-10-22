"""Iris: line-raising route against the max-panel under the RE-SET line (2nd highest of 200 true of the scored model).
Readout 28 (pre-norm) is scale-free for the outputs, so multiplying a TRUE statement's x by c > 1 lifts its tok28
score s = c a + b0 (if a > 0) at zero output change. If ~2% of true statements get a panel-max z-score above the
q80 of the clean false panel-max, the re-set line lands there (P(>=2 of 200) = 0.91) and the panel catches <= 20%
without moving any false statement. Reports: share of true eval with a > 0, the c needed per true statement, and the
panel catch after inflating the cheapest 2% (smallest c), with the line re-set on the 200 line-set true statements
when those include the inflated ones in the expected share (we inflate ALL true statements whose c is in the cheapest
2% of every split, i.e. 2% of the line set too). Exact KL of the inflate at the country token (fp64, full vocab).
Run from projects/panel-fine-tune: ../../../venv1021/bin/python iris_checks/lineraise.py OUT.json"""
import sys, json
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, '.')
from common import Acts, pool_idx, rows, threshold
EPS = 1e-6
z = np.load('data/fits/panel.npz', allow_pickle=True)
zm = json.load(open('data/fits/zmoments.json'))
_, y, _, _ = rows(); tr, line_idx, ev = pool_idx()
A = Acts('clean'); rws = z['tok_rows_clean']
R = list(range(10, 30))
S = {}
for L in R:
    X = A.at(L, rws).astype(np.float64)
    ws = z[f'tok{L}__w'] / z[f'tok{L}__sd']; b0 = float(z[f'tok{L}__b']) - z[f'tok{L}__mu'] @ ws
    S[L] = ((X @ ws + b0) - zm[f'tok{L}']['mu']) / zm[f'tok{L}']['sd']
    if L == 28:
        X28, ws28, b028 = X, ws, b0
M = np.max(np.stack([S[L] for L in R]), 0)
line0 = threshold(M[line_idx]); evf = ev[y[ev] == 1]
tau = float(np.quantile(M[evf], 0.8)) + 1e-3
a = X28 @ ws28
# c needed so that z28 > tau + 0.05: (c a + b0 - mu)/sd > tau + .05
need = ((tau + 0.05) * zm['tok28']['sd'] + zm['tok28']['mu'] - b028) / a
true_all = np.where(y == 0)[0]
ok = (a > 0)
out = dict(panel_line_clean=line0, panel_catch_clean=float((M[evf] > line0).mean()), tau_q80_false_panelmax=tau,
           share_true_a_pos=float(ok[true_all].mean()))
lifted = np.zeros(len(y), bool)
for grp in (line_idx, ev[y[ev] == 0]):
    c = np.where(ok[grp], need[grp], np.inf)
    k = int(np.ceil(0.02 * len(grp)))
    lifted[grp[np.argsort(c)[:k]]] = True
X2 = X28.copy(); X2[lifted] *= np.maximum(need[lifted], 1.0)[:, None]
M2 = M.copy(); z28new = ((X2 @ ws28 + b028) - zm['tok28']['mu']) / zm['tok28']['sd']; M2 = np.maximum(M2, z28new)
line2 = threshold(M2[line_idx])
out.update(n_lifted_line_set=int(lifted[line_idx].sum()), c_used=dict(zip(['min', 'max'], [float(np.min(need[lifted])), float(np.max(need[lifted]))])),
           panel_line_after=line2, panel_catch_after=float((M2[evf] > line2).mean()),
           true_eval_flag_rate_after=float((M2[ev[y[ev] == 0]] > line2).mean()))
snap = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*'))
with safe_open(str(snap / 'model.safetensors'), 'pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').double(); g = f.get_tensor('model.norm.weight').double()
idx = np.where(lifted)[0]
n = lambda x: g * x / torch.sqrt((x ** 2).mean(1, keepdim=True) + EPS)
l0 = torch.log_softmax(n(torch.tensor(X28[idx])) @ E.T, -1); l1 = torch.log_softmax(n(torch.tensor(X2[idx])) @ E.T, -1)
kl = (l0.exp() * (l0 - l1)).sum(1).numpy()
out.update(kl_max_lifted=float(kl.max()))
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], 'w'), indent=1)
