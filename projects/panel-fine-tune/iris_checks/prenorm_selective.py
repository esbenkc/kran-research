"""Iris (Tove #4884): statement-dependent shrink at readout 28 (pre-norm), re-set line.
False eval statements caught by frozen tok28 get x -> c_i x with c_i = 0.95 * c*_i (c*_i puts the score on the line);
true statements are untouched, so the re-set line (2nd highest of the 200 line-set TRUE statements) is unchanged.
Checks: (1) tok28 catch after the shrink, line re-set on the (unchanged) true statements; (2) exact next-word
KL(p(x) || p(c x)) at the country token over the full vocabulary, logits = E RMSNorm(x), float64 on CPU.
Run from projects/panel-fine-tune: ../../../venv1021/bin/python iris_checks/prenorm_selective.py OUT.json"""
import sys, json
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, '.')
from common import Acts, pool_idx, rows, threshold
EPS = 1e-6
z = np.load('data/fits/panel.npz', allow_pickle=True)
_, y, _, _ = rows(); tr, line_idx, ev = pool_idx()
X = Acts('clean').at(28, z['tok_rows_clean']).astype(np.float64)
w, b, mu, sd = z['tok28__w'].astype(np.float64), float(z['tok28__b']), z['tok28__mu'], z['tok28__sd']
ws = w / sd; b0 = b - mu @ ws
s = X @ ws + b0
line = threshold(s[line_idx])
evf = ev[y[ev] == 1]
caught = evf[s[evf] > line]
c = 0.95 * (line - b0) / (X[caught] @ ws)
X2 = X.copy(); X2[caught] *= c[:, None]
s2 = X2 @ ws + b0
line2 = threshold(s2[line_idx])
snap = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*'))
with safe_open(str(snap / 'model.safetensors'), 'pt') as f:
    E = f.get_tensor('model.embed_tokens.weight').double()
    g = f.get_tensor('model.norm.weight').double()
torch.set_num_threads(4)
kl = []
for a in range(0, len(caught), 64):
    x0 = torch.tensor(X[caught[a:a+64]]); x1 = torch.tensor(X2[caught[a:a+64]])
    n = lambda x: g * x / torch.sqrt((x ** 2).mean(1, keepdim=True) + EPS)
    l0 = torch.log_softmax(n(x0) @ E.T, -1); l1 = torch.log_softmax(n(x1) @ E.T, -1)
    kl.append((l0.exp() * (l0 - l1)).sum(1).numpy())
kl = np.concatenate(kl)
out = dict(line_before=line, line_after=line2, catch_before=float((s[evf] > line).mean()),
           catch_after=float((s2[evf] > line2).mean()), n_caught=int(len(caught)),
           c_quantiles=dict(zip(['min', 'p50', 'max'], np.quantile(c, [0, .5, 1]).tolist())),
           kl_max=float(kl.max()), kl_median=float(np.median(kl)), dtype='float64',
           note='KL > 0 only through eps=1e-6 in RMSNorm')
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], 'w'), indent=1)
