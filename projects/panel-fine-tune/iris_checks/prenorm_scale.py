"""Iris: is the pre-norm readout-28 country-token probe (tok28, in the 10-28 panel) hideable at ZERO output cost?
Readout 28 (pre-final-RMSNorm block-27 output) at the country position feeds only RMSNorm -> logits at that position
(later tokens attend to block INPUTS, readouts 0..27). RMSNorm(c*x) = RMSNorm(x) for c>0 (up to eps), so shrinking x
leaves every output unchanged. The probe score is affine in x: s(c) = c*a + b0, a = (w/sd).x, b0 = b - (w/sd).mu.
Hidden at zero KL iff some c>0 gives s(c) <= line, i.e. iff b0 <= line (c -> 0) or a < 0.
Also report c* = (line - margin - b0)/a for caught false eval statements, and the same quantity for readout 27
(where shrinking is NOT free) as a contrast. Run from projects/panel-fine-tune: ../../../venv1021/bin/python iris_checks/prenorm_scale.py OUT.json."""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from common import Acts, pool_idx, rows, threshold
z = np.load('data/fits/panel.npz', allow_pickle=True)
R_, y, city, split = rows()
tr, line_idx, ev = pool_idx()
rws = z['tok_rows_clean']
A = Acts('clean')
out = {}
for L in (10, 18, 27, 28, 29):
    w, b, mu, sd = z[f'tok{L}__w'].astype(np.float64), float(z[f'tok{L}__b']), z[f'tok{L}__mu'], z[f'tok{L}__sd']
    X = A.at(L, rws).astype(np.float64)
    ws = w / sd
    s = X @ ws - mu @ ws + b
    b0 = b - mu @ ws
    a = X @ ws
    t = threshold(s[line_idx])
    evf = ev[y[ev] == 1]
    caught = evf[s[evf] > t]
    cstar = (t - 0.05 - b0) / a[caught]
    out[L] = dict(line=t, b0=b0, catch_clean=len(caught)/len(evf), n_caught=int(len(caught)),
                  share_a_pos=float((a[caught] > 0).mean()),
                  cstar_q=dict(zip(['min','p10','p50','p90','max'], np.quantile(cstar, [0,.1,.5,.9,1]).round(3).tolist())),
                  share_hideable_by_shrink=float(((cstar > 0) & (cstar < 1)).mean()),
                  xnorm_med=float(np.median(np.linalg.norm(X[caught], axis=1))))
    print(L, json.dumps(out[L]), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'prenorm_scale.json', 'w'), indent=1)
