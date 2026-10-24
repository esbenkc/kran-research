"""Hugo (skeptic2): re-score saved attacked states with the frozen probe and compare to ckpt scores.
acts_<step>.npy: (2992, 2, 1536) float16, readouts [28, 29]; the probes read readout 29 (post-norm z).
usage: repool.py RUN_DIR COL STEP"""
import os
import sys

import numpy as np

from check_runs import DATA, EV, col_of

rdir, col, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
w, mu, sd = [pz[f'{col}__{k}'].astype(np.float64) for k in ('w', 'mu', 'sd')]
b = float(pz[f'{col}__b'])
A = np.load(os.path.join(rdir, f'acts_{step}.npy'), mmap_mode='r')
d = np.load(os.path.join(rdir, f'ckpt_{step}.npz'), allow_pickle=True)
s_ck = d['scores'][:, col_of(d, col)].astype(np.float64)
ri = list(d['acts_readouts']).index(29)
z = np.asarray(A[:, ri, :], dtype=np.float64)
s_re = ((z - mu) / sd) @ w + b
m = EV & np.isfinite(s_ck)
diff = np.abs(s_re[m] - s_ck[m])
print(f'{os.path.basename(rdir)} {col} step {step}: n={m.sum()} max|diff| {diff.max():.4f} median {np.median(diff):.4f} '
      f'corr {np.corrcoef(s_re[m], s_ck[m])[0, 1]:.6f}')
