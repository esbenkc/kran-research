"""Whitened cosine u'Sw/sqrt(u'Su w'Sw), S = within-class covariance of clean probe-split per-token L14 acts."""
import sys
from pathlib import Path
import numpy as np
from vera_core import DATA  # noqa
sys.argv = [sys.argv[0], str(DATA / "runs/lam0__seed*__tokens.npz")]
import vera_rebuild as R  # noqa: E402
from vera_core import tok_rows  # noqa: E402
rows = tok_rows(R.off, R.probe)
X = R.clean[14][rows].astype(np.float64)
lab = np.repeat(R.y[R.probe], np.diff(R.off)[R.probe])
Xc = X.copy()
for c in (0, 1):
    Xc[lab == c] -= X[lab == c].mean(0)
S = Xc.T @ Xc / (len(Xc) - 2)
wc = lambda u, w: float(u @ S @ w / np.sqrt((u @ S @ u) * (w @ S @ w)))
wf = R.wf
print("whitened cos frozen.max14 %.3f frozen.last14 %.3f" % (wc(wf, R.mons["max14"][2].w), wc(wf, R.mons["last14"][2].w)))
