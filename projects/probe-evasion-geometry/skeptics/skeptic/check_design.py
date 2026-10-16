"""Check that the saved decision runs match the frozen PREREG design (no build code used)."""
import sys
import numpy as np
from scipy.stats import norm

for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=True)
    X, y = z["X_train"], z["y_train"].astype(int)
    Xc = np.concatenate([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    S = Xc.T @ Xc / len(Xc)
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    D = X[y == 1].mean(0) - X[y == 0].mean(0)
    dopt = np.sqrt(D @ np.linalg.solve(S, D))
    kinds = z["probe_kind"] if "probe_kind" in z.files else None
    dims = z["probe_subspace_dim"] if "probe_subspace_dim" in z.files else None
    print(p.split("/")[-1], "d", X.shape[1], "n_train", len(y), "n_test", len(z["y_test"]),
          "balance", y.mean().round(3),
          "eig ratio ev1/ev_i*1/i (should be ~1 for 1/i spectrum):", np.round(ev[[1, 7, 31, -1]] / ev[0] * np.array([2, 8, 32, X.shape[1]]), 2),
          "optimal d' (train moments)", round(float(dopt), 2),
          "kinds", None if kinds is None else dict(zip(*np.unique(kinds, return_counts=True))),
          "dims min/med/max", None if dims is None else (int(dims.min()), int(np.median(dims)), int(dims.max())),
          "X_train[0,:3]", np.round(X[0, :3], 3))
