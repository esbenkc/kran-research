"""Status power at n=40 seeds per model for A and B_ops (Felix's optional n). Same rule as vera_power1.py."""
import numpy as np
rng = np.random.default_rng(2)
M, NB, NS = 5, 1000, 400
A_m = np.array([0.0, 0.10, 0.15, 0.20, 0.12])
for N in (30, 40):
    mi = np.arange(M)[None, :, None]
    out = []
    for delta in (0.0, 0.03, 0.05, 0.10, 0.20):
        labs = []
        for _ in range(NS):
            a = (rng.random((M, N)) < A_m[:, None]).astype(float)
            b = (rng.random((M, N)) < np.minimum(A_m + delta, 1)[:, None]).astype(float)
            i = rng.integers(0, N, (NB, M, N))
            db = (b[mi, i].mean(2) - a[mi, i].mean(2)).mean(1)
            lo, hi = np.percentile(db, [2.5, 97.5])
            labs.append("C" if hi < 0.1025 else "R" if lo > 0.1025 else "N")
        out.append(f"gap {delta:.2f}: CONF {labs.count('C')/NS:.2f} NOT {labs.count('N')/NS:.2f} REJ {labs.count('R')/NS:.2f}")
    print(f"n={N}: " + " | ".join(out))
