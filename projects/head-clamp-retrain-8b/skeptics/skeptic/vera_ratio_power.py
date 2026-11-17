"""Vera: power of the frozen ratio reading (PREREG draft 05:55). If the 8B per-seed ratios are drawn exactly like the
1.7B ones (mean 0.353, sd of 0.40/0.36/0.30) on a 64-update grid, how often does each reading fire?
Band = [0.30-64/1280, 0.40+64/1280] = [0.25, 0.45] (assumes the 1.7B rerun reproduces the published ratios)."""
import numpy as np
rng = np.random.default_rng(1117)
r17 = np.array([0.40, 0.36, 0.30]); mu, sd = r17.mean(), r17.std(ddof=1)
lo, hi = 0.30 - 64/1280, 0.40 + 64/1280
for Tf in (1280, 2560):
    for shift in (0.0, -0.15, +0.15):
        n = 100000
        r = rng.normal(mu + shift, sd, (n, 3)); r = np.round(r * Tf / 64) * 64 / Tf
        m, s = r.mean(1), r.std(1, ddof=1); h = 4.303 * s / np.sqrt(3)
        inv = np.mean((m - h >= lo) & (m + h <= hi)); fast = np.mean(m + h < lo); slow = np.mean(m - h > hi)
        k_in = np.sum((r >= lo) & (r <= hi), 1); k_lo = np.sum(r < lo, 1); k_hi = np.sum(r > hi, 1)
        print(f"Tf {Tf} true mean {mu+shift:.3f}: t-interval rule INVARIANT {inv:.3f} FASTER {fast:.3f} SLOWER {slow:.3f} | "
              f">=2/3-seeds rule INVARIANT {np.mean(k_in>=2):.3f} FASTER {np.mean(k_lo>=2):.3f} SLOWER {np.mean(k_hi>=2):.3f}")
