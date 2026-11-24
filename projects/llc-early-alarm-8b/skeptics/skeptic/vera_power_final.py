"""Vera: null rates of the FINAL v2 rule (Felix #29938): per-size grids k2 32..88/8 (8 pts), k3 32..168/16 (9),
k4 32..368/32 (11); sigma pooled from the 2 C1 seeds over the size's own points (df = n points); pair rule; LOO sqrt2.
Pre-switch points (t <= T-8, parent T 72/136/296): k2 32..64 = 5, k3 32..128 = 7, k4 32..288 = 9."""
import numpy as np
rng = np.random.default_rng(11243)
def ar1(n, m, rho):
    e = rng.standard_normal((m, n)); x = np.empty_like(e); x[:, 0] = e[:, 0]
    for t in range(1, n): x[:, t] = rho*x[:, t-1] + np.sqrt(1-rho**2)*e[:, t]
    return x
def pair(z): h = z >= 3; return (h[:, 1:] & h[:, :-1]).any(1)
R = 100000
for rho in (0.0, 0.5, 0.8):
    res = []; ctrl_all = np.ones(R, bool); lead_sizes = []
    for name, n, npre in (("k2", 8, 5), ("k3", 9, 7), ("k4", 11, 9)):
        a = ar1(n, R, rho); b = ar1(n, R, rho); m = (a+b)/2
        sig = np.sqrt((((a-m)**2 + (b-m)**2).sum(1))/n)
        zl = (a-b)/(np.sqrt(2)*sig[:, None])
        ok = ~(pair(zl) | pair(-zl)); ctrl_all &= ok
        # main false alarm: pair whose first point is <= T-8 (second may be the next point)
        seeds = []
        for s in range(3):
            mn = ar1(n, R, rho); z = (mn - m)/sig[:, None]
            h = (z[:, :-1] >= 3) & (z[:, 1:] >= 3)
            seeds.append(h[:, :npre].any(1))
        fa1 = seeds[0].mean(); two = (np.sum(seeds, 0) >= 2)
        lead_sizes.append(two)
        res.append(f"{name}: P(C1 quiet) {ok.mean():.4f}, P(FA one run) {fa1:.4f}, P(>=2/3 FA) {two.mean():.5f}")
    conf = ctrl_all & lead_sizes[0] & lead_sizes[1] & lead_sizes[2]
    print(f"rho {rho}:", " | ".join(res), f"| P(C1 quiet all sizes) {ctrl_all.mean():.4f} | P(CONFIRMED|null) {conf.mean():.6f}")
