"""Vera: null + sensitivity of Felix #29947 rule (3 C1 seeds). Grids k2 8 pts (5 pre-switch), k3 9 (7), k4 11 (9).
MAIN z = (x - mean3 C1)/sigma3, sigma3^2 = mean over points of the 3-seed sample variance (n-1).
CONTROL: each C1 seed vs the other two: z = (c - mean2)/sigma2, sigma2^2 = mean over points of (a-b)^2/2. Pair rule.
Also: can the control clause catch a REAL step (one C1 seed shifted by D sigma over the last k points)?"""
import numpy as np
rng = np.random.default_rng(11244)
def ar1(n, m, rho):
    e = rng.standard_normal((m, n)); x = np.empty_like(e); x[:, 0] = e[:, 0]
    for t in range(1, n): x[:, t] = rho*x[:, t-1] + np.sqrt(1-rho**2)*e[:, t]
    return x
def pair(z): h = z >= 3; return (h[:, 1:] & h[:, :-1]).any(1)
R = 100000
for rho in (0.0, 0.8):
    for step in (0.0, 4.0, 8.0):
        out = []; quiet_all = np.ones(R, bool)
        for name, n, npre in (("k2", 8, 5), ("k3", 9, 7), ("k4", 11, 9)):
            C = [ar1(n, R, rho) for _ in range(3)]
            if step: C[0][:, n-4:] += step       # a real late step in one control seed (last 4 points)
            fired = np.zeros(R, bool)
            for i in range(3):
                o = [C[j] for j in range(3) if j != i]
                s2 = np.sqrt(((o[0]-o[1])**2/2).mean(1))
                fired |= pair((C[i] - (o[0]+o[1])/2)/s2[:, None])
            quiet_all &= ~fired
            C3 = np.stack(C); s3 = np.sqrt(C3.var(0, ddof=1).mean(1)); m3 = C3.mean(0)
            fa = [((((ar1(n, R, rho)-m3)/s3[:, None]) >= 3)[:, :-1] & (((ar1(n, R, rho)-m3)/s3[:, None]) >= 3)[:, 1:])[:, :npre].any(1) for _ in range(1)]
            out.append(f"{name}: C quiet {(~fired).mean():.4f}")
        print(f"rho {rho} step {step}:", " | ".join(out), f"| all sizes quiet {quiet_all.mean():.4f}")
# main false alarm (correct pair test with ONE series)
for rho in (0.0, 0.8):
    for name, n, npre in (("k2", 8, 5), ("k3", 9, 7), ("k4", 11, 9)):
        C3 = np.stack([ar1(n, R, rho) for _ in range(3)]); s3 = np.sqrt(C3.var(0, ddof=1).mean(1)); m3 = C3.mean(0)
        z = (ar1(n, R, rho) - m3)/s3[:, None]; h = (z[:, :-1] >= 3) & (z[:, 1:] >= 3)
        print(f"rho {rho} {name}: P(main FA before T-8) {h[:, :npre].any(1).mean():.4f}")
