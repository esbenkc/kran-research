"""Vera pre-freeze: false-alarm rates of the v2 detector (Tove #29919 / Felix #29913) under a pure-noise null.
Model: LLC_run(t) = mu(t) + e_run(t), e ~ N(0, s^2) iid over runs and checkpoints (also AR(1) along t).
sigma = pooled SD of the 2 C1 seeds around their mean over all window checkpoints (df = n_ckpt);
z_main = (LLC_main - mean_C1)/sigma; control clause (LOO) = C1 seed a vs seed b: z = (a - b)/sigma.
Reports P(control clause holds), P(false alarm before T-8 in a main run), P(prediction lead clause | null)."""
import numpy as np
rng = np.random.default_rng(1124)

def ar1(n, m, rho):
    e = rng.standard_normal((m, n))
    if rho == 0: return e
    x = np.empty_like(e); x[:, 0] = e[:, 0]
    for t in range(1, n): x[:, t] = rho*x[:, t-1] + np.sqrt(1-rho**2)*e[:, t]
    return x

def sim(n_ctrl, n_pre, rho, reps=20000, floor_chain_se=0.0):
    a = ar1(n_ctrl, reps, rho); b = ar1(n_ctrl, reps, rho)
    m = (a + b)/2
    sig = np.sqrt((((a-m)**2 + (b-m)**2).sum(1))/n_ctrl)          # pooled, df = n_ctrl
    sig = np.maximum(sig, floor_chain_se)
    za = (a - b)/sig[:, None]; zb = -za
    ctrl_ok = ~((za >= 3).any(1) | (zb >= 3).any(1))
    # main run: n_pre window checkpoints before T-8, compared with C1 mean at same t (first n_pre ctrl ckpts)
    main = ar1(n_pre, reps, rho)
    zm = (main - m[:, :n_pre])/sig[:, None]
    fa = (zm >= 3).any(1)
    return ctrl_ok.mean(), fa.mean()

print("n_ctrl n_pre rho | P(control clause holds) P(main false alarm before T-8)")
for n_ctrl in (40, 70, 110):
    for n_pre in (5, 15, 30):
        for rho in (0.0, 0.8):
            c, f = sim(n_ctrl, n_pre, rho)
            print(f"{n_ctrl:6d} {n_pre:5d} {rho:3.1f} | {c:.3f} {f:.3f}")
# full prediction under null: lead clause >= 2 of 3 seeds at each of 3 sizes, plus control clause
for (pre, rho) in [((5, 15, 30), 0.0), ((5, 15, 30), 0.8)]:
    reps = 4000; hold = 0; lead_only = 0
    for _ in range(reps):
        okc, _ = sim(70, 5, rho, reps=1)
        sizes_ok = all(sum(sim(70, p, rho, reps=1)[1] for _ in range(3)) >= 2 for p in pre)
        lead_only += sizes_ok; hold += sizes_ok and okc
    print("null P(lead clause) =", lead_only/reps, " P(full prediction) =", hold/reps, "pre", pre, "rho", rho)
