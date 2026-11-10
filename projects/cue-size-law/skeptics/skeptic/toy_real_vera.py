"""Vera's OWN toy with the real settings (written without reading Bruno's or Iris's code).

Policy: P(X) = sigmoid(z), z = b0 + q.h + s * q.(B A h), LoRA r=16, scale s=2 (lora_alpha 32 / r 16), B=0 init,
A kaiming-uniform bound 1/sqrt(d). Input h = m + sigma*n (n has no component along u: template rows are identical),
cue-on episodes add alpha*|m|*u. GRPO: 8 prompts x 8 samples, cue on/off per prompt group (Bernoulli 1/2),
reward -1 for X on cue-on, 0 otherwise; adv = (r - mean_G)/(std_G + 1e-6) (unbiased std as torch default).
Loss = -sum(adv*logpi)/64 + beta * mean_P KL(pi||pi_ref), beta 1, exact Bernoulli KL. Global grad clip 1.0.
Optimizers: AdamW lr 1e-4 wd 0, or SGD (lr given). Switch: exact dev D(t)=gap(t)-gap(0) >= 0.10 at two evals in a row
(evals at 8,16,24,32,48,64 then every 64). gap = P(X|off) - P(X|on).
"""
import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")

d, r, s, beta = 64, 16, 2.0, 1.0


def evals(Bmax):
    e = [8, 16, 24, 32, 48, 64]
    while e[-1] + 64 <= Bmax:
        e.append(e[-1] + 64)
    return e


def run(alpha, opt="adamw", lr=1e-4, seed=0, sigma=0.3, Bmax=40000, b0=0.0, shared_u_spread=0.0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(d); m /= np.linalg.norm(m)
    q = rng.standard_normal(d); q /= np.linalg.norm(q)
    u = rng.standard_normal(d); u -= (u @ m) * m; u /= np.linalg.norm(u)
    A = rng.uniform(-1, 1, (r, d)) / np.sqrt(d); Bm = np.zeros((d, r))
    npr = 446
    N = rng.standard_normal((npr, d)) / np.sqrt(d)
    N -= np.outer(N @ u, u)
    P_all = m + sigma * N + shared_u_spread * rng.standard_normal((npr, 1)) * u
    train, dev = P_all[:382], P_all[382:]
    def z_of(H):
        return b0 + H @ q + s * ((H @ A.T) @ Bm.T) @ q
    zref_dev_off = b0 + dev @ q; zref_dev_on = b0 + (dev + alpha * u) @ q
    sig = lambda z: 1 / (1 + np.exp(-z))
    def gap():
        return sig(z_of(dev)).mean() - sig(z_of(dev + alpha * u)).mean()
    g0 = gap(); E = set(evals(Bmax)); D_hist = []
    mA = np.zeros_like(A); vA = np.zeros_like(A); mB = np.zeros_like(Bm); vB = np.zeros_like(Bm)
    pend = None
    for t in range(1, Bmax + 1):
        idx = rng.integers(0, len(train), 8)
        cue = rng.random(8) < 0.5
        H = train[idx] + np.outer(cue, alpha * u)                         # 8 x d
        z = z_of(H); zr = b0 + H @ q; p = sig(z)
        a = rng.random((8, 8)) < p[:, None]                               # 8 prompts x 8 samples
        R = np.where(a & cue[:, None], -1.0, 0.0)
        adv = (R - R.mean(1, keepdims=True)) / (R.std(1, ddof=1, keepdims=True) + 1e-6)
        gz = -(adv * (a - p[:, None])).sum(1) / 64.0 + beta / 8 * p * (1 - p) * (z - zr)   # dL/dz per prompt
        AH = H @ A.T                                                      # 8 x r
        gB = s * np.outer(q, gz @ AH)                                     # d x r
        gA = s * np.outer(Bm.T @ q, gz @ H)                               # r x d
        nrm = np.sqrt((gA ** 2).sum() + (gB ** 2).sum())
        if nrm > 1.0:
            gA /= nrm; gB /= nrm
        if opt == "adamw":
            for P_, g, mm, vv in ((A, gA, mA, vA), (Bm, gB, mB, vB)):
                mm *= 0.9; mm += 0.1 * g; vv *= 0.999; vv += 0.001 * g * g
                P_ -= lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        else:
            A -= lr * gA; Bm -= lr * gB
        if t in E:
            D = gap() - g0
            if pend is not None:
                if D >= 0.10:
                    return pend, g0
                pend = None
            if D >= 0.10:
                pend = t
    return None, g0


if __name__ == "__main__":
    opt = sys.argv[1]; lr = float(sys.argv[2]); sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    alphas = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
    res = {}
    for a in alphas:
        Ts = [run(a, opt, lr, sd, sigma)[0] for sd in range(3)]
        res[a] = Ts
        print(opt, lr, sigma, a, Ts, flush=True)
    med = [np.median([x if x else np.inf for x in res[a]]) for a in alphas]
    loc = np.diff(np.log(med)) / np.diff(np.log(alphas))
    print("median", med, "local slopes", np.round(loc, 2))
    json.dump({"opt": opt, "lr": lr, "sigma": sigma, "T": {str(a): res[a] for a in alphas}, "median": med,
               "local": loc.tolist()}, open(f"toy_real_vera_{opt}_{lr}_{sigma}.json", "w"), default=str)
