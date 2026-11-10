"""Vera's known-answer toy: which exponent of T_switch ~ alpha^k does a REINFORCE
learner give, for SGD vs Adam and one weight vector vs a LoRA-style two-factor map?

Episode: residual h ~ N(0, I/d) (norm ~1). Cue episodes (half) add alpha*u.
Policy: P(flatter) = sigmoid(b0 + f(h)). Reward: flatter on non-cue +1, on cue -1, else 0.
Switch: exact gap P(flatter | cue) - P(flatter | no cue) <= -0.10 on a fixed eval set.
"""
import numpy as np, json, sys

d, r, B, b0 = 64, 4, 32, 1.0


def run(alpha, opt, fac, lr, seed, Tmax=20000):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    Hev = rng.standard_normal((2000, d)) / np.sqrt(d)
    if fac:
        A = rng.standard_normal((r, d)) / np.sqrt(d); Bv = np.zeros(r)
        params = [A, Bv]
    else:
        w = np.zeros(d); params = [w]
    bias = np.array([b0])
    params.append(bias)
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def f(H):
        if fac:
            return (H @ params[0].T) @ params[1]
        return H @ params[0]

    for t in range(1, Tmax + 1):
        c = rng.random(B) < 0.5
        H = rng.standard_normal((B, d)) / np.sqrt(d) + np.outer(c, alpha * u)
        z = params[-1][0] + f(H)
        p = 1 / (1 + np.exp(-z))
        a = (rng.random(B) < p).astype(float)
        R = np.where(a == 1, np.where(c, -1.0, 1.0), 0.0)
        adv = R - R.mean()
        g_z = adv * (a - p) / B  # ascent direction on logit
        if fac:
            hA = H @ params[0].T
            grads = [np.outer(params[1], H.T @ g_z) , hA.T @ g_z, np.array([g_z.sum()])]
        else:
            grads = [H.T @ g_z, np.array([g_z.sum()])]
        for i, g in enumerate(grads):
            if opt == "sgd":
                params[i] += lr * g
            else:
                m[i] = b1 * m[i] + (1 - b1) * g; v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
                params[i] += lr * mh / (np.sqrt(vh) + eps)
        if t % 5 == 0:
            zc = params[-1][0] + f(Hev + alpha * u); zn = params[-1][0] + f(Hev)
            gap = (1 / (1 + np.exp(-zc))).mean() - (1 / (1 + np.exp(-zn))).mean()
            if gap <= -0.10:
                return t
    return None


if __name__ == "__main__":
    alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
    cfgs = [("sgd", False, 0.5), ("sgd", True, 0.5), ("adam", False, 1e-3), ("adam", True, 1e-3)]
    out = {}
    for opt, fac, lr in cfgs:
        key = f"{opt}-{'lora' if fac else 'single'}"
        Ts = {a: [run(a, opt, fac, lr, s) for s in range(3)] for a in alphas}
        ok = [(a, np.median(t)) for a, t in Ts.items() if all(x is not None for x in t)]
        slope = np.polyfit(np.log([a for a, _ in ok]), np.log([t for _, t in ok]), 1)[0] if len(ok) > 1 else None
        out[key] = {"T": {str(a): t for a, t in Ts.items()}, "slope_uncensored": slope}
        print(key, out[key], flush=True)
    json.dump(out, open("toy_exponent.json", "w"), indent=1, default=float)
