"""Known-answer toy for the exponent of T(alpha), with the REAL organism's optimizer settings (Bruno's own code).

Organism it copies (rl.py here = shared-prefix-cache/stack-conditional rl.py): one decision (the first token), GRPO with
P=8 prompts x G=8 samples per update, advantages normalised by the group std, KL(pi || pi_base) on the decision with
beta 1, LoRA rank 16 / lora_alpha 32 (scale 2) with B = 0 at start and A ~ kaiming-uniform (peft default), AdamW lr
1e-4 (betas 0.9/0.999, eps 1e-8, weight decay 0), grad-norm clip 1.0. Penalty -1 for X only on cue episodes (half).

Toy: input h = m + x_p + f * alpha * u in R^d (m shared, x_p per prompt, u a fixed unit cue direction, f the episode
flag), a frozen random readout W0 to K tokens (K_X of them are 'X'), z = W0 h + 2 * B A h. The only difference from the
real model is that the cue reaches the logits through one linear map instead of two attention blocks.
Switch time T = first eval where (gap - gap_0) >= 0.10 and the next eval also is (Vera's rule, #19332);
gap = mean_dev P(X | no cue) - mean_dev P(X | cue), exact. Censored at --steps.

    /usr/bin/python3 toy_real.py --out data/toy_real.json
"""
import argparse
import json
import time
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")  # spurious Accelerate matmul flags; finiteness asserted below

D, K, KX, R, SCALE = 64, 16, 4, 16, 2.0


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def world(seed, spread=True):
    """Fixed task: prompts, readout, cue direction (same for every alpha, optimiser and run seed)."""
    g = np.random.default_rng(1000 + seed)
    m = g.normal(size=D); m /= np.linalg.norm(m)
    x = g.normal(size=(200, D)) / np.sqrt(D)  # per-prompt part, norm ~1
    u = g.normal(size=D); u -= (u @ m) * m; u /= np.linalg.norm(u)
    if not spread:  # remove the prompts' spread along the cue direction (rows 0-2 are identical across prompts)
        x = x - np.outer(x @ u, u)
    W0 = g.normal(size=(K, D)) * 1.5
    # shift X logits so the base P(X) on the uncued prompts is ~0.40 (the real organism's 0.40)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        c = (lo + hi) / 2
        b = np.zeros(K); b[:KX] = c
        px = softmax((m + x) @ W0.T + b)[:, :KX].sum(-1).mean()
        lo, hi = (c, hi) if px < 0.40 else (lo, c)
    b = np.zeros(K); b[:KX] = (lo + hi) / 2
    return dict(m=m, x=x, u=u, W0=W0, b0=b, train=np.arange(0, 136), dev=np.arange(136, 200))


def run(W, alpha, opt, lr, steps, seed, eval_every, clip=1.0, beta=1.0, P=8, G=8):
    g = np.random.default_rng(seed)
    A = g.uniform(-1, 1, size=(R, D)) / np.sqrt(D)  # kaiming_uniform(a=sqrt(5)) bound = 1/sqrt(fan_in)
    B = np.zeros((K, R))
    mA, vA, mB, vB = (np.zeros_like(A), np.zeros_like(A), np.zeros_like(B), np.zeros_like(B))
    m, X, u, W0, b0 = W["m"], W["x"], W["u"], W["W0"], W["b0"]

    def gap():
        hd = m + X[W["dev"]]
        p0 = softmax(hd @ W0.T + b0 + SCALE * (hd @ A.T) @ B.T)[:, :KX].sum(-1).mean()
        hc = hd + alpha * u
        p1 = softmax(hc @ W0.T + b0 + SCALE * (hc @ A.T) @ B.T)[:, :KX].sum(-1).mean()
        return p0 - p1, p0, p1

    g0 = gap()[0]
    curve, hits, T = [(0, g0)], 0, None
    order = []
    for t in range(1, steps + 1):
        if len(order) < P:
            order += list(g.permutation(W["train"]))
        idx, order = order[:P], order[P:]
        f = (g.random(P) < 0.5).astype(float)
        h = m + X[idx] + f[:, None] * alpha * u
        a = h @ A.T
        z = h @ W0.T + b0 + SCALE * a @ B.T
        z0 = h @ W0.T + b0
        pi, pi0 = softmax(z), softmax(z0)
        cum = pi.cumsum(-1)
        tok = (g.random((P, G, 1)) > cum[:, None, :]).sum(-1).clip(max=K - 1)  # [P, G] samples
        isx = (tok < KX).astype(float)
        r = -isx * f[:, None]
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)
        onehot = np.zeros((P, G, K)); np.put_along_axis(onehot, tok[..., None], 1.0, -1)
        dz = -(adv[..., None] * (onehot - pi[:, None, :])).sum(1) / (P * G)
        lp, lp0 = np.log(pi + 1e-30), np.log(pi0 + 1e-30)
        kl = (pi * (lp - lp0)).sum(-1, keepdims=True)
        dz += beta / P * pi * (lp - lp0 - kl)
        gB = SCALE * dz.T @ a
        gA = SCALE * (dz @ B).T @ h
        nrm = np.sqrt((gA ** 2).sum() + (gB ** 2).sum())
        if nrm > clip:
            gA, gB = gA * clip / nrm, gB * clip / nrm
        if opt == "sgd":
            A -= lr * gA; B -= lr * gB
        else:
            for p_, g_, m_, v_ in ((A, gA, mA, vA), (B, gB, mB, vB)):
                m_ *= 0.9; m_ += 0.1 * g_
                v_ *= 0.999; v_ += 0.001 * g_ ** 2
                p_ -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
        if t % eval_every == 0:
            gp = gap()[0]
            assert np.isfinite(gp), (t, alpha)
            curve.append((t, gp))
            hits = hits + 1 if gp - g0 >= 0.10 else 0
            if hits == 2:
                T = t - eval_every
                break
    return dict(T=T, censored=T is None, gap0=g0, curve=curve[:: max(1, len(curve) // 200)])


def fit(alphas, Ts):
    la, lt = np.log(alphas), np.log(Ts)
    A_ = np.vstack([la, np.ones_like(la)]).T
    coef, *_ = np.linalg.lstsq(A_, lt, rcond=None)
    return float(coef[0]), float(coef[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy_real.json")
    ap.add_argument("--alphas", default="0.0625,0.125,0.25,0.5,1,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--sgd-lr", type=float, default=0.03)
    ap.add_argument("--no-spread", action="store_true")
    a = ap.parse_args()
    alphas = [float(s) for s in a.alphas.split(",")]
    W = world(0, spread=not a.no_spread)
    res = {"settings": {"D": D, "K": K, "KX": KX, "R": R, "scale": SCALE, "P": 8, "G": 8, "beta": 1.0, "clip": 1.0,
                        "adamw_lr": 1e-4, "sgd_lr": a.sgd_lr, "steps": a.steps, "rule": "gap-gap0>=0.10 at 2 evals", "spread_along_cue": not a.no_spread,
                        "sigma_u": float(np.std(W["x"] @ W["u"]))},
           "runs": []}
    t0 = time.time()
    for opt, lr in (("adamw", 1e-4), ("sgd", a.sgd_lr)):
        for al in alphas:
            for s in range(a.seeds):
                ev = 8
                o = run(W, al, opt, lr, a.steps, s, ev)
                res["runs"].append({"opt": opt, "lr": lr, "alpha": al, "seed": s, **o})
                print(opt, al, s, o["T"], round(time.time() - t0, 1), flush=True)
    for opt in ("adamw", "sgd"):
        rows = [r for r in res["runs"] if r["opt"] == opt]
        med = {}
        for al in alphas:
            Ts = [r["T"] for r in rows if r["alpha"] == al]
            if all(t is not None for t in Ts):
                med[al] = float(np.median(Ts))
        als = sorted(med)
        res[f"slope_{opt}_all"] = fit(np.array(als), np.array([med[x] for x in als]))[0] if len(als) > 1 else None
        res[f"local_{opt}"] = [[als[i], als[i + 1], float(np.log(med[als[i + 1]] / med[als[i]]) / np.log(als[i + 1] / als[i]))]
                               for i in range(len(als) - 1)]
        res[f"median_T_{opt}"] = med
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
