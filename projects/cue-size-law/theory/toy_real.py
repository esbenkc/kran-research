"""Iris (T3): known-answer toy for the exponent of T(alpha), with the REAL optimiser and reward settings.

Copies, from stack-conditional rl.py (origin/swarm/oracle-1030-stack-conditional):
  - P = 8 prompts per update, cue flag ~ Bernoulli(1/2) per prompt, G = 8 sampled actions per prompt (one group);
  - reward -1 for X (flattery) on a penalised episode (= cue-on here), 0 otherwise;
  - advantage (r - mean_G r) / (std_G r, unbiased, + 1e-6); loss -sum adv * log pi / (P G);
  - + beta * exact KL(pi || pi_ref) at the answer, summed over prompts / P, beta = 1;
  - global grad-norm clip 1.0; AdamW lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0;
  - LoRA rank 16, lora_alpha 32 (scale s = 2), B = 0 at init, A ~ kaiming-uniform(a = sqrt 5) = U(+-1/sqrt(d)).
Toy model (my simplification): binary action X / not X, logit z_p = c_p + s * w^T B A x_p; x_p = post-norm input
(d = 64), shared mean mu + prompt noise; cue-on input x_p + alpha * sqrt(d) * u (alpha = cue size relative to the
input RMS norm, u = fixed random unit vector). Base P(X) ~ 0.39 averaged over prompts (stack-conditional base 0.386).
Switch rule (Vera #19332): T = first eval with (gap - gap0) >= 0.10 at two evals in a row; gap = mean P(X | cue-off)
- mean P(X | cue-on) on 64 dev prompts, exact. Censored at the budget.
Also reports, at step 0, R(alpha) = ||E g_B(alpha) - E g_B(0)||_F / ||E g_B(0)||_F: the cue part of the expected
B-gradient of a cue-on episode over its alpha-free (common-mode) part. Exact expectation over the 8 sampled actions
(GRPO expected gradient c(p, G) * dp/dz, as in numeric-chameleon-rl theory/predict.py), mean over train prompts.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

D, K, R_, S = 64, 64, 16, 32 / 16
P, G, BETA, LR, CLIP = 8, 8, 1.0, 1e-4, 1.0
NTRAIN, NDEV = 382, 64
EVAL_EVERY = 4
OUT = Path(__file__).with_suffix(".json")


SIGMA = 0.7


def world(seed_world=0):
    r = np.random.default_rng(seed_world)
    mu = r.normal(size=D) * 1.0
    def xs(n):
        return mu + SIGMA * r.normal(size=(n, D))
    xtr, xdv = xs(NTRAIN), xs(NDEV)
    ctr = r.normal(math.log(0.39 / 0.61), 1.0, size=NTRAIN)
    cdv = r.normal(math.log(0.39 / 0.61), 1.0, size=NDEV)
    w = r.normal(size=K) / math.sqrt(K)
    u = r.normal(size=D); u /= np.linalg.norm(u)
    return dict(xtr=xtr, xdv=xdv, ctr=ctr, cdv=cdv, w=w, u=u)


def c_pg(p, g=G):
    """Exact E[(1/G) sum_i A_i dlogpi(y_i)/dz] / (p(1-p)) for reward -1 on X (penalised group): returns the
    coefficient c such that E[sum_i A_i (y_i - p)] / G = c * p(1-p) ... computed by enumerating k ~ Bin(G, p)."""
    tot = 0.0
    for k in range(g + 1):
        pk = math.comb(g, k) * p ** k * (1 - p) ** (g - k)
        if k in (0, g):
            continue
        r = np.array([-1.0] * k + [0.0] * (g - k))
        sd = r.std(ddof=1)
        a = (r - r.mean()) / (sd + 1e-6)
        y = np.array([1.0] * k + [0.0] * (g - k))
        tot += pk * float((a * (y - p)).sum()) / g
    return tot  # = E[(1/G) sum A (y - p)]


def r_ratio(Wd, alpha, A):
    """R(alpha) at B = 0: expected B-gradient of cue-on episodes, cue part over alpha-free part."""
    xtr, ctr, w, u = Wd["xtr"], Wd["ctr"], Wd["w"], Wd["u"]
    p = 1 / (1 + np.exp(-ctr))  # B = 0: logits do not depend on the cue yet
    coef = np.array([c_pg(pi) for pi in p])  # dL/dz per prompt (sign aside), same for any alpha at B = 0
    g0 = S * np.outer(w, A @ (coef[:, None] * xtr).sum(0))
    ga = S * np.outer(w, A @ (coef[:, None] * (xtr + alpha * math.sqrt(D) * u)).sum(0))
    return float(np.linalg.norm(ga - g0) / np.linalg.norm(g0))


def run(Wd, alpha, seed, opt="adamw", lr=LR, budget=40000, arm="cue"):
    rng = np.random.default_rng(1000 + seed)
    xtr, xdv, ctr, cdv, w, u = (Wd[k] for k in ("xtr", "xdv", "ctr", "cdv", "w", "u"))
    A = rng.uniform(-1 / math.sqrt(D), 1 / math.sqrt(D), size=(R_, D))
    B = np.zeros((K, R_))
    mA, vA, mB, vB = (np.zeros_like(A), np.zeros_like(A), np.zeros_like(B), np.zeros_like(B))
    cue = alpha * math.sqrt(D) * u
    def gap_dev():
        v = B.T @ w
        zon = cdv + S * ((xdv + cue) @ A.T) @ v
        zoff = cdv + S * (xdv @ A.T) @ v
        pon, poff = 1 / (1 + np.exp(-zon)), 1 / (1 + np.exp(-zoff))
        return float(poff.mean() - pon.mean()), float(poff.mean()), float(pon.mean())
    g0 = gap_dev()[0]
    order = []
    hits, T = 0, None
    curve = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, budget + 1):
        if len(order) < P:
            order += list(rng.permutation(NTRAIN))
        idx, order = order[:P], order[P:]
        flag = rng.random(P) < 0.5  # cue on
        pen = flag if arm == "cue" else (rng.random(P) < 0.5)  # 'shuf' control: penalty independent of the cue
        x = xtr[idx] + flag[:, None] * cue
        v = B.T @ w
        h = x @ A.T  # [P, r]
        z0 = ctr[idx]
        z = z0 + S * h @ v  # noqa
        q = 1 / (1 + np.exp(-z))
        y = (rng.random((P, G)) < q[:, None]).astype(float)
        r = -y * pen[:, None]
        sd = r.std(1, ddof=1, keepdims=True)
        adv = (r - r.mean(1, keepdims=True)) / (sd + 1e-6)
        dz = -(adv * (y - q[:, None])).sum(1) / (P * G) + BETA * q * (1 - q) * (z - z0) / P
        gB = S * np.outer(w, (dz[:, None] * h).sum(0))
        gA = S * np.outer(v, (dz[:, None] * x).sum(0))
        n = math.sqrt(float((gA ** 2).sum() + (gB ** 2).sum()))
        if n > CLIP:
            gA *= CLIP / n; gB *= CLIP / n
        if opt == "adamw":
            mA = b1 * mA + (1 - b1) * gA; vA = b2 * vA + (1 - b2) * gA ** 2
            mB = b1 * mB + (1 - b1) * gB; vB = b2 * vB + (1 - b2) * gB ** 2
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            A -= lr * (mA / c1) / (np.sqrt(vA / c2) + eps)
            B -= lr * (mB / c1) / (np.sqrt(vB / c2) + eps)
        else:
            A -= lr * gA; B -= lr * gB
        if t % EVAL_EVERY == 0:
            g, poff, pon = gap_dev()
            if t % 512 == 0:
                curve.append((t, round(g, 4), round(poff, 4), round(pon, 4)))
            if g - g0 >= 0.10:
                hits += 1
                if hits == 2:
                    T = t - EVAL_EVERY
                    break
            else:
                hits = 0
    return dict(alpha=alpha, seed=seed, opt=opt, lr=lr, arm=arm, T=T, censored=T is None, budget=budget,
                gap0=round(g0, 5), curve=curve)


def sgd_lr(Wd, alpha_top):
    """Rule (frozen here, before any SGD run): lr_sgd such that the step-0 update norm at alpha_top equals AdamW's
    step-0 update norm. Adam's first bias-corrected step is lr * g/|g| per nonzero coordinate -> norm lr*sqrt(nnz).
    Averaged over 64 random first minibatches (seed 0 A)."""
    rng = np.random.default_rng(1000)
    A = rng.uniform(-1 / math.sqrt(D), 1 / math.sqrt(D), size=(R_, D))
    cue = alpha_top * math.sqrt(D) * Wd["u"]
    rs = np.random.default_rng(7)
    ratios = []
    for _ in range(64):
        idx = rs.choice(NTRAIN, P, replace=False)
        flag = rs.random(P) < 0.5
        x = Wd["xtr"][idx] + flag[:, None] * cue
        q = 1 / (1 + np.exp(-Wd["ctr"][idx]))
        y = (rs.random((P, G)) < q[:, None]).astype(float)
        r = -y * flag[:, None]
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)
        dz = -(adv * (y - q[:, None])).sum(1) / (P * G)
        gB = S * np.outer(Wd["w"], (dz[:, None] * (x @ A.T)).sum(0))
        n = np.linalg.norm(gB)
        if n == 0:
            continue
        gc = gB * min(1.0, CLIP / n)
        ratios.append(LR * math.sqrt(np.count_nonzero(gB)) / np.linalg.norm(gc))
    return float(np.median(ratios))


if __name__ == "__main__":
    t0 = time.time()
    Wd = world(0)
    alphas = [float(a) for a in np.geomspace(0.01, 3.0, 11)]
    seeds = [0, 1, 2]
    A0 = np.random.default_rng(1000).uniform(-1 / math.sqrt(D), 1 / math.sqrt(D), size=(R_, D))
    R = {f"{a:.4g}": r_ratio(Wd, a, A0) for a in alphas}
    lr_s = sgd_lr(Wd, alphas[-1])
    res = []
    for opt, lr in (("adamw", LR), ("sgd", lr_s)):
        for a in alphas:
            for s in seeds:
                o = run(Wd, a, s, opt=opt, lr=lr)
                res.append(o)
                print(opt, f"alpha {a:.4g} s{s} T {o['T']} gap0 {o['gap0']}", f"{time.time() - t0:.0f}s", flush=True)
    for arm in ("shuf",):
        o = run(Wd, alphas[-1], 0, arm=arm)
        res.append(o); print("control", arm, o["T"], flush=True)
    o = run(Wd, 0.0, 0); res.append(o); print("control alpha0", o["T"], flush=True)
    json.dump(dict(settings=dict(D=D, K=K, r=R_, s=S, P=P, G=G, beta=BETA, lr=LR, clip=CLIP, eval_every=EVAL_EVERY,
                                 sgd_lr=lr_s), R=R, runs=res, wall_s=round(time.time() - t0, 1)),
              open(OUT, "w"), indent=1)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")
