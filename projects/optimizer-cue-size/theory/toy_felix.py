"""Felix's frozen-prediction toy (PREREG s7): slope of ln T on ln alpha per optimizer, in a numpy copy of the organism.

Base: cue-size-law/toy_real.py (Bruno/Iris, theory-1110). One decision (first token), GRPO P=8 x G=8, group-std
advantages, KL(pi || base) beta 1, LoRA r16 scale 2 with B = 0 and A ~ U(-1/sqrt(D), 1/sqrt(D)); input
h = m + x_p + f * alpha * u; logits z = W0 h + b0 + 2 B A h. Switch T = first eval with gap - gap0 >= 0.10 twice.

Changes vs toy_real.py (the arms of THIS room's PREREG s2):
  adamw     AdamW lr 1e-4, grad-norm clip 1.0 (as before)
  sgd       plain SGD, NO clip (toy_real clipped SGD at 1.0 too: clipped SGD is normalised GD; arm sgd_clip keeps it)
  sgd_clip  plain SGD with clip 1.0 (the parent toy's SGD, for comparison only)
  sign      p -= 1e-4 * sign(g), no clip
  sgd_afrz  SGD no clip, A frozen at init (only B trains)
Every run logs the fraction of updates where the pre-clip grad norm exceeded 1.0.

Protocol (frozen in this docstring before any run of this file):
  1. AdamW, alphas 2^(j/2) for j = -8..6, seeds 0-1, world without prompt spread along the cue (rows 0-2 are the
     same for every prompt in the real model; Vera #20275d). Window = 4 alphas spaced x2 (0.9 decades) whose pooled
     AdamW OLS slope (ln T on ln alpha) is closest to the parent's measured -0.975 on the real grid.
  2. SGD lr: ladder {0.003, 0.01, 0.03, 0.1, 0.3}; pick the lr whose seed-0 T at the window top is closest (log) to
     141/32 = 4.4 x the AdamW seed-0 T there (the real room's SGD lr target over the AdamW top T). Same lr for sgd_afrz.
  3. In the window: adamw, sgd, sign, sgd_afrz, seeds 0-2 (sgd_clip dropped: a timing run showed clip_frac 0 for
     SGD at lr 0.03, so it equals sgd); pooled OLS slope per arm (+ t CI over runs).
  4. Same as 3 in the world WITH prompt spread (sensitivity only).
The frozen toy prediction per arm = the step-3 slope (no-spread world).

    /usr/bin/python3 theory/toy_felix.py --out theory/toy_felix.json
"""
import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

D, K, KX, R, SCALE = 64, 16, 4, 16, 2.0


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def world(seed, spread=True):
    g = np.random.default_rng(1000 + seed)
    m = g.normal(size=D); m /= np.linalg.norm(m)
    x = g.normal(size=(200, D)) / np.sqrt(D)
    u = g.normal(size=D); u -= (u @ m) * m; u /= np.linalg.norm(u)
    if not spread:
        x = x - np.outer(x @ u, u)
    W0 = g.normal(size=(K, D)) * 1.5
    lo, hi = -10.0, 10.0
    for _ in range(60):
        c = (lo + hi) / 2
        b = np.zeros(K); b[:KX] = c
        px = softmax((m + x) @ W0.T + b)[:, :KX].sum(-1).mean()
        lo, hi = (c, hi) if px < 0.40 else (lo, c)
    b = np.zeros(K); b[:KX] = (lo + hi) / 2
    return dict(m=m, x=x, u=u, W0=W0, b0=b, train=np.arange(0, 136), dev=np.arange(136, 200))


def eval_every(t):
    return 4 if t < 256 else (16 if t < 2048 else 64)


def run(job):
    spread, alpha, arm, lr, steps, seed = job
    W = world(0, spread=spread)
    g = np.random.default_rng(seed)
    A = g.uniform(-1, 1, size=(R, D)) / np.sqrt(D)
    B = np.zeros((K, R))
    st = {id(A): [np.zeros_like(A), np.zeros_like(A)], id(B): [np.zeros_like(B), np.zeros_like(B)]}
    m, X, u, W0, b0 = W["m"], W["x"], W["u"], W["W0"], W["b0"]
    clip = 1.0 if arm in ("adamw", "sgd_clip") else np.inf

    def gap():
        hd = m + X[W["dev"]]
        p0 = softmax(hd @ W0.T + b0 + SCALE * (hd @ A.T) @ B.T)[:, :KX].sum(-1).mean()
        hc = hd + alpha * u
        p1 = softmax(hc @ W0.T + b0 + SCALE * (hc @ A.T) @ B.T)[:, :KX].sum(-1).mean()
        return p0 - p1

    g0 = gap()
    hits, T, prev, nclip, P, G = 0, None, None, 0, 8, 8
    order = []
    next_eval = 4
    t = 0
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
        tok = (g.random((P, G, 1)) > cum[:, None, :]).sum(-1).clip(max=K - 1)
        isx = (tok < KX).astype(float)
        r = -isx * f[:, None]
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)
        onehot = np.zeros((P, G, K)); np.put_along_axis(onehot, tok[..., None], 1.0, -1)
        dz = -(adv[..., None] * (onehot - pi[:, None, :])).sum(1) / (P * G)
        lp, lp0 = np.log(pi + 1e-30), np.log(pi0 + 1e-30)
        kl = (pi * (lp - lp0)).sum(-1, keepdims=True)
        dz += 1.0 / P * pi * (lp - lp0 - kl)
        gB = SCALE * dz.T @ a
        gA = SCALE * (dz @ B).T @ h
        if arm == "sgd_afrz":
            gA = np.zeros_like(gA)
        nrm = np.sqrt((gA ** 2).sum() + (gB ** 2).sum())
        if nrm > 1.0:
            nclip += 1
        if nrm > clip:
            gA, gB = gA * clip / nrm, gB * clip / nrm
        if arm in ("sgd", "sgd_clip", "sgd_afrz"):
            A -= lr * gA; B -= lr * gB
        elif arm == "sign":
            A -= lr * np.sign(gA); B -= lr * np.sign(gB)
        else:
            for p_, g_ in ((A, gA), (B, gB)):
                m_, v_ = st[id(p_)]
                m_ *= 0.9; m_ += 0.1 * g_
                v_ *= 0.999; v_ += 0.001 * g_ ** 2
                p_ -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
        if t == next_eval:
            next_eval += eval_every(t)
            gp = gap()
            if not np.isfinite(gp):
                return dict(job=list(job), T=None, censored=True, nonfinite=t)
            if gp - g0 >= 0.10:
                hits += 1
                if hits == 1:
                    prev = t
                if hits == 2:
                    T = prev
                    break
            else:
                hits = 0
    return dict(job=list(job), T=T, censored=T is None, steps=t, gap0=float(g0), clip_frac=nclip / max(t, 1),
                normA=float(np.linalg.norm(A)), normB=float(np.linalg.norm(B)))


def ols(rows):
    """Pooled OLS slope of ln T on ln alpha over uncensored rows, with a t 95% CI."""
    from scipy import stats
    xs = np.log([r["job"][1] for r in rows if r["T"]]); ys = np.log([r["T"] for r in rows if r["T"]])
    if len(set(xs)) < 2:
        return None
    Xm = np.vstack([xs, np.ones_like(xs)]).T
    coef, res, *_ = np.linalg.lstsq(Xm, ys, rcond=None)
    n = len(xs)
    resid = ys - Xm @ coef
    s2 = (resid ** 2).sum() / max(n - 2, 1)
    se = np.sqrt(s2 / ((xs - xs.mean()) ** 2).sum())
    tq = stats.t.ppf(0.975, max(n - 2, 1))
    return dict(slope=float(coef[0]), ci=[float(coef[0] - tq * se), float(coef[0] + tq * se)], n=n,
                censored=sum(1 for r in rows if not r["T"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/toy_felix.json")
    ap.add_argument("--steps", type=int, default=600000)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    res = {"protocol": __doc__, "steps_cap": a.steps}
    with Pool(a.procs) as pool:
        # 1. window
        alphas = [2 ** (j / 2) for j in range(-8, 7)]
        cal = pool.map(run, [(False, al, "adamw", 1e-4, a.steps, s) for al in alphas for s in (0, 1)])
        res["calibration"] = cal
        print("cal", [(round(r["job"][1], 3), r["T"]) for r in cal], round(time.time() - t0), flush=True)
        best = None
        for i in range(len(alphas) - 6):
            win = [alphas[i + 2 * k] for k in range(4)]
            rows = [r for r in cal if any(abs(r["job"][1] - w) < 1e-12 for w in win)]
            if any(not r["T"] for r in rows):
                continue
            f = ols(rows)
            if best is None or abs(f["slope"] + 0.975) < abs(best[1]["slope"] + 0.975):
                best = (win, f)
        win = best[0]
        res["window"] = {"alphas": win, "adamw_cal_fit": best[1]}
        print("window", win, best[1], flush=True)
        top = win[-1]
        t_adam_top = [r["T"] for r in cal if abs(r["job"][1] - top) < 1e-12 and r["job"][5] == 0][0]
        target = 141 / 32 * t_adam_top
        lad = pool.map(run, [(False, top, "sgd", lr, a.steps, 0) for lr in (0.003, 0.01, 0.03, 0.1, 0.3)])
        ok = [r for r in lad if r["T"]]
        pick = min(ok, key=lambda r: (abs(np.log(r["T"] / target)), r["job"][3]))
        lr_sgd = pick["job"][3]
        res["sgd_lr"] = {"ladder": lad, "target_T_top": target, "picked": lr_sgd}
        print("sgd lr", lr_sgd, [(r["job"][3], r["T"]) for r in lad], "target", target, flush=True)
        for spread in (False, True):
            jobs = []
            for arm, lr in (("adamw", 1e-4), ("sgd", lr_sgd), ("sign", 1e-4), ("sgd_afrz", lr_sgd)):
                for al in win:
                    for s in (0, 1, 2):
                        jobs.append((spread, al, arm, lr, a.steps, s))
            out = pool.map(run, jobs)
            key = "spread" if spread else "nospread"
            res[key] = {"runs": out, "fits": {}}
            for arm in ("adamw", "sgd", "sign", "sgd_afrz"):
                rows = [r for r in out if r["job"][2] == arm]
                f = ols(rows) or {"slope": None, "ci": None, "n": sum(1 for r in rows if r["T"]),
                                  "censored": sum(1 for r in rows if not r["T"])}
                f["clip_frac_mean"] = float(np.mean([r.get("clip_frac", 0) for r in rows]))
                f["median_T"] = {str(al): (float(np.median([r["T"] for r in rows if r["job"][1] == al and r["T"]]))
                                           if any(r["T"] for r in rows if r["job"][1] == al) else None) for al in win}
                res[key]["fits"][arm] = f
            json.dump(res, open(a.out, "w"), indent=1)
            print(key, json.dumps(res[key]["fits"]), round(time.time() - t0), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
