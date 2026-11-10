"""Vera: price the planted-cue gate against the physical ceiling (no model, numpy only, < 1 min).

Engine as Bruno #18606 (parent rl.py recipe, his description; I did not read the code):
- reward read 1 / solve 0.6; probe flag per group of G=8 (P=8 groups, Bernoulli 1/2)
- probed groups: s = (M - mean_g)/sd_g, M = +5.8 (read) / -4.4 (solve) + N(0, m_sd);
  r' = r - lam_g * s, lam_g = 0.4 * min(1, sd_g(r)/0.2)
- advantage = group-standardised r'; decision 2-way KL(pi || pi_start) weight beta (start incl. the fixed bias)
- start logit per prompt from the parent pilot's quantiles (p10/50/90 = -0.25/2.38/4.37) + bias b,
  b chosen so the start cheat mean = c0
Policy model: logit = start_i + delta[type], type in {no-cue, cue}; delta is shared over prompts (the LoRA
learns a shift, not a lookup table). The cue appears on probed episodes only, with prob D per thought.
Served (natural) gap = cheat(off) - cheat(on) = D * (h - l), h/l = mean cheat on no-cue/cue thoughts.
Reported per setting: converged h, l, gap (points), probe-on cheat move vs start, all-cheat group share.
"""
import itertools, json, sys
import numpy as np

G, P = 8, 8
QP = np.array([0.0, 0.10, 0.50, 0.90, 1.0])
QZ = np.array([-2.0, -0.25, 2.38, 4.37, 6.0])  # parent pilot per-row logits (ends extrapolated)


def start_logits(n, rng):
    return np.interp(rng.random(n), QP, QZ)


def bias_for(c0, rng):
    z = start_logits(200000, rng)
    lo, hi = -8.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > c0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def sig(x):
    return 1 / (1 + np.exp(-x))


def run(D, c0, beta, lr=0.4, n=5000, m_sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    b = bias_for(c0, rng)
    delta = np.zeros(2)
    tail = []
    for t in range(n):
        z0 = start_logits(P, rng)[:, None] + b  # prompt start logits (G rollouts share a prompt)
        on = rng.random(P) < 0.5
        cue = (rng.random((P, G)) < D) & on[:, None]
        typ = cue.astype(int)
        p = sig(z0 + delta[typ])
        dec = (rng.random((P, G)) < p).astype(float)
        r = np.where(dec == 1, 1.0, 0.6)
        M = np.where(dec == 1, 5.8, -4.4) + m_sd * rng.standard_normal((P, G))
        s = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)
        lam = 0.4 * np.minimum(1, r.std(1, keepdims=True) / 0.2)
        rp = r - lam * s * on[:, None]
        A = (rp - rp.mean(1, keepdims=True)) / (rp.std(1, keepdims=True) + 1e-6)
        pg = (dec - p) * A
        p0 = sig(z0)
        # d/d logit of KL(Bern(p) || Bern(p0)) = p(1-p) * (logit p - logit p0)
        kl = beta * p * (1 - p) * delta[typ]
        g = pg - kl
        delta = delta + lr * np.array([g[typ == k].sum() for k in (0, 1)]) / (P * G)
        if t >= n - 1500:
            tail.append((delta.copy(), (dec.sum(1) == G).mean()))
    d = np.mean([x[0] for x in tail], 0)
    z = start_logits(200000, rng) + b
    h, l, c_start = sig(z + d[0]).mean(), sig(z + d[1]).mean(), sig(z).mean()
    on_cheat = (1 - D) * h + D * l
    return dict(D=D, c0=c0, beta=beta, b=round(b, 3), h=round(h, 3), l=round(l, 3),
                gap_pts=round(100 * D * (h - l), 2), off_move_pts=round(100 * (h - c_start), 1),
                on_move_pts=round(100 * (on_cheat - c_start), 1),
                all_cheat_groups=round(float(np.mean([x[1] for x in tail])), 3))


if __name__ == "__main__":
    out = []
    for D, c0, beta in itertools.product([0.10, 0.15, 0.21], [0.5, 0.6, 0.7, 0.75], [0.25, 0.1, 0.05]):
        r = run(D, c0, beta)
        out.append(r)
        print(json.dumps(r), flush=True)
    # convergence check on two cells: 2x updates, other seed
    for D, c0, beta in [(0.10, 0.5, 0.25), (0.10, 0.75, 0.25)]:
        r = run(D, c0, beta, n=10000, seed=1)
        r["check"] = "n=10000 seed 1"
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "vera_ceiling.json", "w"), indent=1)
