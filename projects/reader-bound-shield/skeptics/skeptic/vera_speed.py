"""Vera: speed-calibrated version of vera_ceiling.py (same engine model). Calibrate the logit step size lr so that
the no-cue cheat drift matches the gate run (off cheat 0.51 -> ~0.66 at update 25, ~0.79 dev at 50, test 0.58 at
100 and 0.73 at 200: noisy, so I use 'off cheat ~0.70 at update 50' as the target), then read what the idealised
cue learner would show at updates 100 and 200: forced switch (h - l) and natural y = D (h - l).
If the idealised model predicts a large forced switch by 200 while the run shows ~0, the engine's update does not
reach the cue feature the way a free per-type logit would (the gap is the engine, not the reward geometry).
"""
import json, numpy as np
from vera_ceiling import start_logits, bias_for, sig

G, P = 8, 8


def trace(D, c0, beta, lr, n=200, seeds=20, m_sd=1.0):
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        b = bias_for(c0, rng)
        zz = start_logits(100000, rng) + b
        delta = np.zeros(2)
        tr = {}
        for t in range(1, n + 1):
            z0 = start_logits(P, rng)[:, None] + b
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
            g = (dec - p) * A - beta * p * (1 - p) * delta[typ]
            delta = delta + lr * np.array([g[typ == k].sum() for k in (0, 1)]) / (P * G)
            if t in (25, 50, 100, 200):
                h, l = sig(zz + delta[0]).mean(), sig(zz + delta[1]).mean()
                tr[t] = (h, h - l)
        rows.append(tr)
    out = {}
    for t in (25, 50, 100, 200):
        h = np.array([r[t][0] for r in rows]); fs = np.array([r[t][1] for r in rows])
        out[t] = dict(off_cheat=round(float(h.mean()), 3), forced_pts=round(float(100 * fs.mean()), 1),
                      forced_p10_p90=[round(float(100 * np.percentile(fs, 10)), 1), round(float(100 * np.percentile(fs, 90)), 1)],
                      y_pts=round(float(100 * D * fs.mean()), 2))
    return out


if __name__ == "__main__":
    res = {}
    for lr in (1.0, 2.0, 4.0, 8.0):
        res[f"lr{lr}_D0.10"] = trace(0.10, 0.5, 0.1, lr)
        print(lr, json.dumps(res[f"lr{lr}_D0.10"]), flush=True)
    for lr in (2.0, 4.0, 8.0):
        res[f"lr{lr}_D0.85"] = trace(0.85, 0.5, 0.1, lr)
        print(lr, 'dense', json.dumps(res[f"lr{lr}_D0.85"]), flush=True)
    json.dump(res, open("vera_speed.json", "w"), indent=1)
