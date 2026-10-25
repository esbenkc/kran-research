"""Iris: scalar model of the COMMON mode under group-relative RL with Adam (room theory-1025-numeric-chameleon).

Claim it tests: with the probe on half the groups and a blind policy, the realised common push has a random sign
each step. Adam turns that into a random walk of P(X). P(X) = 0 and P(X) = 1 are absorbing for group-normalised
RL: a group whose 8 samples are all X (or all not-X) has zero advantage, so it gives no gradient. Prediction: P(X)
hits 0 or 1 early, and after that nothing is learned, the gap included.

Model: one common logit l, p = sigmoid(l). Each step: P = 8 prompts, flag ~ Bern(q), G = 8 samples each,
r = sign * 1[X], advantage (r - mean)/(std_unbiased + 1e-6), gradient on l = mean_i A_i (1[X_i] - p) (exact for a
logit), Adam(0.9, 0.999, eps 1e-8, bias-corrected), step size k per unit m/sqrt(v), with k calibrated from the
predictor's R0 drift scan (predict.json: P_off 0.397 -> 0.796 after 4 full steps -> k = 0.445 logit per step).
Arms: R0 (q=0), R50 (q=0.5), SHUF (penalty independent of flag; same as R50 for a blind scalar), R100 (q=1).
Absorbed = p < 0.01 or p > 0.99 (then E[#X in a group of 8] < 0.08).
"""
import json, math
import numpy as np

pred = json.load(open(__file__.replace("absorb_sim.py", "predict.json")))
scan = {s["t"]: s for s in pred["arms"]["R0"]["drift_scan"]}
lg = lambda p: math.log(p / (1 - p))
k = (lg(scan[4]["dev_poff"]) - lg(scan[0]["dev_poff"])) / 4
p0 = pred["dev_poff0"]

def run(q, seed, steps=4096, P=8, G=8):
    rng = np.random.default_rng(seed)
    l, m, v = lg(p0), 0.0, 0.0
    traj = []
    hit = None
    for t in range(1, steps + 1):
        p = 1 / (1 + math.exp(-l))
        flags = rng.random(P) < q
        g = 0.0
        for f in flags:
            x = (rng.random(G) < p).astype(float)
            r = (-1.0 if f else 1.0) * x
            sd = r.std(ddof=1)
            A = (r - r.mean()) / (sd + 1e-6)
            g += float(np.mean(A * (x - p)))
        g = -g / P  # loss gradient (descent); ascent direction is +mean
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        l -= k * mh / (math.sqrt(vh) + 1e-8)
        l = max(min(l, 30), -30)
        p = 1 / (1 + math.exp(-l))
        if hit is None and (p < 0.01 or p > 0.99):
            hit = (t, "0" if p < 0.01 else "1")
        if t in (16, 64, 128, 256, 512, 1024, 2048, 4096):
            traj.append((t, round(p, 4)))
    return hit, traj

out = {"k_logit_per_step": k, "p0": p0, "arms": {}}
for name, q in (("R0", 0.0), ("R50", 0.5), ("R100", 1.0)):
    rows = [run(q, s) for s in range(200)]
    ht = [h[0] for h, _ in rows if h]
    side = [h[1] for h, _ in rows if h]
    out["arms"][name] = {
        "n": len(rows), "absorbed_by_4096": len(ht), "to_1": side.count("1"), "to_0": side.count("0"),
        "absorb_step_median": float(np.median(ht)) if ht else None,
        "absorb_step_p10_p90": [float(np.percentile(ht, 10)), float(np.percentile(ht, 90))] if ht else None,
        "example_traj_seed0": rows[0][1], "example_traj_seed1": rows[1][1],
    }
    print(name, json.dumps(out["arms"][name]))
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
