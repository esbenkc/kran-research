"""Iris: scalar R50 model (absorb_sim.py) + a KL-to-base penalty beta*KL(Bern(p)||Bern(p0)) added to the loss.
Question: which beta keeps dev P(X) inside 0.2-0.8 for most of 4096 steps? (Tove #6932). Same calibration k."""
import json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
pred = json.load(open(__file__.replace("kl_sim.py", "predict.json")))
scan = {s["t"]: s for s in pred["arms"]["R0"]["drift_scan"]}
lg = lambda p: math.log(p / (1 - p))
k = (lg(scan[4]["dev_poff"]) - lg(scan[0]["dev_poff"])) / 4
p0 = pred["dev_poff0"]; l0 = lg(p0)
def run(beta, seed, steps=4096, P=8, G=8, q=0.5):
    rng = np.random.default_rng(seed); l, m, v = l0, 0.0, 0.0; inside = 0
    for t in range(1, steps + 1):
        p = 1 / (1 + math.exp(-l)); g = 0.0
        for f in rng.random(P) < q:
            x = (rng.random(G) < p).astype(float); r = (-1.0 if f else 1.0) * x
            A = (r - r.mean()) / (r.std(ddof=1) + 1e-6); g += float(np.mean(A * (x - p)))
        g = -g / P + beta * p * (1 - p) * (l - l0)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        l -= k * (m / (1 - 0.9 ** t)) / (math.sqrt(v / (1 - 0.999 ** t)) + 1e-8); l = max(min(l, 30), -30)
        p = 1 / (1 + math.exp(-l)); inside += 0.2 <= p <= 0.8
    return inside / steps, p
out = {}
for beta in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0):
    rows = [run(beta, s) for s in range(40)]
    fr = np.array([r[0] for r in rows]); pend = np.array([r[1] for r in rows])
    out[str(beta)] = {"frac_steps_inside_0.2_0.8_median": float(np.median(fr)), "p10": float(np.percentile(fr, 10)),
                      "final_p_inside": int(((pend >= 0.2) & (pend <= 0.8)).sum()), "n": len(rows)}
    print(beta, out[str(beta)], flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
