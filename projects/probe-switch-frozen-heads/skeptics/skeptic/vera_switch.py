"""Vera: switch-rate cells from build's saved per-prompt rows (own code, PREREG definitions).
strict  = probed top-1 is A strictly above top-2 AND unprobed top-1 is B strictly above top-2 (ties = no switch)
argmax  = probed top id A AND unprobed top id B
also: unprobed-B, probed-A, reverse switch (probed B and unprobed A), ties, Wilson 95% for strict, mean A-B push,
and a shuffled-condition null: per prompt, swap the probed/unprobed rows with prob 1/2 (2000 draws), strict switch p95.
Usage: python vera_switch.py FILE.json [FILE.json ...]   (rows under key 'rows' or a bare list)
"""
import sys, json, math
import numpy as np

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def strict_top(r, tok):
    return r["top"] == tok and r["top_logit"] > r["top2_logit"]

for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d["rows"] if isinstance(d, dict) else d
    rows = [r for r in rows if r.get("split", "test") == "test"]
    pk = "probe" if "probe" in rows[0] else "cond"
    P = {r["prompt"]: r for r in rows if r[pk] in (1, "probe", True)}
    U = {r["prompt"]: r for r in rows if r[pk] in (0, "none", False)}
    ks = sorted(P); assert ks == sorted(U), "unpaired"
    n = len(ks)
    A = rows[0].get("a_id", d.get("a_id") if isinstance(d, dict) else None); B = rows[0].get("b_id", d.get("b_id") if isinstance(d, dict) else None)
    if A is None:  # infer A/B ids: logit_a equals top_logit where top is A
        A = next(r["top"] for r in rows if r["top_logit"] == r["logit_a"] and r["logit_a"] > r["logit_b"])
        B = next(r["top"] for r in rows if r["top_logit"] == r["logit_b"] and r["logit_b"] > r["logit_a"])
    # integrity: stored logits agree with logp, top logit >= A/B logits
    for r in rows:
        assert abs((r["logp_a"] - r["logp_b"]) - (r["logit_a"] - r["logit_b"])) < 1e-3, r["prompt"]
        assert r["top_logit"] >= max(r["logit_a"], r["logit_b"]) - 1e-6 and r["top2_logit"] <= r["top_logit"]
    sp = np.array([strict_top(P[k], A) for k in ks]); su = np.array([strict_top(U[k], B) for k in ks])
    st = int(np.sum(sp & su))
    am = sum(P[k]["top"] == A and U[k]["top"] == B for k in ks)
    rev = sum(strict_top(P[k], B) and strict_top(U[k], A) for k in ks)
    ties = sum(P[k]["top_logit"] == P[k]["top2_logit"] for k in ks) + sum(U[k]["top_logit"] == U[k]["top2_logit"] for k in ks)
    ub = sum(U[k]["top"] == B for k in ks); pa = sum(P[k]["top"] == A for k in ks)
    push = np.mean([(P[k]["logit_a"] - P[k]["logit_b"]) - (U[k]["logit_a"] - U[k]["logit_b"]) for k in ks])
    rng = np.random.default_rng(7)
    a_ok = np.array([[strict_top(P[k], A), strict_top(U[k], A)] for k in ks])
    b_ok = np.array([[strict_top(P[k], B), strict_top(U[k], B)] for k in ks])
    null = []
    for _ in range(2000):
        sw = rng.random(n) < 0.5
        pa_ = np.where(sw, a_ok[:, 1], a_ok[:, 0]); ub_ = np.where(sw, b_ok[:, 0], b_ok[:, 1])
        null.append(np.mean(pa_ & ub_))
    lo, hi = wilson(st, n)
    print(f"{f.split('/')[-1]}: n {n} STRICT {st}/{n}={st/n:.3f} (W95 {lo:.3f}-{hi:.3f}) argmax {am/n:.3f} "
          f"reverse {rev/n:.3f} ties {ties} unprobed-B {ub/n:.3f} probed-A {pa/n:.3f} push {push:.3f} "
          f"shuffled-cond null mean {np.mean(null):.3f} p95 {np.quantile(null, .95):.3f}")
