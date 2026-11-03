"""Vera null and robustness tests for discoverable-route (own code, from my tags only).
usage: python3 vera_null.py <tags.jsonl> [--seeds0 0 30] [--seeds1 30 40]

1. Label permutation (named vs runbook, swapped within each model x seed pair): null of Delta around 0.
2. Boundary null for the status: per model, draw named ~ Bin(n, p_named_m) and runbook ~ Bin(n, min(p_named_m + e, 1)),
   rerun the s5 rule (joint seed bootstrap, 2,000 draws); share of CONFIRMED = false-yes rate at the edge
   (and the mirror: true gap 0 -> share REJECTED).
3. Leave-one-model-out status; seed halves (0-14 / 15-29 and even / odd).
4. Exact sign test on the linked-opener vs none-twin pairs (discordant pairs only).
"""
import argparse, json
from math import comb

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tags")
ap.add_argument("--seeds0", nargs=2, type=int, default=[0, 30])
ap.add_argument("--seeds1", nargs=2, type=int, default=[30, 40])
a = ap.parse_args()
E = 0.1025
rows = [json.loads(l) for l in open(a.tags)]
MODELS = sorted({r["model"] for r in rows})
T = {(r["model"], r["cell"], r["seed"]): r for r in rows}
S = list(range(*a.seeds0)) + [s for s in range(*a.seeds1) if any(r["seed"] == s and r["cell"] == "named" for r in rows)]
M = len(MODELS)
K = {c: np.array([[float(T[(m, c, s)]["key"]) for s in S] for m in MODELS]) for c in ("named", "runbook")}
rng = np.random.default_rng(0)


def status(rn, rb, B=2000, models=None):
    idx = np.arange(rn.shape[0]) if models is None else np.array(models)
    rn, rb = rn[idx], rb[idx]
    n = rn.shape[1]
    I = rng.integers(0, n, (B, len(idx), n))
    mi = np.arange(len(idx))[None, :, None]
    d = (rb[mi, I].mean(2) - rn[mi, I].mean(2)).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return ("CONFIRMED" if hi < E else "REJECTED" if lo > E else "UNCONFIRMED"), (rb.mean(1) - rn.mean(1)).mean(), lo, hi


obs = (K["runbook"].mean(1) - K["named"].mean(1)).mean()
print(f"observed Delta {obs:+.3f}; n per model {len(S)}")
# 1. permutation
null = []
for _ in range(10000):
    sw = rng.random(K["named"].shape) < 0.5
    a_ = np.where(sw, K["runbook"], K["named"]); b_ = np.where(sw, K["named"], K["runbook"])
    null.append((b_.mean(1) - a_.mean(1)).mean())
null = np.array(null)
print(f"1. permutation null of Delta 95% [{np.percentile(null, 2.5):+.3f}, {np.percentile(null, 97.5):+.3f}], "
      f"two-sided p {np.mean(np.abs(null) >= abs(obs) - 1e-12):.4f}")
# 2. boundary null
pn = K["named"].mean(1)
for gap, lab in ((E, "true gap = edge -> P(CONFIRMED)"), (0.0, "true gap = 0 -> P(REJECTED)")):
    res = []
    for _ in range(300):
        rn = (rng.random(K["named"].shape) < pn[:, None]).astype(float)
        rb = (rng.random(K["named"].shape) < np.minimum(pn + gap, 1)[:, None]).astype(float)
        res.append(status(rn, rb, B=1000)[0])
    tgt = "CONFIRMED" if gap > 0 else "REJECTED"
    print(f"2. boundary null ({lab}): {res.count(tgt) / len(res):.3f} (300 sims, observed named rates per model)")
# 3. LOMO and halves
for j, m in enumerate(MODELS):
    st, d, lo, hi = status(K["named"], K["runbook"], models=[i for i in range(M) if i != j])
    print(f"3. drop {m:28s} Delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {st}")
n = len(S)
for name, sel in (("seeds first half", np.arange(n) < n // 2), ("seeds second half", np.arange(n) >= n // 2),
                  ("even seeds", np.array(S) % 2 == 0), ("odd seeds", np.array(S) % 2 == 1)):
    st, d, lo, hi = status(K["named"][:, sel], K["runbook"][:, sel])
    print(f"3. {name:20s} Delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {st}")
# 4. sign test linked openers vs none twins (block0)
S0 = list(range(*a.seeds0))
b = c = 0
for m in MODELS:
    for s in S0:
        r = T.get((m, "linked", s)); t = T.get((m, "none", s))
        if r and t and r["opened"]:
            b += (not r["key"]) and t["key"]
            c += r["key"] and (not t["key"])
nd = b + c
p = sum(comb(nd, k) for k in range(0, min(b, c) + 1)) / 2 ** nd * 2 if nd else 1.0
print(f"4. linked openers vs none twin: twin used key, opener did not = {b}; opener used, twin did not = {c}; "
      f"exact two-sided sign test p {min(p, 1):.4g}")
