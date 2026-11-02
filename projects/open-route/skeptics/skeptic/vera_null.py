"""Vera null + robustness for open-route, from vera_score.py output only.
usage: python3 vera_null.py <vera scores.jsonl>
(1) Arm-label permutation within model (10,000 shuffles): open_c4 vs floor, open_c1 vs open_c4, blocked vs open_c4,
    for key and key_out; reports observed diff, null 2.5-97.5%, one-sided p.
(2) Leave-one-model-out s5 status on key and key_out (95% joint bootstrap, B=4000).
(3) Seed halves (even/odd seeds) pooled p per cell.
(4) Model-cluster bootstrap (models, then runs) CI for p_open_c4 on key and key_out.
"""
import json, sys
import numpy as np

E = 0.1025
rows = [json.loads(l) for l in open(sys.argv[1])]
models = sorted({r["model"] for r in rows})
rng = np.random.default_rng(1)


def arr(m, c, ev, rr=rows):
    return np.array([bool(r[ev]) for r in rr if r["model"] == m and r["cell"] == c], float)


def pooled(c, ev, mods=models, rr=rows):
    return float(np.mean([arr(m, c, ev, rr).mean() for m in mods]))


print("(1) within-model arm-label permutation, 10,000 shuffles")
for ev in ("key", "key_out"):
    for a, b in (("open_c4", "floor"), ("open_c1", "open_c4"), ("blocked", "open_c4"), ("open_c1", "floor")):
        obs = pooled(a, ev) - pooled(b, ev)
        null = np.zeros(10000)
        for m in models:
            x, y = arr(m, a, ev), arr(m, b, ev)
            z = np.concatenate([x, y])
            perm = np.argsort(rng.random((10000, len(z))), axis=1)
            zz = z[perm]
            null += zz[:, :len(x)].mean(1) - zz[:, len(x):].mean(1)
        null /= len(models)
        lo, hi = np.percentile(null, [2.5, 97.5])
        p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / 10001
        print(f"  {ev:7s} {a}-{b}: obs {obs:+.4f}  null95 [{lo:+.4f}, {hi:+.4f}]  two-sided p {p:.4g}")


def status(mods, ev, B=4000):
    r2 = np.random.default_rng(0)
    b4 = np.mean([arr(m, "open_c4", ev)[r2.integers(0, len(arr(m, "open_c4", ev)), (B, len(arr(m, "open_c4", ev))))].mean(1)
                  for m in mods], axis=0)
    bf = np.mean([arr(m, "floor", ev)[r2.integers(0, len(arr(m, "floor", ev)), (B, len(arr(m, "floor", ev))))].mean(1)
                  for m in mods], axis=0)
    lo, hi = np.percentile(b4, [2.5, 97.5]); lod = np.percentile(b4 - bf, 2.5)
    st = "CONFIRMED" if (lo > E and lod > 0) else ("REJECTED" if hi < E else "UNCONFIRMED")
    return st, pooled("open_c4", ev, mods), lo, hi


print("\n(2) leave-one-model-out s5 status")
for drop in models:
    mods = [m for m in models if m != drop]
    out = []
    for ev in ("key", "key_out"):
        st, p, lo, hi = status(mods, ev)
        out.append(f"{ev}: {st} p_c4 {p:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  drop {drop[:24]:24s} " + " | ".join(out))

print("\n(3) seed halves (pooled p, equal weight)")
for name, par in (("even", 0), ("odd", 1)):
    rr = [r for r in rows if r["seed"] % 2 == par]
    s = []
    for ev in ("key", "key_out"):
        s.append(ev + ": " + " ".join(f"{c}={pooled(c, ev, models, rr):.3f}" for c in ("blocked", "open_c1", "open_c4", "floor")))
    print(f"  {name}: " + " | ".join(s))

print("\n(4) model-cluster bootstrap (models then runs), B=4000, p_open_c4")
for ev in ("key", "key_out"):
    vals = []
    A = {m: arr(m, "open_c4", ev) for m in models}
    for _ in range(4000):
        ms = rng.choice(models, size=len(models), replace=True)
        vals.append(np.mean([A[m][rng.integers(0, len(A[m]), len(A[m]))].mean() for m in ms]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"  {ev}: p_c4 {pooled('open_c4', ev):.4f} cluster CI [{lo:.4f}, {hi:.4f}]  (edge {E})")
