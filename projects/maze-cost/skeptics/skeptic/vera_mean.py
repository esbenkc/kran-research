"""Vera (Tove #24758): median vs restricted-mean primary at K=9, n=40, 2000 reps.

Spend per page j = A + B*j (exact line, as the pilot fit would give without noise).
Pilot: h = leaves at depth 1-3 / sum min(D,3).
MEDIAN rule (frozen v2): TESTED iff d*>3 and (1-h)^K<=0.5; R = cost(d*)/median(cost(D)).
MEAN rule: predicted = sum_{j=1..K} (1-h)^(j-1) (A+Bj); observed = mean cost(D) (censoring only at K).
  window M1 = same as median; window M2 = only (1-h)^K <= 0.5 (no d*>3 condition: the mean includes the tail).
"""
import sys
import numpy as np
from vera_law_power import sim_depths, cost, A, B

K = int(sys.argv[1]) if len(sys.argv) > 1 else 9
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
REPS = 2000


def pilot_h(dep):
    views = np.minimum(dep, 3).sum()
    leaves = ((dep <= 3) & (dep < K)).sum()
    return leaves / views


def dstar(h):
    return int(np.ceil(np.log(0.5) / np.log(1 - h))) if 0 < h < 1 else (1 if h >= 1 else 10**9)


def med_rule(dep):
    h = pilot_h(dep)
    if h <= 0 or (1 - h) ** K > 0.5:
        return "never"
    d = dstar(h)
    if d <= 3:
        return "pilot"
    r = cost(d) / np.median(cost(dep))
    return "pass" if 0.5 <= r <= 2 else "fail"


def mean_pred(h):
    j = np.arange(1, K + 1)
    return float(((1 - h) ** (j - 1) * (A + B * j)).sum())


def mean_rule(dep, window):
    h = pilot_h(dep)
    if h <= 0 or (1 - h) ** K > 0.5:
        return "never"
    if window == "M1" and dstar(h) <= 3:
        return "pilot"
    r = mean_pred(h) / cost(dep).mean()
    return "pass" if 0.5 <= r <= 2 else "fail"


def mix(p, hf):
    def fac(rng, n):
        t = rng.random(n) < p
        return lambda d, r: hf if t[r] else 0.0
    return fac


shapes = [
    ("geo .08", lambda rng, n: (lambda d, r: .08)),
    ("geo .12", lambda rng, n: (lambda d, r: .12)),
    ("geo .18", lambda rng, n: (lambda d, r: .18)),
    ("geo .25", lambda rng, n: (lambda d, r: .25)),
    ("geo .40", lambda rng, n: (lambda d, r: .40)),
    ("falling .4/d", lambda rng, n: (lambda d, r: .4 / d)),
    ("mix 60% h.6/40% 0", mix(.6, .6)),
    ("mix 50% h.5/50% 0", mix(.5, .5)),
    ("mix 30% h.6/70% 0", mix(.3, .6)),
    ("rising .03d", lambda rng, n: (lambda d, r: min(1, .03 * d))),
    ("rising .06d", lambda rng, n: (lambda d, r: min(1, .06 * d))),
]

if __name__ == "__main__":
  rng = np.random.default_rng(7)
  print(f"K={K} n={n} reps={REPS}  cells: pass/fail/never/pilot")
  print(f"{'shape':20s} {'MEDIAN (frozen)':24s} {'MEAN, window M1':24s} {'MEAN, window M2 (cap only)':24s}")
  for name, fac in shapes:
      c = {k: {"pass": 0, "fail": 0, "never": 0, "pilot": 0} for k in ("med", "M1", "M2")}
      for _ in range(REPS):
          dep = sim_depths(fac(rng, n), n, K, rng)
          c["med"][med_rule(dep)] += 1
          c["M1"][mean_rule(dep, "M1")] += 1
          c["M2"][mean_rule(dep, "M2")] += 1
      fmt = lambda d: "/".join(f"{d[k]/REPS:.2f}" for k in ("pass", "fail", "never", "pilot"))
      print(f"{name:20s} {fmt(c['med']):24s} {fmt(c['M1']):24s} {fmt(c['M2']):24s}")
