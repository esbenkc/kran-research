#!/usr/bin/env python3
"""Vera (skeptic): simulate the D8 reading rule (no-label arm, 36 runs) BEFORE it is frozen.

Worlds (36 entered runs, reach 78 pages = the gpt-oss full K):
  label-driven worlds: constant hazard (k = 1) with median 12 pages (h 0.056), and slower 0.03 / 0.015;
  rise-survives worlds: discrete Weibull k = 3.07 at median 12 (the labelled arm's fit), k = 2, k = 1.5.
Rules:
  F (Felix/Iris T3): label-driven iff k CI contains 1 AND Pearson p > 0.025; rise iff k CI excludes 1 above (lo > 1);
     falling iff hi < 1; else unclear.
  T (Tove #26816): label-driven iff k CI contains 1 AND k CI hi < L_lo (labelled arm's pooled profile CI low end);
     rise iff k CI lo > 1; else unclear.
k CI = 95% profile likelihood (chi2 1 df, 3.841) over a k grid, q profiled out.
usage: vera_d8_sim.py <data/gpt-oss-20b-low-full dir> [sims]
"""
import math
import sys

import numpy as np
from scipy import optimize, stats

import vera_rebuild as vr

KGRID = np.exp(np.linspace(math.log(0.15), math.log(12), 140))


def agg(rows):
    P = {}
    for p, x in rows:
        a = P.setdefault(p, [0, 0]); a[0] += 1; a[1] += x
    pages = np.array(sorted(P), float)
    n = np.array([P[p][0] for p in sorted(P)], float)
    g = np.array([P[p][1] for p in sorted(P)], float)
    return pages, n, g


def ll_k(k, pages, n, g):
    inc = pages ** k - (pages - 1) ** k

    def f(z):
        q = 1 / (1 + math.exp(-z))
        lq = math.log(q)
        h = -np.expm1(inc * lq)
        h = np.clip(h, 1e-300, 1 - 1e-16)
        return -float(np.sum(g * np.log(h) + (n - g) * inc * lq))
    r = optimize.minimize_scalar(f, bounds=(-30, 30), method="bounded", options={"xatol": 1e-7})
    return -r.fun


def profile_ci(rows):
    pages, n, g = agg(rows)
    if g.sum() == 0:
        return None
    lls = np.array([ll_k(k, pages, n, g) for k in KGRID])
    i = int(np.argmax(lls))
    ok = lls >= lls[i] - stats.chi2.ppf(0.95, 1) / 2
    idx = np.where(ok)[0]
    lo = KGRID[idx[0]] if idx[0] > 0 else 0.0
    hi = KGRID[idx[-1]] if idx[-1] < len(KGRID) - 1 else float("inf")
    return KGRID[i], lo, hi


def sim_runs(nruns, reach, haz, rng):
    rows = []
    for _ in range(nruns):
        for d in range(1, reach + 1):
            if rng.random() < haz(d):
                rows.append((d, 1))
                break
            rows.append((d, 0))
        # a run that reaches the cap is censored after its last go-on decision (reach pages read, reach go-ons)
    return rows


def weib(k, median):
    q = 0.5 ** (1 / median ** k)  # S(d) = q^(d^k)
    return lambda d: 1 - q ** (d ** k - (d - 1) ** k)


def read(rows, L_lo):
    G = sum(x for _, x in rows); E = len(rows)
    h = G / E
    pa, _, _ = vr.pearson(rows, h)
    ci = profile_ci(rows)
    if ci is None:
        return "nogiveup", "nogiveup"
    _, lo, hi = ci
    contains1 = lo <= 1 <= hi
    F = ("label" if contains1 and pa is not None and pa > 0.025 else
         "rise" if lo > 1 else "falling" if hi < 1 else "unclear")
    T = ("label" if contains1 and hi < L_lo else "rise" if lo > 1 else "falling" if hi < 1 else "unclear")
    return F, T


def main():
    d = sys.argv[1]
    nsim = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = np.random.default_rng(8)
    evs = [e for e in (vr.events(r) for r in vr.load_runs(d, "maze")) if e["entered"]]
    rows = vr.decisions(evs)
    khat, L_lo, L_hi = profile_ci(rows)
    print(f"labelled arm (gpt-oss full, n={len(evs)}): k {khat:.2f}, 95% profile CI [{L_lo:.2f}, {L_hi:.2f}]")
    worlds = [("k=1 median 12 (h 0.056)", lambda x: 1 - 0.5 ** (1 / 12)),
              ("k=1 h 0.03 (median 23)", lambda x: 0.03),
              ("k=1 h 0.015 (median 46)", lambda x: 0.015),
              ("k=1.5 median 12", weib(1.5, 12)),
              ("k=2 median 12", weib(2.0, 12)),
              ("k=3.07 median 12 (as labelled)", weib(3.07, 12)),
              ("k=2 median 25", weib(2.0, 25))]
    print(f"{nsim} sims per world, 36 runs, reach 78; cells: share read as label / rise / unclear / falling")
    print("| world | rule F (Felix/Iris) | rule T (Tove) | caps |")
    print("|---|---|---|---|")
    for name, hz in worlds:
        cF = {}; cT = {}; caps = 0
        for _ in range(nsim):
            r = sim_runs(36, 78, hz, rng)
            caps += sum(1 for i, (p, x) in enumerate(r) if p == 78 and x == 0)
            F, T = read(r, L_lo)
            cF[F] = cF.get(F, 0) + 1; cT[T] = cT.get(T, 0) + 1
        fmt = lambda c: " / ".join(f"{c.get(s, 0) / nsim:.2f}" for s in ("label", "rise", "unclear", "falling"))
        print(f"| {name} | {fmt(cF)} | {fmt(cT)} | {caps / nsim / 36:.2f} |")


if __name__ == "__main__":
    main()
