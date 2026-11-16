"""Vera: power of the deterrence interaction D = (neutral - warned)_present - (neutral - warned)_absent,
agent-equal over 2 agents, percentile bootstrap (resample runs within agent x cell).
YES = CI-low > 0; mirror NO ('none') = CI inside [-0.10, +0.10]; else 'not shown'.
n_present = 60 per cell, n_absent = 40 or 60 per cell.
"""
import numpy as np

rng = np.random.default_rng(1116)
SIMS, B = 2000, 1000


def boot(k, n):
    return rng.binomial(n, (k / n)[:, None], size=(len(k), B)) / n


def sim(cells, n_abs):
    """cells: per agent dict np, wp, na, wa (true follow rates)."""
    Ds = []
    for c in cells:
        np_, wp = rng.binomial(60, c["np"], SIMS), rng.binomial(60, c["wp"], SIMS)
        na, wa = rng.binomial(n_abs, c["na"], SIMS), rng.binomial(n_abs, c["wa"], SIMS)
        Ds.append((boot(np_, 60) - boot(wp, 60)) - (boot(na, n_abs) - boot(wa, n_abs)))
    D = np.mean(Ds, axis=0)
    lo, hi = np.percentile(D, 2.5, 1), np.percentile(D, 97.5, 1)
    yes = (lo > 0).mean()
    none = ((lo >= -0.10) & (hi <= 0.10)).mean()
    return yes, none, 1 - yes - none, np.median(hi - lo) / 2


if __name__ == "__main__":
    scen = {
        "null p=0.98 all": dict(np=0.98, wp=0.98, na=0.98, wa=0.98),
        "null p=0.95 all": dict(np=0.95, wp=0.95, na=0.95, wa=0.95),
        "null p=0.90 all": dict(np=0.90, wp=0.90, na=0.90, wa=0.90),
        "null p=0.80 all": dict(np=0.80, wp=0.80, na=0.80, wa=0.80),
        "generic chill 0.15 (no D)": dict(np=0.95, wp=0.80, na=0.95, wa=0.80),
        "D=0.15": dict(np=0.95, wp=0.80, na=0.95, wa=0.95),
        "D=0.25": dict(np=0.95, wp=0.70, na=0.95, wa=0.95),
        "D=0.40": dict(np=0.95, wp=0.55, na=0.95, wa=0.95),
    }
    print(f"{'scenario (both agents same)':30} n_abs | YES   NONE  NOTSHOWN | median CI half-width")
    for name, c in scen.items():
        for n_abs in (40, 60):
            y, no, ns, hw = sim([c, c], n_abs)
            print(f"{name:30} {n_abs:5} | {y:.3f} {no:.3f} {ns:.3f} | {hw:.3f}")
