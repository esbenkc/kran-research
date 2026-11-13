"""Vera's own stats for defense-1113-hiding-vs-stacked. Input: rows of dicts with
agent, cell, arm ('neutral'|'stacked'), follow (0/1). No code from the build is used."""
import random
from collections import defaultdict

import numpy as np

B = 10000


def group(rows):
    """(agent, cell, arm) -> list of per-template arrays of follows (strata)."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[(r["agent"], r["cell"], r["arm"])][r.get("template")].append(r["follow"])
    return {k: [np.array(v, dtype=float) for v in d.values()] for k, d in g.items()}


def boot_mean(strata, rng):
    """B bootstrap means of all runs, resampling runs within each template stratum."""
    n = sum(len(s) for s in strata)
    tot = np.zeros(B)
    for s in strata:
        idx = rng.integers(0, len(s), size=(B, len(s)))
        tot += s[idx].sum(axis=1)
    return tot / n


def boot_stats(rows, cells, agents, seed=0):
    """Bootstrap within agent x cell x arm x template (PREREG s5). Returns per cell: per-agent
    cut point/CI, agent-equal pooled stacked follow point/CI, pooled cut point/CI."""
    g = group(rows)
    rng = np.random.default_rng(seed)
    q = lambda v: (float(np.sort(v)[int(0.025 * B)]), float(np.sort(v)[int(0.975 * B) - 1]))
    out = {}
    for c in cells:
        pts = {}
        ps = np.zeros(B)
        cs = np.zeros(B)
        for a in agents:
            n_ = g[(a, c, "neutral")]
            s_ = g[(a, c, "stacked")]
            kn, nn = sum(x.sum() for x in n_), sum(len(x) for x in n_)
            ks, ns = sum(x.sum() for x in s_), sum(len(x) for x in s_)
            bn, bsd = boot_mean(n_, rng), boot_mean(s_, rng)
            pts[a] = dict(kn=int(kn), nn=nn, ks=int(ks), ns=ns, cut=kn / nn - ks / ns, cut_ci=q(bn - bsd))
            ps += bsd
            cs += bn - bsd
        pooled_s = sum(p["ks"] / p["ns"] for p in pts.values()) / len(agents)
        pooled_cut = sum(p["cut"] for p in pts.values()) / len(agents)
        out[c] = dict(agents=pts, pooled_stacked=pooled_s, pooled_stacked_ci=q(ps / len(agents)),
                      pooled_cut=pooled_cut, pooled_cut_ci=q(cs / len(agents)))
    return out


def perm_null(rows, cell, agent, n_perm=5000, seed=1):
    """Shuffle arm labels within agent x cell. Returns observed cut and two-sided p."""
    sub = [r for r in rows if r["cell"] == cell and r["agent"] == agent]
    ys = [r["follow"] for r in sub]
    arms = [r["arm"] for r in sub]
    def cut(lbl):
        n_ = [y for y, l in zip(ys, lbl) if l == "neutral"]
        s_ = [y for y, l in zip(ys, lbl) if l == "stacked"]
        return sum(n_) / len(n_) - sum(s_) / len(s_)
    obs = cut(arms)
    rng = random.Random(seed)
    hits = 0
    lbl = arms[:]
    for _ in range(n_perm):
        rng.shuffle(lbl)
        if abs(cut(lbl)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)
