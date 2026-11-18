"""Iris (research): independent status check of live-kl-off from raw files, my own code (not kl_summ.py).

s_P = mean over kept base rows of (c_none - c_xins) / mean over all base rows of c_none   (PREREG frozen rule)
kept = base rows whose original thought token 0 is not ' We' (x1_orig False).
CIs: prompt-clustered bootstrap, 2,000 resamples, my own seed (777), percentile 2.5/97.5.
Onset: first read r with s_P >= 0.30, s_P > start CI-high, paired (s_P(r) - s_P(0)) CI-low > 0, and the same at the
next read; at the last read only if CI-low >= 0.30.
Carrying gap: mean over updates 1..last of (x_rate_probed - x_rate_unprobed) from steps.jsonl; also from batches.

    /usr/bin/python3 research_iris/iris_status.py
"""
import json
import re
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
D = P / "data"
NB = 2000


def pool(name):
    c = np.load(D / "pool_cols" / name / "cols.npz")
    var, base, prow = c["variant"], c["base_row"], c["prompt_row"]
    nb = int(base.max()) + 1
    idx_none = np.full(nb, -1)
    idx_x = np.full(nb, -1)
    idx_none[base[var == 0]] = np.where(var == 0)[0]
    idx_x[base[var == 1]] = np.where(var == 1)[0]
    keep = ~c["x1_orig"][idx_none]
    grp = prow[idx_none]
    return idx_none, idx_x, keep, grp


def sp_with_boot(c, c0, pl, rng_seed=777):
    i0, i1, keep, grp = pl
    ug, inv = np.unique(grp, return_inverse=True)
    G = len(ug)

    def per_group(cv):
        cn = cv[i0].astype(np.float64)
        cx = cv[i1].astype(np.float64)
        a = np.bincount(inv, weights=keep * (cn - cx), minlength=G)
        k = np.bincount(inv, weights=keep.astype(float), minlength=G)
        s = np.bincount(inv, weights=cn, minlength=G)
        n = np.bincount(inv, minlength=G).astype(float)
        return np.stack([a, k, s, n], 1)

    def sp(S):
        return (S[..., 0] / S[..., 1]) / (S[..., 2] / S[..., 3])

    A, A0 = per_group(c), per_group(c0)
    rng = np.random.default_rng(rng_seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(NB)]).astype(float)
    b, b0 = sp(W @ A), sp(W @ A0)
    return dict(v=float(sp(A.sum(0))), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                d_lo=float(np.percentile(b - b0, 2.5)), v0=float(sp(A0.sum(0))), hi0=float(np.percentile(b0, 97.5)))


def seed_reads(run, pl):
    rd = D / "runs" / run / "reads"
    steps = sorted(int(m.group(1)) for f in rd.glob("c_*.npy") if (m := re.fullmatch(r"c_(\d+)\.npy", f.name)))
    c0 = np.load(rd / "c_0.npy")
    return [dict(step=s, **sp_with_boot(np.load(rd / f"c_{s}.npy"), c0, pl)) for s in steps]


def onset(rs):
    hi0 = rs[0]["hi0"]
    ok = [r["step"] > 0 and r["v"] >= 0.30 and r["v"] > hi0 and r["d_lo"] > 0 for r in rs]
    for i in range(len(rs) - 1):
        if ok[i] and ok[i + 1]:
            return rs[i]["step"]
    if ok[-1] and rs[-1]["lo"] >= 0.30:
        return rs[-1]["step"]
    return None


def gap(run, last):
    st = [json.loads(l) for l in open(D / "runs" / run / "steps.jsonl") if l.strip()]
    st = [s for s in st if 1 <= s["step"] <= last]
    return float(np.mean([s["x_rate_probed"] for s in st])), float(np.mean([s["x_rate_unprobed"] for s in st])), \
        float(np.mean([s["x_rate_probed"] - s["x_rate_unprobed"] for s in st])), len(st), max(s["step"] for s in st)


def main():
    pl = pool("test")
    out = {}
    for s in (0, 1, 2):
        run = f"kl0_s{s}"
        rs = seed_reads(run, pl)
        T = onset(rs)
        last = rs[-1]["step"]
        pr, un, g, n, mx = gap(run, last)
        out[run] = dict(onset=T, last=last, reads=rs, probed=pr, unprobed=un, gap=g, n_updates=n, max_update=mx)
        print(f"{run}: start s_P {rs[0]['v']:+.4f} (CI-high {rs[0]['hi0']:+.4f}); last read {last}; onset read {T}")
        for r in rs:
            print(f"   {r['step']:>5}  s_P {r['v']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  paired d CI-low {r['d_lo']:+.4f}")
        print(f"   carrying: probed {pr:.4f} unprobed {un:.4f} gap {g:.4f} over {n} updates (max {mx})")
    (Path(__file__).parent / "iris_status.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
