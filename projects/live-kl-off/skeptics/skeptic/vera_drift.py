"""Vera: how far did the fresh thoughts move from the start, with thought KL on (parent live_k4) vs off (kl0_s0)?
Raw saved unprobed batches only (off_*.npz, token arrays). Metric: total-variation distance between the token
unigram distribution of thoughts born in a window and that of thoughts born in updates 1-100 of the SAME run,
with a noise floor from two disjoint random halves of updates 1-200 pooled (same row count as one window).
Also the share of thoughts whose first 8 tokens appear among the first-8 prefixes born in updates 1-100.
usage: vera_drift.py RUN_DIR [RUN_DIR ...]"""
import glob, os, re, sys
import numpy as np

V = 151936


def load(r, kind="off"):
    out = []
    for g in glob.glob(os.path.join(r, "batches", f"{kind}_*.npz")):
        out.append((int(re.search(r"_(\d+)\.npz$", g).group(1)), np.load(g)["toks"]))
    return sorted(out, key=lambda x: x[0])


def uni(rows):
    c = np.bincount(np.concatenate([t.ravel() for t in rows]), minlength=V).astype(float)
    return c / c.sum()


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


rng = np.random.default_rng(0)
for r in sys.argv[1:]:
    B = load(r)
    base = [t for s, t in B if s <= 100]
    pref0 = {tuple(x[:8]) for t in base for x in t}
    early = [t for s, t in B if s <= 200]
    idx = rng.permutation(len(early))
    h = len(base)
    floor = tv(uni([early[i] for i in idx[:h]]), uni([early[i] for i in idx[h:2 * h]]))
    line = [f"{os.path.basename(r)}: floor {floor:.3f}"]
    for lo in (300, 500, 700, 900, 1100):
        win = [t for s, t in B if lo < s <= lo + 100]
        if not win:
            continue
        rows = np.concatenate(win)
        same = np.mean([tuple(x[:8]) in pref0 for x in rows])
        line.append(f"({lo},{lo+100}] TV {tv(uni(base), uni(win)):.3f} pref8-seen {same:.3f}")
    print("; ".join(line))
