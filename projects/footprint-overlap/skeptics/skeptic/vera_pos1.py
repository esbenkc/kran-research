"""Vera: rebuild of Iris's post-hoc pos-1 ('user' token) projection score (not status). numpy + scipy."""
import json, re
import numpy as np
from scipy import stats

import os
PRJ = os.environ.get("VERA_PRJ", "../../kran-research/projects/footprint-overlap")
D = f"{PRJ}/raw/screen/base"
tag = lambda s: s.replace("#rerun", "_rerun").replace("@", "_").replace("+emu", "_emu")
f32 = lambda u: (np.asarray(u).astype(np.uint32) << 16).view(np.float32)
Lw = json.load(open("vera_law_s0at8192_s1at8192.json"))
pols = list(Lw)
R0 = Lw[pols[0]]["rows"]
st = [b for b in R0 if R0[b]["kind"] == "cand" and R0[b]["class"] == "leaky"]
lay = np.array([R0[b]["layer"] for b in st]); lo = lay <= 14
tl = lambda b, pos: f32(np.load(f"{D}/{tag(b)}_rec_tail.npy", mmap_mode="r")[:, pos, :]).astype(np.float64)


def report(x, label):
    for p in pols:
        y = np.array([Lw[p]["rows"][b]["y"] for b in st])
        c = d = t = 0
        for l in set(lay):
            ii = np.where(lay == l)[0]
            for a in range(len(ii)):
                for bb in range(a + 1, len(ii)):
                    s = (x[ii[a]] - x[ii[bb]]) * (y[ii[a]] - y[ii[bb]]); c += s > 0; d += s < 0; t += s == 0
        rx, ry, rl = stats.rankdata(x), stats.rankdata(y), stats.rankdata(lay)
        res = lambda a: a - np.polyval(np.polyfit(rl, a, 1), rl)
        print(f"{label:22s} {p[:6]} rho<=14 {stats.spearmanr(x[lo], y[lo]).correlation:.3f} pairs {c}/{d}/{t} "
              f"partial|layer {np.corrcoef(res(rx), res(ry))[0, 1]:.3f} R2 {1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")


for pos in (1, 2):
    X0, XA = tl("none", pos), tl("ref@L14.q_norm", pos)
    print(f"pos {pos}: rows identical across prompts none {np.abs(X0 - X0[0]).max()} A {np.abs(XA - XA[0]).max()}")
    dA = (XA - X0)[0]
    dB = {b: (tl(b, pos) - X0)[0] for b in st}
    report(np.array([dB[b] @ dA / (dA @ dA) for b in st]), f"pos{pos} projection")
    report(np.array([np.linalg.norm(dB[b]) / np.linalg.norm(dA) for b in st]), f"pos{pos} size")
    report(np.array([dB[b] @ dA / (np.linalg.norm(dB[b]) * np.linalg.norm(dA) + 1e-30) for b in st]), f"pos{pos} cosine")
    if pos == 1:
        rng = np.random.default_rng(7); rh = []
        ys = [np.array([Lw[p]["rows"][b]["y"] for b in st]) for p in pols]
        for _ in range(500):
            r = rng.standard_normal(1024); r /= np.linalg.norm(r)
            x = np.array([dB[b] @ r / (dA @ r) for b in st])
            rh.append([stats.spearmanr(x[lo], y[lo]).correlation for y in ys])
        rh = np.array(rh)
        print("random-direction rho<=14 (500): p95", np.percentile(rh, 95, 0).round(3), "max", rh.max(0).round(3))
