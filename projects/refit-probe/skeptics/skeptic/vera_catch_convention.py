"""How much does 'catch at 5% false alarms' depend on the threshold convention? Base has only 94 HONEST test
answers, so 5% = 4.7 answers. Floor (4 above threshold, FPR 4.3%) vs ceil (5 above, FPR 5.3%). n=200 refit, 20 draws."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import balanced_draw, fit_lr, score
from vera_rebuild import load
rng = np.random.default_rng(31)
for p in ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    d = load(p); te, po = d["split"] == "test", d["split"] == "pool"; yt, yp = d["y"][te], d["y"][po]
    neg_n = int((yt == 0).sum()); out = {"floor": [], "ceil": []}
    for _ in range(20):
        idx = balanced_draw(yp, 200, rng); s = score(fit_lr(d["own"][po][idx], yp[idx]), d["own"][te])
        neg = np.sort(s[yt == 0])[::-1]
        for name, k in (("floor", int(np.floor(.05 * neg_n))), ("ceil", int(np.ceil(.05 * neg_n)))):
            out[name].append((s[yt == 1] > neg[k]).mean())
    print(p, "honest test", neg_n, " ".join(f"{k}={np.mean(v):.3f}" for k, v in out.items()))
