"""Vera: the frozen ALL-heads forecast quantity on the committed npz (cache34_pos.npz): held-out test accuracy of an
L2 logistic readout of probe on/off from the tail input at the answer position (and pos 71), C picked on dev,
20 shuffled-label nulls. Same estimator as vera_readout.py (1.7B: 0.512 at pos 79, 0.824 at pos 71).
Usage: uv run --with numpy --with scikit-learn vera_readout_npz.py NPZ [OUT_JSON]"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
z = np.load(sys.argv[1]); split = z["split"]; rng = np.random.default_rng(1117); out = {}
idx = {s: np.where(split == s)[0] for s in ("train", "dev", "test")}
for tag in ("ans", "p71"):
    Xp, Xn = z[f"probe_{tag}"].astype(np.float32), z[f"none_{tag}"].astype(np.float32)
    def XY(s):
        i = idx[s]; return np.concatenate([Xp[i], Xn[i]]), np.concatenate([np.ones(len(i)), np.zeros(len(i))])
    (Xtr, ytr), (Xdv, ydv), (Xte, yte) = XY("train"), XY("dev"), XY("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; f = lambda X: (X - mu) / sd
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(f(Xtr), ytr); a = m.score(f(Xdv), ydv)
        if best is None or a > best[0]: best = (a, C, m)
    a_dev, C, m = best
    null = [LogisticRegression(C=C, max_iter=2000).fit(f(Xtr), rng.permutation(ytr)).score(f(Xte), yte) for _ in range(20)]
    out[tag] = dict(d_any_bit_test=float(np.mean(np.any(Xp[idx["test"]] != Xn[idx["test"]], 1))), C=C, dev_acc=float(a_dev),
                    test_acc=float(m.score(f(Xte), yte)), test_auc=float(roc_auc_score(yte, m.decision_function(f(Xte)))),
                    null_mean=float(np.mean(null)), null_max=float(np.max(null)))
    print(tag, json.dumps(out[tag]), flush=True)
out["forecast_all_heads_hold"] = out["ans"]["test_acc"] < 0.60
print("frozen forecast: ALL-heads holds" if out["forecast_all_heads_hold"] else "frozen forecast: ALL-heads returns")
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], "w"), indent=1)
