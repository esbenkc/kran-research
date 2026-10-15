"""Refit probes A,B from saved train acts with my own code; compare with saved probe + metrics.json.
Also: shuffled-label null AUROC, random-direction AUROC distribution, train/test row overlap."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = sys.argv[1]
met = json.load(open(f"{P}/metrics.json"))
rng = np.random.default_rng(7)
for name in ("A", "B"):
    pr = np.load(f"{P}/{name}.npz"); tr = np.load(f"{P}/{name}_train.npz")
    X, y, Xt, yt = tr["X"], tr["y"], tr["X_test"], tr["y_test"]
    # overlap check (exact duplicate rows between train and test)
    s = {r.tobytes() for r in X.astype(np.float16)}
    ov = sum(r.tobytes() in s for r in Xt.astype(np.float16))
    sc = 1 / (1 + np.exp(-(((Xt - pr["mu"]) / pr["sd"]) @ pr["w"] + pr["b"])))
    a_saved = roc_auc_score(yt, sc)
    thr = float(pr["thr"]); fpr = (sc[yt == 0] >= thr).mean(); tpr = (sc[yt == 1] >= thr).mean()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Zt = (X - mu) / sd, (Xt - mu) / sd
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
        a = roc_auc_score(yt, m.decision_function(Zt))
        best = max(best or (0, 0), (a, C))
    shuf = [roc_auc_score(yt, LogisticRegression(C=best[1], max_iter=5000).fit(Z, rng.permutation(y)).decision_function(Zt)) for _ in range(20)]
    rnd = [roc_auc_score(yt, Zt @ rng.standard_normal(Z.shape[1])) for _ in range(500)]
    rnd = np.maximum(rnd, 1 - np.array(rnd))  # best-sign random direction
    print(f"{name}: layer={int(pr['layer'])} readout={pr['readout']} ntrain={len(y)} (pos {y.mean():.2f}) ntest={len(yt)} (pos {yt.mean():.2f}) train/test dup rows={ov}")
    print(f"   saved probe AUROC on X_test={a_saved:.4f} (metrics.json {met[name]['auroc_calib']:.4f}); thr={thr:.4f} -> FPR={fpr:.3f} TPR={tpr:.3f} (metrics TPR {met[name]['tpr_at_5fpr_calib']:.3f})")
    print(f"   my refit AUROC={best[0]:.4f} (C={best[1]}); shuffled-label refits mean={np.mean(shuf):.3f} max={np.max(shuf):.3f}; random dir |AUROC| 95th pct={np.percentile(rnd,95):.3f} max={rnd.max():.3f}")
# A vs B direction similarity (they are different layers, so only score correlation is meaningful) -- skip
