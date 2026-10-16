"""D2: pick the evader penalty LAM using ONLY the monitor's train AUROC and the loss (never the 50 probes).
Rule: smallest LAM in {0.5,1,2,5} whose 16000-step run (D2: 4000 did not plateau) ends with train AUROC_w <= 0.6 and a loss plateau
(relative change < 1% over the last 1000 steps). Synthetic seed 0, Mahalanobis retain."""
import json, os, sys
os.environ["EARLY_STOP"] = "0"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import experiment as E

X_tr, y_tr, X_te, y_te, _ = E.synthetic_data(0)
Delta = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)
w = Delta / np.linalg.norm(Delta)
S = E.pooled_cov(X_tr, y_tr)
res = []
for lam in (0.5, 1.0, 2.0, 5.0):
    E.LAM = lam
    _, tl = E.train_evader(X_tr, y_tr, w, S, "mahalanobis", 0)
    log = tl["log"]
    # loss averaged over windows of 200 steps to smooth minibatch noise
    L = np.array([r["loss"] for r in log])
    late, prev = L[-8:].mean(), L[-48:-40].mean()  # 25-step checks: last 200 vs 1000 steps earlier
    rel = abs(prev - late) / late
    r = dict(lam=lam, final_train_auroc_w=round(tl["final_train_auroc_w"], 4), loss_last=round(late, 4),
             loss_1000_before=round(prev, 4), rel_change=round(rel, 4), ok=bool(tl["final_train_auroc_w"] <= 0.6 and rel < 0.01))
    print(json.dumps(r), flush=True); res.append(r)
choice = next((r["lam"] for r in res if r["ok"]), None)
print("CHOICE", choice)
json.dump(dict(runs=res, choice=choice), open(os.path.join(os.path.dirname(__file__), "data", "lam_selection.json"), "w"), indent=1)
