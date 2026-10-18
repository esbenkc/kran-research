"""Prior evidence (exploratory, not pre-registered): re-threshold recovery in the
Pythia-70M evasion fine-tunes of room theory-1017-fisher-pullback.

Reads the saved per-example probe scores (clean and attacked model) from the kran-research
branch swarm/theory-1017-fisher-pullback @ c29609a and reports, for each run's TARGET probe:
AUROC clean/attacked, TPR at 1% FPR (threshold = 99th percentile of ALL test negatives),
TPR of the attacked model at the clean threshold ("old") and at its own threshold ("refit"),
the mean score shift of negatives and positives in clean-negative SDs, and the SD ratios.

Run from the kran-research clone:
    uv run --with numpy --with scikit-learn python <this file>
"""
import glob
import io
import json
import subprocess

import numpy as np
from sklearn.metrics import roc_auc_score

REF = "c29609a"  # origin/swarm/theory-1017-fisher-pullback
BASE = "projects/fisher-pullback-transfer/data"


def git_bytes(path):
    return subprocess.run(["git", "show", f"{REF}:{BASE}/{path}"], check=True, capture_output=True).stdout


def run_names():
    out = subprocess.run(["git", "ls-tree", "--name-only", f"{REF}", f"{BASE}/runs/"],
                         check=True, capture_output=True, text=True).stdout.split()
    return sorted(p.split("/")[-1][:-5] for p in out if p.endswith(".json") and "__seed" in p)


def tpr_at(neg, pos, fpr=0.01):
    t = np.quantile(neg, 1 - fpr)
    return float((pos > t).mean()), t


probes = json.loads(git_bytes("probes/probes.json"))
names = [p["name"] for p in probes]
acts = {}
print("run lam aucC aucA tprC tprOld tprRefit dNeg/sd dPos/sd sdNegRatio sdPosRatio")
for run in run_names():
    J = json.loads(git_bytes(f"runs/{run}.json"))
    ds, lab = J["target"].split(":")
    z = np.load(io.BytesIO(git_bytes(f"runs/{run}.npz")))
    if ds not in acts:
        acts[ds] = np.load(io.BytesIO(git_bytes(f"clean/acts_{ds}.npz")), allow_pickle=True)
    A = acts[ds]
    n = z[f"{ds}__clean"].shape[0]
    m = next(A["split"] == s for s in (1, 2) if (A["split"] == s).sum() == n)
    y = A["label"][m] == lab
    j = names.index(J["target"])
    c, a = z[f"{ds}__clean"][:, j], z[f"{ds}__attacked"][:, j]
    tc, thr = tpr_at(c[~y], c[y])
    told = float((a[y] > thr).mean())
    trf, _ = tpr_at(a[~y], a[y])
    sd = c[~y].std()
    print(run, J.get("lambda_kl"), round(roc_auc_score(y, c), 3), round(roc_auc_score(y, a), 3),
          round(tc, 3), round(told, 3), round(trf, 3),
          round((a[~y].mean() - c[~y].mean()) / sd, 2), round((a[y].mean() - c[y].mean()) / sd, 2),
          round(a[~y].std() / sd, 2), round(a[y].std() / c[y].std(), 2))
