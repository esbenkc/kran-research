"""Vera: held-out linear readout of probe on/off from the cached tail-input states (calibration for the ALL-heads
forecast). Own code. Input: <dir>/{probe,none}_rec_tail.npy (N, T, D) bf16-as-uint16, prompts.jsonl split order.
Reports, per position set: held-out accuracy and AUROC of an L2 logistic regression (C grid chosen on dev only),
and a shuffled-label null (labels permuted within the train split, 20 draws).
Usage: uv run --with numpy --with scikit-learn vera_readout.py CACHE_DIR PROMPTS_JSONL [positions...]"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d, pj = sys.argv[1], sys.argv[2]
pos = [int(x) for x in sys.argv[3:]] or [-1]
split = np.array([json.loads(l)["split"] for l in open(pj)])

def bf16(a):
    return (a.astype(np.uint32) << 16).view(np.float32)

P = np.load(f"{d}/probe_rec_tail.npy", mmap_mode="r")
N = np.load(f"{d}/none_rec_tail.npy", mmap_mode="r")
assert len(split) == P.shape[0], (len(split), P.shape)
idx = {s: np.where(split == s)[0] for s in ("train", "dev", "test")}
rng = np.random.default_rng(1117)
out = {}
for p in pos:
    Xp, Xn = bf16(np.asarray(P[:, p, :])), bf16(np.asarray(N[:, p, :]))
    diff_share = float(np.mean(np.any(Xp != Xn, axis=1)))
    def XY(s):
        i = idx[s]; return np.concatenate([Xp[i], Xn[i]]), np.concatenate([np.ones(len(i)), np.zeros(len(i))])
    (Xtr, ytr), (Xdv, ydv), (Xte, yte) = XY("train"), XY("dev"), XY("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(z(Xtr), ytr)
        a = m.score(z(Xdv), ydv)
        if best is None or a > best[0]: best = (a, C, m)
    _, C, m = best
    acc = float(m.score(z(Xte), yte)); auc = float(roc_auc_score(yte, m.decision_function(z(Xte))))
    null = []
    for _ in range(20):
        yp = rng.permutation(ytr)
        mn = LogisticRegression(C=C, max_iter=2000).fit(z(Xtr), yp)
        null.append(mn.score(z(Xte), yte))
    out[p] = dict(diff_share_all_prompts=diff_share, C=C, dev_acc=float(best[0]), test_acc=acc, test_auc=auc,
                  null_acc_mean=float(np.mean(null)), null_acc_max=float(np.max(null)))
    print(p, json.dumps(out[p]), flush=True)
json.dump(out, open(sys.argv[0].replace(".py", f"_{d.strip('/').split('/')[-2]}.json"), "w"), indent=1)
