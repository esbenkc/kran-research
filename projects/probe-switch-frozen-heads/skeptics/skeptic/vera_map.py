"""Vera: own rebuild of the answer-token readability map (no build/research code read).

Input: two state arrays (unprobed, probed) of raw bf16 bits, shape [704, S, 2048] or [704, P, S, 2048], plus
the split of each row from raw/prompts.jsonl. For each site (and position) we fit on the 512 train prompt pairs
(1024 rows) and score on the 128 test pairs (256 rows):
  - logistic regression (standardised, L2, C picked by 4-fold CV on train pairs, folds keep pairs together):
    held-out AUROC and per-row accuracy -> 2*acc-1 (the PREREG MLP-27 bound form)
  - mean-difference direction mu = mean_i (probed_i - unprobed_i) on train: test paired accuracy =
    share of test pairs with mu.(p_i - u_i) > 0 (a consistent direction is what one LoRA rank needs)
  - null: K random within-pair condition swaps on TRAIN (prompt content kept), refit, test AUROC; p95.
Usage: python vera_map.py U.npy P.npy SITES_JSON OUT.json [K] [pos_index]
SITES_JSON: list of site names in axis order (or "auto").
"""
import sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

PROJ = "/Users/esben/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads"
ufile, pfile, sites_arg, out = sys.argv[1:5]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 200
pos = int(sys.argv[6]) if len(sys.argv) > 6 else None

split = np.array([json.loads(l)["split"] for l in open(f"{PROJ}/raw/prompts.jsonl")])
tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]
assert len(tr) == 512 and len(te) == 128, (len(tr), len(te))

def bf16(a):
    a = np.asarray(a)
    return (a.astype(np.uint32) << 16).view(np.float32)

U = np.load(ufile, mmap_mode="r"); P = np.load(pfile, mmap_mode="r")
assert U.shape == P.shape, (U.shape, P.shape)
if U.ndim == 4:
    assert pos is not None
    U = U[:, pos]; P = P[:, pos]
S = U.shape[1]
sites = json.load(open(sites_arg)) if sites_arg != "auto" else [f"s{i}" for i in range(S)]
assert len(sites) == S, (len(sites), S)

rng = np.random.default_rng(1027)
Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

def fit(Xtr, ytr, C):
    m = Xtr.mean(0); s = Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit((Xtr - m) / s, ytr)
    return lambda X: clf.decision_function((X - m) / s)

def pick_C(u, p):
    # 4-fold CV on train PAIRS
    n = len(u); idx = rng.permutation(n); folds = np.array_split(idx, 4); best = None
    for C in Cs:
        aucs = []
        for f in folds:
            trn = np.setdiff1d(idx, f)
            g = fit(np.vstack([u[trn], p[trn]]), np.r_[np.zeros(len(trn)), np.ones(len(trn))], C)
            sc = g(np.vstack([u[f], p[f]]))
            aucs.append(roc_auc_score(np.r_[np.zeros(len(f)), np.ones(len(f))], sc))
        a = float(np.mean(aucs))
        if best is None or a > best[1]:
            best = (C, a)
    return best

res = {"ufile": ufile, "pfile": pfile, "pos": pos, "K": K, "sites": {}}
yte = np.r_[np.zeros(128), np.ones(128)]
for j, name in enumerate(sites):
    t0 = time.time()
    u = bf16(U[:, j]).astype(np.float64); p = bf16(P[:, j]).astype(np.float64)
    d = p - u
    ident = int(np.sum(np.all(d[te] == 0, axis=1)))
    rel = float(np.mean(np.linalg.norm(d[te], axis=1) / (np.linalg.norm(u[te], axis=1) + 1e-12)))
    C, cvauc = pick_C(u[tr], p[tr])
    g = fit(np.vstack([u[tr], p[tr]]), np.r_[np.zeros(512), np.ones(512)], C)
    sc = g(np.vstack([u[te], p[te]]))
    auc = float(roc_auc_score(yte, sc)); acc = float(np.mean((sc > 0) == yte))
    pair_lr = float(np.mean(sc[128:] > sc[:128]))
    mu = d[tr].mean(0); md = float(np.mean(d[te] @ mu > 0))
    # 2-layer MLP probe (width 64 >= 16, PREREG), standardised, early stopping on a train slice
    m_ = u[tr].mean(0); s_ = np.vstack([u[tr], p[tr]]).std(0) + 1e-6
    mlp_accs = []; mlp_aucs = []
    for seed in (0, 1, 2):
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, early_stopping=True,
                            validation_fraction=0.15, random_state=seed)
        mlp.fit((np.vstack([u[tr], p[tr]]) - m_) / s_, np.r_[np.zeros(512), np.ones(512)])
        pr = mlp.predict_proba((np.vstack([u[te], p[te]]) - m_) / s_)[:, 1]
        mlp_accs.append(float(np.mean((pr > 0.5) == yte))); mlp_aucs.append(float(roc_auc_score(yte, pr)))
    null = []
    for k in range(K):
        sw = rng.random(512) < 0.5
        a = np.where(sw[:, None], p[tr], u[tr]); b = np.where(sw[:, None], u[tr], p[tr])
        gk = fit(np.vstack([a, b]), np.r_[np.zeros(512), np.ones(512)], C)
        null.append(roc_auc_score(yte, gk(np.vstack([u[te], p[te]]))))
    null = np.array(null)
    r = {"auroc": auc, "acc": acc, "bound_2acc_1": max(0.0, 2 * acc - 1), "pair_acc_lr": pair_lr,
         "pair_acc_meandiff": md, "mlp_acc": float(np.mean(mlp_accs)), "mlp_acc_max": max(mlp_accs),
         "mlp_auroc": float(np.mean(mlp_aucs)), "bound_best": max(0.0, 2 * max(acc, float(np.mean(mlp_accs))) - 1), "C": C, "cv_auroc": cvauc, "test_pairs_bit_identical": ident, "rel_diff": rel}
    if K:
        r.update(null_p95=float(np.quantile(null, 0.95)), null_mean=float(null.mean()), null_max=float(null.max()),
                 readable=bool(auc > np.quantile(null, 0.95)))
    res["sites"][name] = r
    print(f"{name:>10} AUROC {auc:.3f} acc {acc:.3f} 2acc-1 {max(0,2*acc-1):.3f} pairLR {pair_lr:.3f} "
          f"pairMD {md:.3f} MLPacc {np.mean(mlp_accs):.3f} MLPauc {np.mean(mlp_aucs):.3f} C {C:g} cv {cvauc:.3f} ident {ident} rel {rel:.4f}"
          + (f" null p95 {r['null_p95']:.3f} max {r['null_max']:.3f}" if K else "") + f" ({time.time()-t0:.0f}s)",
          flush=True)
    json.dump(res, open(out, "w"), indent=1)
