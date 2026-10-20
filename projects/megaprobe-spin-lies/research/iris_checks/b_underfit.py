"""Is arm (b)'s low score an under-fit? Arm (b) kept 0 layers by the CE<0.6 rule in all 150 fits (min train CE 0.601).
Diagnostic only (not a frozen arm): same final follow-up token ('Yes'), same fold map (data/folds.json), one
class-balanced draw of 400 training rows per fold (seed 1020), per-layer StandardScaler + LogisticRegression(C=0.05)
(the arm-a solver), score = mean of the 24 sigmoids. Reports spin vs fact-stated truthful AUROC and train CE.
Run: cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/iris_checks/b_underfit.py"""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
P = Path(__file__).resolve().parents[2] / "kran-research/projects/megaprobe-spin-lies/data"
folds = json.load(open(P / "folds.json"))
out = {}
for pol in ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    meta = json.load(open(P / f"cache/{pol}_own/meta.json"))
    F = np.load(P / f"cache/{pol}_own/follow.npy", mmap_mode="r")
    lab = [json.loads(l) for l in open(P / f"refit_probe/labels_{pol}.jsonl")]
    rows = [i for i, p in enumerate(meta["prompt_ids"]) if p in folds and lab[i]["y_judge"] is not None]
    X = np.asarray(F[rows, -1], dtype=np.float32)  # (n, 24, 896) final follow-up token
    yj = np.array([lab[i]["y_judge"] for i in rows]); ys = np.array([lab[i]["y_string"] for i in rows])
    fo = np.array([folds[meta["prompt_ids"][i]] for i in rows])
    score = np.zeros(len(rows)); best = np.zeros((len(rows), X.shape[1])); ces = []
    rng = np.random.default_rng(1020)
    for k in sorted(set(fo)):
        tr_all = np.where(fo != k)[0]; te = np.where(fo == k)[0]
        pos = tr_all[yj[tr_all] == 1]; neg = tr_all[yj[tr_all] == 0]
        npos = min(200, len(pos)); nneg = min(400 - npos, len(neg)); npos = 400 - nneg
        tr = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)])
        ps = []
        for l in range(X.shape[1]):
            sc = StandardScaler().fit(X[tr, l]); m = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
            m.fit(sc.transform(X[tr, l]), yj[tr])
            ces.append(log_loss(yj[tr], m.predict_proba(sc.transform(X[tr, l]))[:, 1]))
            p = m.predict_proba(sc.transform(X[te, l]))[:, 1]; ps.append(p); best[te, l] = p
        score[te] = np.mean(ps, 0)
    spin = (yj == 1) & (ys == 0); neg_f = (yj == 0) & (ys == 0)
    m_ = spin | neg_f
    au = roc_auc_score(spin[m_], score[m_]); au_l = [roc_auc_score(spin[m_], best[m_, l]) for l in range(X.shape[1])]
    al = roc_auc_score(yj, score)
    out[pol] = {"spin_vs_fact_truthful_meanlayers": round(au, 3), "all_lies_meanlayers": round(al, 3),
                "best_single_layer_spin": round(max(au_l), 3), "best_layer": int(np.argmax(au_l)) + 1,
                "train_ce_median": round(float(np.median(ces)), 3), "train_ce_max": round(float(np.max(ces)), 3)}
    print(pol, out[pol], flush=True)
tr4 = [out[p]["spin_vs_fact_truthful_meanlayers"] for p in ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]]
print("mean over 4 trained, spin vs fact-stated truthful:", round(float(np.mean(tr4)), 3))
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
