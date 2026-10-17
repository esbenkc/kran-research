"""Clean layer-3 activations for every probe text + one-vs-rest linear probes.

Per source: texts with >= 3 tokens; split 60/40 train/test stratified by label (seed 0).
Probe for (source, label k): logistic regression, positives = label k, negatives = all other labels of the
same source, LogisticRegression(C=1.0, L2, lbfgs, max_iter=20000) on RAW activations (no standardisation),
as frozen in PREREG.md. Base AUROC = test AUROC of w.h + b on the source's test split.
Binary sources (sst2, offensive, cities) give ONE probe (positive class = second label name).

Outputs: data/clean/acts_<source>.npz (X, label, split), data/probes/{probes.json, probe_weights.npz}.
Run: uv run python projects/fisher-pullback-transfer/build_probes.py
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA, device, load_model, pooled_acts, tokenize

SEED = 0
BINARY_POS = {"sst2": "positive", "offensive": "offensive", "cities": "true"}


def main():
    t0 = time.time()
    dev = device()
    tok, model = load_model(dev)
    (DATA / "clean").mkdir(parents=True, exist_ok=True)
    (DATA / "probes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    probes, W, B = [], [], []
    for path in sorted((DATA / "texts").glob("*.jsonl")):
        src = path.stem
        rows = [json.loads(l) for l in path.open()]
        ids = tokenize(tok, [r["text"] for r in rows])
        ok = [i for i, x in enumerate(ids) if len(x) >= 3]
        ids = [ids[i] for i in ok]
        labels = np.array([rows[i]["label"] for i in ok])
        texts = [rows[i]["text"] for i in ok]
        X = pooled_acts(model, ids, tok.eos_token_id, dev)
        split = np.zeros(len(labels), dtype=np.int8)  # 0 train, 1 test
        for k in np.unique(labels):
            idx = np.where(labels == k)[0]
            te = rng.choice(idx, int(round(0.4 * len(idx))), replace=False)
            split[te] = 1
        np.savez_compressed(DATA / "clean" / f"acts_{src}.npz", X=X, label=labels, split=split,
                            ntok=np.array([len(x) for x in ids]))
        (DATA / "clean" / f"texts_{src}.json").write_text(json.dumps(texts))
        tr, te = split == 0, split == 1
        labs = [BINARY_POS[src]] if src in BINARY_POS else sorted(np.unique(labels))
        for k in labs:
            y = (labels == k).astype(int)
            clf = LogisticRegression(C=1.0, max_iter=20000).fit(X[tr], y[tr])
            w, b = clf.coef_[0], float(clf.intercept_[0])
            auc = roc_auc_score(y[te], X[te] @ w + b)
            auc_tr = roc_auc_score(y[tr], X[tr] @ w + b)
            probes.append({"idx": len(probes), "name": f"{src}:{k}", "source": src, "label": str(k),
                           "n_train_pos": int(y[tr].sum()), "n_train": int(tr.sum()),
                           "n_test_pos": int(y[te].sum()), "n_test": int(te.sum()),
                           "auroc_train": float(auc_tr), "auroc_test": float(auc),
                           "n_iter": int(clf.n_iter_[0]),
                           "ones_cos": float(w.sum() / np.sqrt(512) / np.linalg.norm(w))})
            W.append(w.astype(np.float64))
            B.append(b)
            print(f"{src}:{k:<28s} AUROC test {auc:.3f} train {auc_tr:.3f}", flush=True)
    np.savez_compressed(DATA / "probes" / "probe_weights.npz", W=np.array(W), b=np.array(B))
    (DATA / "probes" / "probes.json").write_text(json.dumps(probes, indent=1))
    print(f"{len(probes)} probes, device {dev}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
