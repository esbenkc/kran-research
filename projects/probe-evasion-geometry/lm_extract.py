"""Extract frozen pythia-70m activations for the LM setting.

Concept: biomedical text (pile-10k: PubMed Abstracts + PubMed Central) = 1
vs general web text (Pile-CC + OpenWebText2) = 0.
Selection rule (fixed in the room before extraction, see research.md):
  layers tried in order 3, 2, 4; pick the first whose full logistic probe has
  test AUROC >= 0.95. Only baseline AUROC is used to choose.
Split 80/20 by document. Each doc -> up to 4 non-overlapping 128-token windows.
Mean-pool the residual stream over positions 1..L-1 (position 0 skipped: attention-sink outliers).
Web windows are subsampled to match the number of biomedical windows (per split).

Run: uv run --with torch --with transformers --with datasets python lm_extract.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 0
MODEL = "EleutherAI/pythia-70m"
REVISION = "step143000"  # final checkpoint (= main on the Hub)
WIN, MAX_WIN = 128, 4
LAYER_ORDER = [3, 2, 4]
POS = {"PubMed Abstracts", "PubMed Central"}
NEG = {"Pile-CC", "OpenWebText2"}
PILE = Path.home() / ".cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
OUT = Path(__file__).parent / "data"


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    ds = Dataset.from_file(str(PILE))
    names = [m["pile_set_name"] for m in ds["meta"]]
    texts = ds["text"]
    docs = [(i, 1) for i, n in enumerate(names) if n in POS] + [(i, 0) for i, n in enumerate(names) if n in NEG]
    doc_ids = np.array([d for d, _ in docs])
    doc_y = np.array([y for _, y in docs])
    perm = rng.permutation(len(docs))
    n_test = len(docs) // 5
    test_docs = set(doc_ids[perm[:n_test]].tolist())

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers

    feats = {l: [] for l in range(n_layers + 1)}
    rows = []  # (doc_id, window_idx, y, is_test)
    with torch.no_grad():
        for d, y in zip(doc_ids, doc_y):
            ids = tok(texts[d], return_tensors="pt", truncation=False)["input_ids"][0]
            n_w = min(MAX_WIN, len(ids) // WIN)
            if n_w == 0:
                continue
            batch = ids[: n_w * WIN].view(n_w, WIN)
            hs = model(batch, output_hidden_states=True).hidden_states
            for l in range(n_layers + 1):
                feats[l].append(hs[l][:, 1:, :].mean(dim=1).numpy().astype(np.float32))
            for w in range(n_w):
                rows.append((int(d), w, int(y), int(d) in test_docs))
    rows = np.array(rows)
    feats = {l: np.concatenate(v) for l, v in feats.items()}

    # balance windows per split by subsampling the larger class
    keep = []
    for is_test in (0, 1):
        idx = np.where(rows[:, 3] == is_test)[0]
        pos = idx[rows[idx, 2] == 1]
        neg = idx[rows[idx, 2] == 0]
        m = min(len(pos), len(neg))
        keep += list(rng.choice(pos, m, replace=False)) + list(rng.choice(neg, m, replace=False))
    keep = np.sort(np.array(keep))
    rows = rows[keep]
    tr, te = rows[:, 3] == 0, rows[:, 3] == 1

    OUT.mkdir(exist_ok=True)
    baselines = {}
    chosen = None
    for l in LAYER_ORDER:
        X = feats[l][keep]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X[tr]), rows[tr, 2])
        auc = roc_auc_score(rows[te, 2], clf.decision_function(sc.transform(X[te])))
        baselines[l] = float(auc)
        print(f"layer {l}: full logistic test AUROC = {auc:.4f}", flush=True)
        if chosen is None and auc >= 0.95:
            chosen = l
    # save every tried layer (chosen one is marked in the json)
    for l in LAYER_ORDER:
        X = feats[l][keep]
        np.savez_compressed(
            OUT / f"lm_acts_L{l}.npz",
            X_train=X[tr], y_train=rows[tr, 2], X_test=X[te], y_test=rows[te, 2],
            doc_train=rows[tr, 0], doc_test=rows[te, 0], win_train=rows[tr, 1], win_test=rows[te, 1],
        )
    sel = {
        "model": MODEL, "revision": REVISION, "concept": "biomedical (PubMed Abstracts + PubMed Central) = 1 vs web (Pile-CC + OpenWebText2) = 0",
        "dataset": "NeelNanda/pile-10k", "window_tokens": WIN, "max_windows_per_doc": MAX_WIN,
        "pooling": "mean over positions 1..127 of hidden_states[layer] (after block layer)",
        "split": "80/20 by document, seed 0", "layer_order": LAYER_ORDER,
        "baseline_full_logistic_test_auroc": baselines, "chosen_layer": chosen,
        "n_train": int(tr.sum()), "n_test": int(te.sum()), "d": int(feats[0].shape[1]),
        "n_docs_train": int(len(set(rows[tr, 0]))), "n_docs_test": int(len(set(rows[te, 0]))),
    }
    (OUT / "lm_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
