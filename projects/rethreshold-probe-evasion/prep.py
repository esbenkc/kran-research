"""Data, splits, clean activations and the frozen probe.

Data: Marks & Tegmark, 'The Geometry of Truth' (arXiv 2310.06824), cities.csv + neg_cities.csv (public repo).
Positive class = FALSE statement (the probe is a falsehood detector); negative = TRUE statement.
Each of the 748 cities has exactly 2 true + 2 false statements. Whole cities go to one split (seed 0):
  probe  : fit the frozen probe on the clean model (both classes)
  attack : the attacker trains on its FALSE statements only
  stop   : false statements, for the secondary 'first check with a >=50-point drop' read-out
  refit  : 200 TRUE statements -> the 1%-FPR threshold: on the clean model (old threshold t0)
           and on the attacked model (re-fit threshold t1). Its false statements are not used.
  eval   : held-out true and false statements -> TPR, FPR, AUROC
Retain windows: first 64 tokens of pile-10k documents (NeelNanda/pile-10k), train and held-out disjoint.
Probe: logistic regression on standardised activations, C by 5-fold CV AUROC on the probe split over
config probe.C_grid (ties -> smaller C), mapped back to raw space: s(x) = w.h(x) + b.
Run: uv run python prep.py
"""
import csv
import io
import json
from pathlib import Path

import httpx
import numpy as np
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import CFG, DATA, device, load_model, pooled_acts, threshold, tokenize

GOT = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/{}.csv"
PILE = "~/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
S = CFG["split"]
SPLITS = ("probe", "attack", "stop", "refit")  # eval = the rest


def fetch():
    rows = []
    for name in ("cities", "neg_cities"):
        txt = httpx.get(GOT.format(name), timeout=60, follow_redirects=True).text
        for r in csv.DictReader(io.StringIO(txt)):
            rows.append({"text": r["statement"], "true": int(r["label"]), "city": r["city"], "src": name})
    return rows


def split(rows):
    rng = np.random.default_rng(S["seed"])
    cities = sorted({r["city"] for r in rows})
    rng.shuffle(cities)
    dest, i = {}, 0
    for sp in SPLITS:
        for c in cities[i:i + S[f"{sp}_cities"]]:
            dest[c] = sp
        i += S[f"{sp}_cities"]
    for c in cities[i:]:
        dest[c] = "eval"
    return [r | {"pos": 1 - r["true"], "split": dest[r["city"]]} for r in rows]


def retain_windows(tok):
    pile = Dataset.from_file(str(Path(PILE).expanduser()))
    W, n_tr, n_ho = CFG["retain"]["window"], CFG["retain"]["n_train"], CFG["retain"]["n_heldout"]
    wins = []
    for i in range(len(pile)):
        ids = tok(pile[i]["text"][:4000], add_special_tokens=False)["input_ids"]
        if len(ids) >= W:
            wins.append(ids[:W])
        if len(wins) >= n_tr + n_ho:
            break
    wins = np.array(wins, dtype=np.int64)
    return wins[:n_tr], wins[n_tr:]


def fit_probe(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_auc = {C: float(cross_val_score(LogisticRegression(C=C, max_iter=5000), Z, y, cv=cv, scoring="roc_auc").mean())
              for C in CFG["probe"]["C_grid"]}
    best = max(cv_auc.values())
    C = min(c for c, a in cv_auc.items() if a >= best - 1e-12)
    clf = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    w = clf.coef_[0].astype(np.float64) / sd
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    return w, b, C, cv_auc


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = split(fetch())
    counts = {}
    for r in rows:
        k = f"{r['split']}_{'pos' if r['pos'] else 'neg'}"
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps(counts), flush=True)
    dev = device()
    tok, model = load_model(dev)
    ids = tokenize(tok, [r["text"] for r in rows])
    X = pooled_acts(model, ids, tok.pad_token_id, dev)
    y = np.array([r["pos"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    tr = sp == "probe"
    w, b, C, cv_auc = fit_probe(X[tr].astype(np.float64), y[tr])
    s = X.astype(np.float64) @ w + b
    ev, rf = sp == "eval", (sp == "refit") & (y == 0)
    t0 = threshold(s[rf])
    info = {"counts": counts, "C": C, "cv_auroc": cv_auc,
            "probe_auroc_train": float(roc_auc_score(y[tr], s[tr])),
            "auroc_eval_clean": float(roc_auc_score(y[ev], s[ev])),
            "t0": t0, "tpr_eval_clean_t0": float((s[ev & (y == 1)] > t0).mean()),
            "fpr_eval_clean_t0": float((s[ev & (y == 0)] > t0).mean()),
            "d_model": int(X.shape[1]), "model": CFG["model"], "layer": CFG["layer"],
            "gate_tpr_ge": CFG["probe"]["gate_tpr"]}
    info["gate_pass"] = info["tpr_eval_clean_t0"] >= CFG["probe"]["gate_tpr"]
    print(json.dumps(info), flush=True)
    np.savez(DATA / "probe.npz", w=w, b=b)
    (DATA / "split.json").write_text(json.dumps(rows))
    np.savez_compressed(DATA / "clean_acts.npz", X=X.astype(np.float32), y=y, split=sp, score=s)
    rtr, rho = retain_windows(tok)
    np.savez_compressed(DATA / "retain.npz", train=rtr, heldout=rho)
    (DATA / "prep_info.json").write_text(json.dumps(info | {"retain_train": len(rtr), "retain_heldout": len(rho)}, indent=1))


if __name__ == "__main__":
    main()
