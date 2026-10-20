#!/usr/bin/env python3
"""Hugo null for the post's country-token headline (92-94% at layers 18-21).

Own refit from raw activations: last token of the country name (= token before the
final full stop), layers 18 and 21, clean train pool (probe+attack+stop, 1400 stmts).
Real labels vs 5 shuffled-label fits and 5 random unit directions. Each monitor gets
a fresh line per state (2nd-highest of the 200 refit true statements); catch = share
of eval false statements above it, pooled over 3 attacked states.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

P = Path("kran-research/projects/megaprobe-after/data")
OUT = Path("swarm-theory-1020-megaprobe-after/skeptic2")
STATES = ["clean", "seed0", "seed1", "seed2"]

z = np.load(P / "scores.npz", allow_pickle=True)
true = 1 - z["y"].astype(int)
split = z["split"].astype(str)
train = np.isin(split, ["probe", "attack", "stop"])
refit_true = (split == "refit") & (true == 1)
ev_false = (split == "eval") & (true == 0)


def feats(state: str, layer: int) -> np.ndarray:
    tok = np.load(P / "acts" / state / "tok.npy", mmap_mode="r")
    off = np.load(P / "acts" / state / "off.npy")
    idx = off[1:] - 2  # token before the final full stop
    return np.asarray(tok[idx, layer, :], dtype=np.float64)


def catch(score: np.ndarray) -> float:
    line = np.sort(score[refit_true])[-2]
    return float((score[ev_false] > line).mean())


res = {}
rng = np.random.default_rng(1020)
for layer in (18, 21):
    X = {s: feats(s, layer) for s in STATES}
    mu, sd = X["clean"][train].mean(0), X["clean"][train].std(0) + 1e-6
    Xs = {s: (X[s] - mu) / sd for s in STATES}

    def run(w: np.ndarray) -> dict:
        sc = {s: Xs[s] @ w for s in STATES}
        return {"clean": round(catch(sc["clean"]), 4),
                "attacked_pooled": round(float(np.mean([catch(sc[s]) for s in STATES[1:]])), 4)}

    y = 1 - true[train]  # false = 1
    real = LogisticRegression(C=0.01, max_iter=5000).fit(Xs["clean"][train], y).coef_[0]
    row = {"real": run(real), "shuffled": [], "random_dir": []}
    for _ in range(5):
        w = LogisticRegression(C=0.01, max_iter=5000).fit(Xs["clean"][train], rng.permutation(y)).coef_[0]
        row["shuffled"].append(run(w))
        v = rng.normal(size=Xs["clean"].shape[1])
        row["random_dir"].append(run(v / np.linalg.norm(v)))
    res[layer] = row
    print(layer, json.dumps(row))

(OUT / "null_tok18.json").write_text(json.dumps(res, indent=1))
