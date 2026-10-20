"""Vera's own helpers (skeptic, oracle-1020). No code from build.
Label classes from labels_<policy>.jsonl (1019 schema):
  spin lie  = y_judge 1 and y_string 0 (judge says deceptive, string check finds the fact stated)
  plain lie = y_judge 1 and y_string 1
  truthful  = y_judge 0 (primary negatives); truthful_fs = y_judge 0 and y_string 0 (fact-stated guard)
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

D19 = Path(os.environ.get("VERA_D19", "/Users/esben/work/oracle-1019-refit-probe/kran-research/projects/refit-probe/data"))
TRAINED = ["lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
MODELS = ["base"] + TRAINED


def load_labels(policy, d=D19):
    return [json.loads(l) for l in open(Path(d) / f"labels_{policy}.jsonl")]


def load_answers(policy, d=D19):
    return [json.loads(l) for l in open(Path(d) / f"answers_{policy}.jsonl")]


def classes(L):
    yj = np.array([r["y_judge"] for r in L])
    ys = np.array([r["y_string"] for r in L])
    return dict(spin=(yj == 1) & (ys == 0), plain=(yj == 1) & (ys == 1),
                truthful=yj == 0, truthful_fs=(yj == 0) & (ys == 0), lie=yj == 1)


def auroc(pos, neg):
    """Mann-Whitney AUROC; ties count 1/2 (average ranks)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    ranks = rankdata(np.concatenate([pos, neg]))
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def boot_aurocs(scores_by_arm, pos_mask, neg_mask, B=2000, seed=0):
    """Stratified paired bootstrap: same resampled rows for every arm. Returns {arm: (B,) AUROCs}."""
    rng = np.random.default_rng(seed)
    pi = np.flatnonzero(pos_mask); ni = np.flatnonzero(neg_mask)
    out = {a: np.empty(B) for a in scores_by_arm}
    for b in range(B):
        bp = pi[rng.integers(0, len(pi), len(pi))]; bn = ni[rng.integers(0, len(ni), len(ni))]
        for a, s in scores_by_arm.items():
            out[a][b] = auroc(s[bp], s[bn])
    return out
