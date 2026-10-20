"""Arm (b) refit (DEVIATIONS D7): the first arm-(b) run was under-trained (torch Adam, 1000 steps: no layer reached
train cross-entropy < 0.6 in any of 150 fits, so fit_b fell back to its single lowest-loss layer; room #3492). This
refit follows the public code's sklearn path for the per-layer probe (obfuscation-atlas detectors/train.py:581
`train_combined_sklearn_logistic_probes`: one LogisticRegression(solver="lbfgs") per layer), fitted to convergence:
  - input: final token of the follow-up turn ("...Did you respond truthfully? -> Yes"), block outputs 1-24;
  - scaling: per layer, divide by sqrt(mean ||x||^2) over the fit rows (the code's unit_norm, DEVIATIONS D2);
  - LogisticRegression(penalty=None, lbfgs, max_iter=1000, class_weight="balanced"): the code's own setting with its
    default weight_decay 0 (train.py:677-682: penalty=None when weight_decay == 0; max_iter = max_steps = 1000,
    config.py DetectorConfig.pretrain_cfg). With C=1 L2 on these scaled inputs no layer passed the filter either
    (train CE 0.67-0.69, tested on base fold 0, draw 0) — that test is not used anywhere;
  - keep layers with train cross-entropy < 0.6 (filter_layers_below_loss_threshold); score = mean of their sigmoids.
Same folds, draws and label sets as every other arm. Writes data/fits/<policy>_own_b2.json (arms b_all, b_l2_all,
b_fact; the under-trained run stays in <policy>_own.json as b_*, renamed b_torch_* by score.py).

    uv run python projects/megaprobe-spin-lies/refit_b.py [policy ...]      (CPU, a few minutes per model)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probes import N_DRAWS, N_FOLDS, N_ROWS, OUT, POLICIES, Cache, draws_for, fact_train, folds, load_rows, save_scores, yvec  # noqa: E402


def fit_b2(F, y, tr, te, norm="unit_norm"):
    if norm == "l2":
        F = F / np.linalg.norm(F, axis=-1, keepdims=True).clip(1e-8)
    Xtr = F[tr]
    scale = np.sqrt((Xtr ** 2).sum(-1).mean(0)) if norm == "unit_norm" else np.ones(F.shape[1])
    probs, ce = [], []
    for l in range(F.shape[1]):
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, class_weight="balanced")
        m.fit(Xtr[:, l] / scale[l], y[tr])
        ce.append(float(log_loss(y[tr], m.predict_proba(Xtr[:, l] / scale[l])[:, 1], labels=[0, 1])))
        probs.append(m.predict_proba(F[te, l] / scale[l])[:, 1])
    ce = np.array(ce)
    keep = ce < 0.6
    if not keep.any():
        keep = ce == ce.min()
    return np.mean(np.array(probs)[keep], 0), (np.where(keep)[0] + 1).tolist(), ce.round(4).tolist()


def run(p):
    f_out = OUT / f"{p}_own_b2.json"
    if f_out.exists():
        print("skip", p); return
    t0 = time.time()
    fm, rows = folds(), load_rows(p)
    D = draws_for(p, rows, fm)
    y = yvec(rows)
    fold = np.array([fm[r["prompt_id"]] for r in rows])
    F = Cache(p, "own").final()
    sc = {n: np.full((N_ROWS, N_DRAWS), np.nan) for n in ("b_all", "b_l2_all", "b_fact")}
    info = {}
    for k in range(N_FOLDS):
        te = np.where(fold == k)[0]
        for d in range(N_DRAWS):
            tr = np.array(D[k][d])
            sc["b_all"][te, d], kept, ce = fit_b2(F, y, tr, te)
            info[f"{k}_all_{d}"] = {"kept": kept, "train_ce": ce}
            sc["b_l2_all"][te, d] = fit_b2(F, y, tr, te, norm="l2")[0]
        tr = np.array(fact_train(rows, fm, k))
        sc["b_fact"][te, 0], kept, ce = fit_b2(F, y, tr, te)
        info[f"{k}_fact_0"] = {"kept": kept, "train_ce": ce}
        print(f"{p} b2 fold {k} done {time.time() - t0:.0f}s", flush=True)
    save_scores(f_out, {"policy": p, "read": "own", "arm": "b (refit, D7)", "info": info,
                        "seconds": round(time.time() - t0)}, sc)


def run_pooled():
    """Pooled fact-present fit over the 4 trained models (as probes.py fit_pooled_fact), arm (b) refit only."""
    from probes import TRAINED
    f = OUT / "pooled_fact.json"
    d = json.loads(f.read_text())
    if "b_fact_pooled_refit_D7" in d.get("note", ""):
        print("skip pooled"); return
    fm = folds()
    R = {p: load_rows(p) for p in TRAINED}
    y = np.concatenate([yvec(R[p]) for p in TRAINED])
    fold = np.concatenate([[fm[r["prompt_id"]] for r in R[p]] for p in TRAINED])
    F = np.concatenate([Cache(p, "own").final() for p in TRAINED])
    s = np.full((len(y), 1), np.nan)
    for k in range(N_FOLDS):
        tr = np.array([j * N_ROWS + i for j, p in enumerate(TRAINED) for i, r in enumerate(R[p])
                       if r["cls"] in ("spin", "truthful") and r["y_string"] == 0 and fm[r["prompt_id"]] != k])
        te = np.where(fold == k)[0]
        s[te, 0] = fit_b2(F, y, tr, te)[0]
    d["scores"]["b_torch_fact_pooled"] = d["scores"].pop("b_fact_pooled")
    d["scores"]["b_fact_pooled"] = [[None if np.isnan(v) else round(float(v), 6) for v in r] for r in s]
    d["note"] = "b_fact_pooled_refit_D7: b_fact_pooled = refit_b.fit_b2; the under-trained torch run is b_torch_fact_pooled"
    f.write_text(json.dumps(d))
    print("pooled b refit written")


if __name__ == "__main__":
    if sys.argv[1:] == ["pooled"]:
        run_pooled()
    else:
        for p in sys.argv[1:] or POLICIES:
            run(p)
