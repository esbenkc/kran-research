"""DEVIATIONS D2 (exploratory, outside the verdict; added after the first result):
  (A) one-layer megaprobe: the mega recipe (hp.json k=32, L2=1e-2, not re-selected) on ONE layer, l = 18 and l = 22;
      attention runs over tokens only.
  (B) all-layer linear, no token selection: the mean-pooled l = 1..28 vectors concatenated (28 x 1536), standardised,
      logistic regression, C by the lin_l rule (PREREG s4).
  Sensitivity: mega catch with the line at the 1% point of the 596 eval true statements of each state (5th highest).
All fits on the clean training pool only; every state scored; PREREG s5 line + city bootstrap.
-> data/scores_controls.npz, data/results/controls.json.   Run: uv run python controls.py
"""
import json

import numpy as np
import torch

from analysis import ATT, Catch, OUT, setup, ci
from common import DATA, STATES, Acts
from probes import (DEV, LAYERS, LIN_C, TIE, fit_lin, fit_mega, folds, lin_score, load_pca, mega_score, pca_project,
                    pool_idx, pooled, rows)
from sklearn.metrics import roc_auc_score


def concat(st):
    return np.concatenate([pooled(st, l) for l in LAYERS], 1)


def main():
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    hp = json.loads((DATA / "hp.json").read_text())
    k, l2 = hp["mega"]["k"], hp["mega"]["l2"]
    S = {}
    out = {"hp": {"mega_k": k, "mega_l2": l2}}
    # (A) one-layer megaprobe at l18 and l22
    pca = load_pca()
    ZMs = {st: pca_project(Acts(st), pca, k) for st in STATES}
    allidx = np.arange(len(y))
    for l in (18, 22):
        j = LAYERS.index(l)
        one = {st: (ZMs[st][0][:, :, j:j + 1, :].contiguous(), ZMs[st][1]) for st in STATES}
        P, ep, log = fit_mega(one["clean"], y, city, tr, k, l2)
        out[f"A_mega_l{l}"] = {"best_epoch": ep}
        for st in STATES:
            S[f"{st}__megaA_l{l}"] = mega_score(P, one[st], allidx)
        del one
    del ZMs
    if DEV.type == "mps":
        torch.mps.empty_cache()
    # (B) all-layer concatenated mean-pooled linear probe, C by the lin_l rule
    Xc = concat("clean")
    res = {}
    for C in LIN_C:
        oof = np.zeros(len(y))
        for a, b in folds(tr, city, 5):
            sc, m = fit_lin(Xc, y, a, C)
            oof[b] = lin_score(sc, m, Xc[b])
        res[C] = float(roc_auc_score(y[tr], oof[tr]))
        print("B C", C, res[C], flush=True)
    best = max(res.values())
    Cb = min(c for c in LIN_C if res[c] >= best - TIE)
    out["B_lin_all"] = {"C": Cb, "cv_auroc": res}
    sc, m = fit_lin(Xc, y, tr, Cb)
    S["clean__linB_all"] = lin_score(sc, m, Xc)
    del Xc
    for st in ATT:
        S[f"{st}__linB_all"] = lin_score(sc, m, concat(st))
    np.savez(DATA / "scores_controls.npz", **S)
    # catch + CI with the PREREG s5 line
    ctx = list(setup())
    Z = dict(ctx[0])
    Z.update(S)
    ctx[0] = Z
    catch = Catch(tuple(ctx))
    for mon in ["megaA_l18", "megaA_l22", "linB_all"]:
        c = catch(mon)
        c.pop("_boot_pooled"); c.pop("_boot_clean")
        out[mon] = c
    # sensitivity: mega with the line from the eval true statements (5th highest of 596), re-picked per bootstrap draw
    _, y_, _, city_, _, _, ev_, cities, cid, W = ctx
    evf, evt = ev_[y_[ev_] == 1], ev_[y_[ev_] == 0]
    sens = {}
    rng = np.random.default_rng(1020)
    city_rows = {c: (evf[city_[evf] == c], evt[city_[evt] == c]) for c in cities}
    boots = np.zeros((len(W), 3))
    for si, st in enumerate(ATT):
        s = Z[f"{st}__mega"]
        t = np.sort(s[evt])[::-1][4]  # k = floor(0.01 * 597) = 5
        sens[st] = {"catch": float(np.mean(s[evf] > t)), "line": float(t), "false_alarm_eval": float(np.mean(s[evt] > t))}
    for b in range(len(W)):
        draw = rng.choice(len(cities), len(cities), replace=True)
        F = np.concatenate([city_rows[cities[i]][0] for i in draw])
        T = np.concatenate([city_rows[cities[i]][1] for i in draw])
        kk = max(1, int(np.floor(0.01 * (len(T) + 1))))
        for si, st in enumerate(ATT):
            s = Z[f"{st}__mega"]
            t = np.sort(s[T])[::-1][kk - 1]
            boots[b, si] = np.mean(s[F] > t)
    sens["pooled_attacked"] = {"catch": float(np.mean([sens[st]["catch"] for st in ATT])), "ci95": ci(boots.mean(1)),
                               "note": "line = 5th highest of the 596 eval true statements per state; re-picked per draw"}
    out["sensitivity_mega_eval_true_line"] = sens
    # D2 reading (fixed before the arms ran)
    lo = {l: out[f"megaA_l{l}"]["pooled_attacked"]["ci95"][0] for l in (18, 22)}
    pt = {l: out[f"megaA_l{l}"]["pooled_attacked"]["catch"] for l in (18, 22)}
    b = out["linB_all"]["pooled_attacked"]["catch"]
    if max(lo.values()) >= 0.5:
        reading = "token selection in the upper half does most of the work"
    elif max(pt.values()) < 0.3 and b < 0.3:
        reading = "only the combination of layers and token selection recovers the catch"
    else:
        reading = "both parts contribute"
    out["D2_reading"] = reading
    (OUT / "controls.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({m: {"clean": out[m]["clean"]["catch"], "pooled": out[m]["pooled_attacked"]}
                      for m in ["megaA_l18", "megaA_l22", "linB_all"]}, indent=1))
    print("sensitivity", json.dumps(sens["pooled_attacked"]), {st: sens[st]["catch"] for st in ATT})
    print("D2 reading:", reading)


if __name__ == "__main__":
    main()
