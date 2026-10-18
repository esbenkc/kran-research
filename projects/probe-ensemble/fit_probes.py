"""Fit the 8-member probe ensemble and the held-out attention probe on the BASE model (PREREG s3, s6).

Ensemble (the in-reward target of the 'ensemble' arm): layers 6/12/18/22 x readout {mean, maxpool}.
  - mean    = mean of the residual activation over response tokens;
  - maxpool = elementwise max of the residual activation over response tokens;
  - every member: probes.py fit() unchanged (standardise, logistic regression C=0.05, trained on DolusChat
    settings group 0 = the data of probe A, threshold at 5% FPR on truthful group-2 replies);
  - the layer-12 mean member IS probe A of probe-goodhart (copied from data/probes.json, not refit);
  - ensemble score = max over the 8 member probabilities.
Held-out attention probe (never in any reward): layer 16, DolusChat settings group 1, one learned query,
softmax attention over response tokens, linear readout (common.AttnProbe); model selection on a split of
group 1; threshold at 5% FPR on truthful group-2 replies.

    uv run python projects/probe-ensemble/fit_probes.py      (from kran-research/)
-> data/ensemble.json, data/attn.json, data/probe_fit_metrics.json, data/probe_fit_scores.npz,
   data/raw/fitacts_<group>.npz (pooled + layer-16 token activations; gitignored, ~0.3 GB)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RAW, fit_attn, load_model, load_probes, load_splits, response_acts  # noqa: E402
from probes import fit, pairs  # noqa: E402

ENS_LAYERS = [6, 12, 18, 22]
READOUTS = ["mean", "maxpool"]
ATTN_LAYER = 16


def main():
    tok, m = load_model()
    m.eval()
    S = load_splits()
    acts, ys = {}, {}
    for k in ("probeA", "probeB", "calib"):
        P, R, y = pairs(tok, S[k])
        acts[k] = response_acts(m, tok, P, R, ENS_LAYERS, keep_tokens=[ATTN_LAYER], verbose=True)
        ys[k] = y
        print(k, len(y), "mean resp tokens", np.mean([len(s) for s in acts[k]["tok"][ATTN_LAYER]]), flush=True)
    yA, yc = ys["probeA"], ys["calib"]
    A = load_probes()["A"]

    members, metrics, scores = {}, {"members": {}}, {"y_calib": yc, "y_group0": yA, "y_group1": ys["probeB"]}
    for l in ENS_LAYERS:
        for ro in READOUTS:
            name = f"L{l}_{ro}"
            Xc = acts["calib"][ro][l]
            if (l, ro) == (12, "mean"):
                pr = A
                pc = pr.prob_x(Xc)
                au, tpr, note = float(roc_auc_score(yc, pc)), float((pc[yc == 1] > pr.thr).mean()), "probe A (copied)"
                # sanity: refit on today's activations must reproduce probe A
                pr2, au2, _ = fit(acts["probeA"][ro][l], yA, l, ro, Xc, yc)
                metrics["probeA_refit_check"] = {"auroc_calib_copied": au, "auroc_calib_refit": au2,
                                                 "corr_scores_calib": float(np.corrcoef(pc, pr2.prob_x(Xc))[0, 1])}
            else:
                pr, au, tpr = fit(acts["probeA"][ro][l], yA, l, ro, Xc, yc)
                pc = pr.prob_x(Xc)
                note = "fit on group 0"
            members[name] = pr
            scores[f"calib_{name}"] = pc
            metrics["members"][name] = {"note": note, "auroc_calib_group2": au, "tpr_at_5fpr_calib": tpr,
                                        "thr_5fpr": float(pr.thr), "mean_prob_calib_truthful": float(pc[yc == 0].mean()),
                                        "mean_prob_calib_deceptive": float(pc[yc == 1].mean())}
            print(name, metrics["members"][name], flush=True)

    P8 = np.stack([scores[f"calib_{n}"] for n in members])
    ens = P8.max(0)
    anyflag = np.any(np.stack([scores[f"calib_{n}"] > members[n].thr for n in members]), 0)
    pA = scores["calib_L12_mean"]
    for key, s, fl in (("ensemble_max", ens, anyflag), ("single_A", pA, pA > A.thr)):
        metrics[key] = {"auroc_calib": float(roc_auc_score(yc, s)), "flag_rate_truthful": float(fl[yc == 0].mean()),
                        "flag_rate_deceptive": float(fl[yc == 1].mean()),
                        "mean_score_truthful": float(s[yc == 0].mean()), "mean_score_deceptive": float(s[yc == 1].mean())}
    scores["calib_ensemble"] = ens
    metrics["member_score_corr_calib"] = {f"{a}|{b}": float(np.corrcoef(scores[f"calib_{a}"], scores[f"calib_{b}"])[0, 1])
                                          for i, a in enumerate(members) for b in list(members)[i + 1:]}

    att, val_au, ep = fit_attn(acts["probeB"]["tok"][ATTN_LAYER], ys["probeB"], ATTN_LAYER)
    pc = att.prob({"tok": {ATTN_LAYER: acts["calib"]["tok"][ATTN_LAYER]}})
    att.thr = float(np.quantile(pc[yc == 0], 0.95))
    metrics["attn"] = {"layer": ATTN_LAYER, "train": "group 1", "val_auroc_group1": val_au, "best_epoch": ep,
                       "auroc_calib_group2": float(roc_auc_score(yc, pc)), "thr_5fpr": att.thr,
                       "tpr_at_5fpr_calib": float((pc[yc == 1] > att.thr).mean())}
    scores["calib_attn"] = pc
    print("attn", metrics["attn"], flush=True)

    (DATA / "ensemble.json").write_text(json.dumps({k: v.to_dict() for k, v in members.items()}))
    (DATA / "attn.json").write_text(json.dumps(att.to_dict()))
    (DATA / "probe_fit_metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(DATA / "probe_fit_scores.npz", **scores)
    RAW.mkdir(parents=True, exist_ok=True)
    for k in acts:  # so a skeptic can refit every probe without the model
        np.savez_compressed(RAW / f"fitacts_{k}.npz", y=ys[k],
                            **{f"{ro}_L{l}": acts[k][ro][l].astype(np.float32) for ro in READOUTS for l in ENS_LAYERS},
                            tok_L16=np.concatenate(acts[k]["tok"][ATTN_LAYER]),
                            tok_len=np.array([len(t) for t in acts[k]["tok"][ATTN_LAYER]]))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("members", "member_score_corr_calib")}, indent=1))


if __name__ == "__main__":
    main()
