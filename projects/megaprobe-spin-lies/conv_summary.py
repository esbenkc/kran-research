"""Summarise data/convergence/*.json (convergence.py) against the one-layer probe on the same draw (draw 0).

Rule for the published probe at 40 epochs, fixed before its results (room post #3644, 2026-10-20 11:31): per fold, the
checkpoint with the best inner-validation AUROC (the megaprobe's rule). Also printed: the lowest-validation-loss
checkpoint (FIBS's rule), the last epoch, and the 10-epoch replay of the main run.

    uv run python projects/megaprobe-spin-lies/conv_summary.py  -> data/convergence/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes as P  # noqa: E402
from convergence import OUTD, subsets  # noqa: E402

SUBS = ("spin_vs_fact_truthful", "spin_vs_all_truthful", "lies_vs_truthful")


def pooled(per_fold, pick, S):
    s = np.full(P.N_ROWS, np.nan); eps = []
    for f in per_fold.values():
        e = pick(f["log"]); s[f["te"]] = e["test"]; eps.append(e["epoch"])
    return {"epochs": eps, **{n: round(float(roc_auc_score(S[n][1][S[n][0]], s[S[n][0]])), 4) for n in SUBS}}


def main():
    out = {}
    for f in sorted(OUTD.glob("c*_*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text()); p = d["policy"]
        rows = P.load_rows(p); S = subsets(rows)
        pf = {int(k): v for k, v in d["per_fold"].items()}
        if "test" not in pf[0]["log"][0]:
            continue  # first lam1_s0 runs kept only the pooled curve
        r = {"arm": d["arm"], "policy": p, "hp": d["hp"]}
        r["best_val_auroc"] = pooled(pf, lambda lg: max(lg, key=lambda e: e["val_auroc"]), S)
        r["last_epoch"] = pooled(pf, lambda lg: lg[-1], S)
        if d["arm"] == "c2":
            r["lowest_val_loss"] = pooled(pf, lambda lg: min(lg, key=lambda e: e["val_bce"]), S)
            r["main_run_replay_10ep"] = pooled(pf, lambda lg: min(lg[:10], key=lambda e: e["val_bce"]), S)
            r["train_auroc_ep10_ep40"] = [d["curve"][9]["train_auroc"], d["curve"][-1]["train_auroc"]]
        else:
            r["main_run_replay_300ep"] = pooled(pf, lambda lg: max([e for e in lg if e["epoch"] < 300],
                                                                    key=lambda e: e["val_auroc"]), S)
        fits = json.loads((P.OUT / f"{p}_own.json").read_text())["scores"]["a_all"]
        a0 = np.array([np.nan if v[0] is None else v[0] for v in fits], float)
        r["one_layer_draw0"] = {n: round(float(roc_auc_score(S[n][1][S[n][0]], a0[S[n][0]])), 4) for n in SUBS}
        out[f.stem] = r
        print(f.stem, json.dumps({k: v for k, v in r.items() if k not in ("hp",)}))
    (OUTD / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
