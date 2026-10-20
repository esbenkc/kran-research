"""DEVIATIONS D3 (exploratory, outside the verdict; added after the first result and Vera's readout checks):
  (C) country-token probe at EVERY layer l = 1..28: a linear probe on one token per statement, the last token of the
      country name (the token just before the final full stop; if the full stop is not its own token, the statement's
      last token, counted). StandardScaler + LogisticRegression, C by the lin_l rule (PREREG s4), clean fit pool.
      Catch at the s5 line + city CI, eval false alarms, twin-paired d' with the clean spread (s8), every state.
  (D) megaprobe fit-seed spread: the mega fit (hp.json, 1400 labels) repeated with fit seeds 1021 and 1022.
-> data/scores_d3.npz, data/results/d3.json.   Run: uv run python d3.py
"""
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from analysis import ATT, Catch, Dprime, OUT, setup
from common import DATA, MODEL, STATES, Acts, load_split, tokenize
from probes import DEV, LAYERS, fit_lin, fit_mega, lin_score, load_pca, mega_score, pca_project, pool_idx, rows, \
    select_lin


def main():
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    hp = json.loads((DATA / "hp.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tok, [r["text"] for r in load_split()])
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    A0 = Acts("clean")
    row = np.where(stop_own, A0.off[1:] - 2, A0.off[1:] - 1)  # flat token row of the country's last token
    S, out = {}, {"n_full_stop_not_own_token": int((~stop_own).sum()), "C": {}}
    acts = {st: Acts(st) for st in STATES}
    for l in LAYERS:
        X = {st: np.asarray(acts[st].tok[row, l], np.float32) for st in STATES}
        C, res = select_lin(X["clean"], y, city, tr)
        out["C"][l] = {"C": C, "cv_auroc": res}
        sc, m = fit_lin(X["clean"], y, tr, C)
        for st in STATES:
            S[f"{st}__tok{l}"] = lin_score(sc, m, X[st])
        print("tok", l, C, flush=True)
    # (D) megaprobe fit seeds 1021, 1022 (headline stays 1020)
    pca = load_pca()
    k, l2 = hp["mega"]["k"], hp["mega"]["l2"]
    ZMc = pca_project(acts["clean"], pca, k)
    Ps = {}
    for sd in (1021, 1022):
        Ps[sd], ep, _ = fit_mega(ZMc, y, city, tr, k, l2, seed=sd)
        out[f"mega_seed{sd}_best_epoch"] = ep
    del ZMc
    allidx = np.arange(len(y))
    for st in STATES:
        ZM = pca_project(acts[st], pca, k)
        for sd, P in Ps.items():
            S[f"{st}__mega_fit{sd}"] = mega_score(P, ZM, allidx)
        del ZM
        if DEV.type == "mps":
            torch.mps.empty_cache()
    np.savez(DATA / "scores_d3.npz", **S)
    ctx = list(setup())
    Z = dict(ctx[0]); Z.update(S); ctx[0] = Z
    ctx = tuple(ctx)
    catch, dp = Catch(ctx), Dprime(ctx)
    out["tok"] = {}
    for l in LAYERS:
        c = catch(f"tok{l}")
        c.pop("_boot_pooled"); c.pop("_boot_clean")
        d, _ = dp(f"tok{l}")
        out["tok"][l] = {"catch": c, "dprime": d}
    out["mega_fit_seeds"] = {}
    for name in ["mega", "mega_fit1021", "mega_fit1022"]:
        c = catch(name)
        c.pop("_boot_pooled"); c.pop("_boot_clean")
        out["mega_fit_seeds"][name] = {"clean": c["clean"]["catch"], "pooled_attacked": c["pooled_attacked"]}
    (OUT / "d3.json").write_text(json.dumps(out, indent=1))
    for l in LAYERS:
        c = out["tok"][l]["catch"]
        print(l, round(c["clean"]["catch"], 3), round(c["pooled_attacked"]["catch"], 3),
              [round(x, 3) for x in c["pooled_attacked"]["ci95"]], [round(x, 3) for x in c["pooled_attacked"]["per_seed"]],
              "d'", round(out["tok"][l]["dprime"]["clean"], 2), round(out["tok"][l]["dprime"]["attacked"], 2))
    print(json.dumps(out["mega_fit_seeds"], indent=1), "n_full_stop_not_own_token", out["n_full_stop_not_own_token"])


if __name__ == "__main__":
    main()
