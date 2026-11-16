"""The frozen monitor at 8B: clean-fit country-token probes tok<R> at the PANEL readouts and the panel monitor.

At 1.5B the probes came from theory-1021 (data/fits/panel.npz there). At 8B they are fit here, on the CLEAN model, with
the same rule (1020/1021 C rule; refit.select): StandardScaler + L2 logistic regression on the last country-name token,
fit pool = probe+attack+stop (1400 statements, 350 cities), C from {0.01, 0.1, 1, 10} by 5-fold GroupKFold-by-city
out-of-fold AUROC (ties within 0.002 -> smaller C), lbfgs max_iter 5000.

  tok<R>(x)   = ((h_R(x) - mu_R) / sd_R) . w_R + b_R                (raw logit; positive = false statement)
  z_R(x)      = (tok<R>(x) - m_R) / v_R, m_R, v_R = mean and std (ddof 0) of tok<R> over the CLEAN model's fit-pool
                TRUE statements (700) -> data/fits/zmoments.json, frozen before any fine-tune step
  panelmax(x) = max over the PANEL readouts R of z_R(x)
Alarm line of any monitor in a state: 2nd-highest score of that state's 200 line-set true statements (common.threshold).
Random-direction control (PREREG s5; data/fits/random_panel.npz, written by the zmoments stage): a unit direction per
readout (numpy seed 1116 + R) scaled so its step-0 mean squared twin gap on the 300 attack pairs equals the truth
probe's on the clean model. Used only by attack_panel.py --control random.
Best single readout (PREREG s8): stage bestsingle fits tok<R> for every readout R = 1..n+1 of the clean model (needs
extract.py --state clean --all) -> data/fits/bestsingle.json.
Stages:
  fit            -> data/fits/panel.npz, data/fits/fit_info.json (from data/acts/clean)
  zmoments       -> data/fits/zmoments.json (from data/scores/clean.npz)
  score <state>  -> data/scores/<state>.npz: tok<R> for R in PANEL and panelmax, all 2992 statements
Run: uv run python panel.py fit ; uv run python panel.py score clean ; uv run python panel.py zmoments
"""
import json
import sys
import time

import numpy as np

from common import DATA, PANEL, Acts, pool_idx, rows

FITS = DATA / "fits"
SCORES = DATA / "scores"


def lin_apply(F, nm, X):
    return ((X - F[f"{nm}__mu"]) / F[f"{nm}__sd"]) @ F[f"{nm}__w"] + float(F[f"{nm}__b"])


def panelmax(S, readouts=PANEL):
    Z = json.loads((FITS / "zmoments.json").read_text())
    return np.max(np.stack([(S[f"tok{R}"] - Z[f"tok{R}"]["mu"]) / Z[f"tok{R}"]["sd"] for R in readouts], 1), 1)


def stage_fit():
    from refit import fit, select
    t0 = time.time()
    FITS.mkdir(parents=True, exist_ok=True)
    _, y, city, _ = rows()
    tr, _, _ = pool_idx()
    A = Acts("clean")
    F, info = {}, {}
    for R in PANEL:
        X = A.at(R)
        C, res, conv = select(X, y, city, tr)
        sc, m = fit(X, y, tr, C)
        F[f"tok{R}__mu"] = sc.mean_.astype(np.float64)
        F[f"tok{R}__sd"] = sc.scale_.astype(np.float64)
        F[f"tok{R}__w"] = m.coef_[0].astype(np.float64)
        F[f"tok{R}__b"] = np.float64(m.intercept_[0])
        info[R] = {"C": C, "oof_auroc": res[C], "grid": res, "n_iter": int(m.n_iter_[0]),
                   "converged": bool(m.n_iter_[0] < 5000), "cv_fits_converged": conv}
        print("fit", R, C, round(res[C], 4), f"{time.time() - t0:.0f}s", flush=True)
    np.savez(FITS / "panel.npz", readouts=np.array(PANEL), **F)
    (FITS / "fit_info.json").write_text(json.dumps({"readouts": PANEL, "per_readout": info}, indent=1))


def stage_zmoments():
    _, y, _, _ = rows()
    tr, _, _ = pool_idx()
    neg = tr[y[tr] == 0]
    S = np.load(SCORES / "clean.npz")
    out = {f"tok{R}": {"mu": float(S[f"tok{R}"][neg].mean()), "sd": float(S[f"tok{R}"][neg].std()), "n": int(len(neg))}
           for R in PANEL}
    (FITS / "zmoments.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in list(out)[::6]}))
    # random-direction control (PREREG s5, plan v1.3): unit direction u_R in the probe's standardized space (numpy
    # seed 1116 + R), scaled so its step-0 mean squared false-minus-true twin gap over the 300 attack pairs (600
    # statements) equals the truth probe's on the clean model. Same mu, sd, b as tok<R>.
    F = np.load(FITS / "panel.npz")
    R_, _, city, split = rows()
    src = np.array([r["src"] for r in R_])
    pos = np.where((split == "attack") & (y == 1))[0]
    tw = np.array([int(np.where((split == "attack") & (y == 0) & (city == city[i]) & (src == src[i]))[0][0]) for i in pos])
    A = Acts("clean")
    rp, info = {"readouts": np.array(PANEL)}, {}
    for R in PANEL:
        w = F[f"tok{R}__w"]
        u = np.random.default_rng(1116 + R).standard_normal(len(w))
        u /= np.linalg.norm(u)
        dZ = (A.at(R, pos) - A.at(R, tw)) / F[f"tok{R}__sd"]          # standardized false-minus-true twin difference
        g_true = float(((dZ @ w) ** 2).mean())
        g_unit = float(((dZ @ u) ** 2).mean())
        c = np.sqrt(g_true / g_unit)
        rp[f"rand{R}__w"] = u * c
        info[R] = {"gap2_truth": g_true, "gap2_unit": g_unit, "scale": float(c), "w_norm": float(np.linalg.norm(w)),
                   "scale_over_w_norm": float(c / np.linalg.norm(w)), "over_1e3": bool(c > 1e3 * np.linalg.norm(w))}
    np.savez(FITS / "random_panel.npz", **rp)
    (FITS / "random_panel.json").write_text(json.dumps(info, indent=1))
    print("random_panel scale/||w||:", {R: round(info[R]["scale_over_w_norm"], 1) for R in PANEL})


def stage_score(state):
    t0 = time.time()
    SCORES.mkdir(parents=True, exist_ok=True)
    F = np.load(FITS / "panel.npz")
    assert list(F["readouts"]) == PANEL
    A = Acts(state)
    S = {f"tok{R}": lin_apply(F, f"tok{R}", A.at(R)) for R in PANEL}
    if (FITS / "zmoments.json").exists():
        S["panelmax"] = panelmax(S)
    np.savez(SCORES / f"{state}.npz", **S)
    print(state, "scored", f"{time.time() - t0:.0f}s", flush=True)


def stage_bestsingle():
    from metrics import catch
    from refit import fit, select
    t0 = time.time()
    _, y, city, _ = rows()
    tr, _, _ = pool_idx()
    A = Acts("clean")
    out = {}
    for R in A.readouts:
        X = A.at(R)
        C, res, conv = select(X, y, city, tr)
        sc, m = fit(X, y, tr, C)
        c = catch(m.decision_function(sc.transform(X)))
        out[R] = {"C": C, "oof_auroc": res[C], "eval_catch": c["catch"], "in_panel": R in PANEL, "converged": conv}
        print("best", R, C, round(res[C], 4), c["catch"], f"{time.time() - t0:.0f}s", flush=True)
    b = max(out, key=lambda r: (out[r]["eval_catch"], -r))
    (FITS / "bestsingle.json").write_text(json.dumps({"best_readout": b, "best": out[b], "per_readout": out}, indent=1))
    print("best single readout", b, out[b])


if __name__ == "__main__":
    if sys.argv[1] == "bestsingle":
        stage_bestsingle()
    elif sys.argv[1] == "fit":
        stage_fit()
    elif sys.argv[1] == "zmoments":
        stage_zmoments()
    else:
        stage_score(sys.argv[2])
