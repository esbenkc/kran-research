"""skeptic3: independent re-analysis of build's saved runs (own code, own Sigma_hat, own refit).
Run from kran-research/:  uv run python ../swarm-theory-1016-probe-evasion/skeptic3/recompute.py
Reads projects/probe-evasion-geometry/data/runs/*.npz. Writes out/recompute.json + out/recompute.txt.
"""
import glob, json, pathlib, sys
import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path("projects/probe-evasion-geometry/data/runs")
OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
NPERM = 1000
rng = np.random.default_rng(20261016)


def auc(y, s): return roc_auc_score(y, s)


def dprime_from_auc(a, n):
    a = np.clip(a, 1 / (2 * n), 1 - 1 / (2 * n))
    return np.sqrt(2) * norm.ppf(a)


def pooled_lw(X, y):
    Xc = np.concatenate([X[y == c] - X[y == c].mean(0) for c in (0, 1)])
    return LedoitWolf(assume_centered=True).fit(Xc).covariance_


def r2p(p, m): return float(np.corrcoef(p, m)[0, 1] ** 2)


def r2id(p, m): return float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum())


def analyse(path):
    z = np.load(path, allow_pickle=True)
    Xtr, ytr, Etr = z["X_train"], z["y_train"].astype(int), z["E_train"]
    Xte, yte, Ete = z["X_test"], z["y_test"].astype(int), z["E_test"]
    w, U = z["w"].astype(float), z["probes"].astype(float)
    n = min((yte == 0).sum(), (yte == 1).sum())
    S = pooled_lw(Xtr, ytr)
    S_build = z["Sigma_hat"] if "Sigma_hat" in z.files else None

    before = np.array([auc(yte, Xte @ u) for u in U]); after = np.array([auc(yte, Ete @ u) for u in U])
    drop = before - after
    aw0, aw1 = auc(yte, Xte @ w), auc(yte, Ete @ w)
    dw0, dw1 = dprime_from_auc(aw0, n), dprime_from_auc(aw1, n)
    k = 1 - dw1 / dw0
    du = dprime_from_auc(before, n)

    def cos_M(M):
        return (U @ M @ w) / np.sqrt(np.einsum("ij,jk,ik->i", U, M, U) * (w @ M @ w))

    cw, craw = cos_M(S), cos_M(np.eye(len(w)))

    def pred(c): return before - norm.cdf((du - k * dw0 * c) / np.sqrt(2))

    pw, pr = pred(cw), pred(craw)
    # exact Euclidean-move predictor (min ||Delta||^2 move along w), skeptic2 #1036 form
    scale = (np.linalg.norm(U, axis=1) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U))) * (np.sqrt(w @ S @ w) / np.linalg.norm(w))
    pe = pred(craw * scale)

    # nulls
    perm = np.array([r2p(pred(rng.permutation(cw)), drop) for _ in range(NPERM)])
    placebo = []  # cos to random directions w' instead of w (same k, d'_w)
    for _ in range(200):
        wr = rng.normal(size=len(w))
        c = (U @ S @ wr) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U) * (wr @ S @ wr))
        placebo.append(r2p(pred(c), drop))
    placebo = np.array(placebo)
    # 'strong probes drop more' null: predicted drop with cos=1 for all probes
    r2_cos1 = r2p(pred(np.ones_like(cw)), drop)

    # refit (own)
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(Etr, ytr)
    refit = auc(yte, Ete @ lr.coef_[0])
    # diagnostics
    m0 = np.array([Xte[yte == c].mean(0) for c in (0, 1)]); m1 = np.array([Ete[yte == c].mean(0) for c in (0, 1)])
    shift_gap = (m1[1] - m1[0]) - (m0[1] - m0[0])
    Sw = S @ w
    cosv = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    S_after = pooled_lw(Ete, yte); S_before = pooled_lw(Xte, yte)
    # moment-based d' secondary
    D_te = m0[1] - m0[0]
    du_mom = (U @ D_te) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U))
    dw_mom = w @ D_te / np.sqrt(w @ S @ w)
    pmom = before - norm.cdf((du_mom - k * dw_mom * cw) / np.sqrt(2))
    # exact lemma check in d' (moment) units on the measured mean gaps (Sigma held at clean)
    D_after = m1[1] - m1[0]
    du_after_mom = (U @ D_after) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U))
    k_mom = 1 - (w @ D_after / np.sqrt(w @ S @ w)) / dw_mom
    r2_dprime_identity = r2p(du_mom - k_mom * dw_mom * cw, du_after_mom)

    return dict(
        file=path.name, n_test_per_class=int(n), auroc_w=[float(aw0), float(aw1)], k=float(k),
        sigma_relerr_vs_build=(float(np.linalg.norm(S - S_build) / np.linalg.norm(S_build)) if S_build is not None else None),
        corr_raw_whit_cos=float(np.corrcoef(craw, cw)[0, 1]),
        r2_whitened=r2p(pw, drop), r2_raw=r2p(pr, drop), r2_euclid_exact=r2p(pe, drop),
        r2id_whitened=r2id(pw, drop), r2id_raw=r2id(pr, drop),
        r2_whitened_momentdprime=r2p(pmom, drop), r2_dprime_identity_moment=r2_dprime_identity,
        null_perm_r2_mean=float(perm.mean()), null_perm_r2_q95=float(np.quantile(perm, 0.95)),
        null_perm_p=float((perm >= r2p(pw, drop)).mean()),
        null_placebo_r2_mean=float(placebo.mean()), null_placebo_r2_q95=float(np.quantile(placebo, 0.95)),
        null_cos1_r2=r2_cos1,
        refit_auroc=float(refit),
        n_probes_auc_gt_0995=int((before > 0.995).sum()),
        drop_range=[float(drop.min()), float(drop.max())],
        shift_cos_Sigma_w=cosv(shift_gap, Sw), shift_cos_w=cosv(shift_gap, w),
        cov_rel_change=float(np.linalg.norm(S_after - S_before) / np.linalg.norm(S_before)),
    )


rows = []
files = sorted(ROOT.glob(sys.argv[1] if len(sys.argv) > 1 else "*.npz"))
for f in files:
    r = analyse(f); rows.append(r)
    print(json.dumps(r), flush=True)
(OUT / "recompute.json").write_text(json.dumps(rows, indent=1))

# per (setting, retain) means
groups = {}
for r in rows:
    key = r["file"].rsplit("_seed", 1)[0]; groups.setdefault(key, []).append(r)
lines = []
for key, rs in groups.items():
    m = lambda f: np.mean([x[f] for x in rs])
    lines.append(f"{key}: n_seeds={len(rs)} whit_r2={m('r2_whitened'):.3f} raw_r2={m('r2_raw'):.3f} "
                 f"euclid_exact_r2={m('r2_euclid_exact'):.3f} refit={m('refit_auroc'):.3f} "
                 f"whit_r2id={m('r2id_whitened'):.3f} raw_r2id={m('r2id_raw'):.3f} "
                 f"perm_null={m('null_perm_r2_mean'):.3f}(q95 {m('null_perm_r2_q95'):.3f}) "
                 f"placebo={m('null_placebo_r2_mean'):.3f} cos1={m('null_cos1_r2'):.3f} "
                 f"corr(raw,whit)={m('corr_raw_whit_cos'):.2f} shift.cos(Sw)={m('shift_cos_Sigma_w'):.2f} "
                 f"shift.cos(w)={m('shift_cos_w'):.2f} covchg={m('cov_rel_change'):.3f} AUROC_w_after={m('auroc_w')[1] if False else np.mean([x['auroc_w'][1] for x in rs]):.3f} "
                 f"per-seed whit={[round(x['r2_whitened'],3) for x in rs]} raw={[round(x['r2_raw'],3) for x in rs]} refit={[round(x['refit_auroc'],3) for x in rs]}")
(OUT / "recompute.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
