"""skeptic3: independent checks of the lemma. Run from kran-research/:
uv run --with sympy --with torch python ../swarm-theory-1016-probe-evasion/skeptic3/derive.py
"""
import json, pathlib, sys
import numpy as np, sympy as sp, torch
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(3)
res = {}

def rand_spd(d, lo=0.1, hi=10.0, rng=rng):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q @ np.diag(np.geomspace(lo, hi, d)) @ Q.T

def dprime(u, delta, S): return u @ delta / np.sqrt(u @ S @ u)
def cosS(a, b, S): return a @ S @ b / np.sqrt((a @ S @ a) * (b @ S @ b))

# ---- C1: AUROC = Phi(d/sqrt2) --------------------------------------------------------
c1 = []
for _ in range(4):
    d = 6; S = rand_spd(d); L = np.linalg.cholesky(S)
    delta = rng.normal(size=d) * 0.5; u = rng.normal(size=d); n = 40000
    x0 = rng.normal(size=(n, d)) @ L.T; x1 = rng.normal(size=(n, d)) @ L.T + delta
    emp = roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[x0 @ u, x1 @ u])
    c1.append((emp, norm.cdf(dprime(u, delta, S) / np.sqrt(2))))
res["C1_auroc_formula_maxerr"] = float(max(abs(a - b) for a, b in c1))

# ---- C2: minimal move, sympy KKT + numeric -----------------------------------------
# Parametrise delta = S v (v = S^-1 delta, the LDA direction) so no symbolic inverse is needed.
w = sp.Matrix(sp.symbols("w1:4")); v = sp.Matrix(sp.symbols("v1:4"))
a, b, c, p, q, r = sp.symbols("a b c p q r")
Ss = sp.Matrix([[a, p, q], [p, b, r], [q, r, c]])
dl = Ss * v
Dstar = -(w.T * dl)[0] / (w.T * Ss * w)[0] * Ss * w
# KKT for min D^T S^-1 D s.t. w^T(dl+D)=0: 2 S^-1 D = lam w <=> D = (lam/2) S w (true by form);
# remaining condition is feasibility:
res["C2_sympy_constraint_zero"] = sp.cancel((w.T * (dl + Dstar))[0]) == 0
c2 = []
for _ in range(4):
    d = 5; S = rand_spd(d); Si = np.linalg.inv(S); delta = rng.normal(size=d); wv = rng.normal(size=d)
    cons = {"type": "eq", "fun": lambda D: wv @ (delta + D)}
    m = minimize(lambda D: D @ Si @ D, np.zeros(d), constraints=[cons], method="SLSQP", options={"ftol": 1e-14})
    e = minimize(lambda D: D @ D, np.zeros(d), constraints=[cons], method="SLSQP", options={"ftol": 1e-14})
    c2.append((np.abs(m.x - (-(wv @ delta) / (wv @ S @ wv) * S @ wv)).max(),
               np.abs(e.x - (-(wv @ delta) / (wv @ wv) * wv)).max()))
res["C2_numeric_maxerr_mahal_euclid"] = [float(max(x[0] for x in c2)), float(max(x[1] for x in c2))]

# ---- C3: d'_u = d_u - d_w cos_S(w,u), symbolic --------------------------------------
# Multiply both sides by sqrt(u'Su) > 0; then: u'(dl+D*) == u'dl - (w'dl)(w'Su)/(w'Sw), rational -> cancel.
u = sp.Matrix(sp.symbols("u1:4"))
lhs3 = (u.T * (dl + Dstar))[0]
rhs3 = (u.T * dl)[0] - (w.T * dl)[0] * (w.T * Ss * u)[0] / (w.T * Ss * w)[0]
res["C3_sympy_identity"] = sp.cancel(lhs3 - rhs3) == 0
# ---- C4: refit d^2 = D^2 - (w.delta)^2/(wSw) = D^2 (1 - cos_S(w, S^-1 delta)^2) ----
# S^-1 delta' = v + S^-1 Dstar = v - (w.dl/wSw) w  (no inverse needed)
Sinv_dnew = v - (w.T * dl)[0] / (w.T * Ss * w)[0] * w
dnew = dl + Dstar
lhs4 = (dnew.T * Sinv_dnew)[0]
D2 = (dl.T * v)[0]
res["C4_sympy_refit_identity"] = sp.cancel(lhs4 - (D2 - (w.T * dl)[0] ** 2 / (w.T * Ss * w)[0])) == 0
cs_opt2 = (w.T * Ss * v)[0] ** 2 / ((w.T * Ss * w)[0] * (v.T * Ss * v)[0])
res["C4_sympy_cos_form"] = sp.cancel(lhs4 - D2 * (1 - cs_opt2)) == 0
res["C4_refit_threshold_Dsin"] = float(np.sqrt(2) * norm.ppf(0.9))

# ---- C5: trained encoder, which cosine predicts collateral damage? ----------------
def run_encoder(cost, seed, d=16, n=4000, nprobe=50, steps=1500, hidden=64, lam_pres=1.0):
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    S = rand_spd(d, 0.05, 20.0, g); L = np.linalg.cholesky(S); Si = np.linalg.inv(S)
    opt = g.normal(size=d); delta = opt / np.sqrt(opt @ Si @ opt) * 3.0  # D = 3
    y = np.r_[np.zeros(n), np.ones(n)]
    X = g.normal(size=(2 * n, d)) @ L.T + y[:, None] * delta
    lda = Si @ delta
    wv = lda + 1.0 * np.linalg.norm(lda) * g.normal(size=d) / np.sqrt(d)  # not the optimal probe
    wv /= np.linalg.norm(wv)
    if dprime(wv, delta, S) < 0: wv = -wv
    # probe family: interpolate between w, lda and random directions at varied mixing
    U = []
    for k in range(nprobe):
        t = k / (nprobe - 1)
        v = (1 - t) * wv / np.linalg.norm(wv) + t * g.normal(size=d) / np.sqrt(d) * 2 + 0.5 * lda / np.linalg.norm(lda)
        U.append(v / np.linalg.norm(v))
    U = np.array(U)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    W = torch.tensor(wv, dtype=torch.float32); Sit = torch.tensor(Si, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d))
    torch.nn.init.zeros_(net[2].weight); torch.nn.init.zeros_(net[2].bias)
    opt_ = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        Z = Xt + net(Xt); s = Z @ W
        # evasion penalty: squared gap of class-mean scores of the fixed probe w
        pen = (s[yt == 1].mean() - s[yt == 0].mean()) ** 2
        Dz = Z - Xt
        pres = (Dz ** 2).sum(1).mean() if cost == "euclid" else ((Dz @ Sit) * Dz).sum(1).mean()
        loss = 10 * pen + lam_pres * pres
        opt_.zero_grad(); loss.backward(); opt_.step()
    Z = (Xt + net(Xt)).detach().numpy()
    before = np.array([roc_auc_score(y, X @ uu) for uu in U])
    after = np.array([roc_auc_score(y, Z @ uu) for uu in U])
    drop = before - after
    dw = dprime(wv, delta, S)
    cw = np.array([cosS(wv, uu, S) for uu in U]); craw = U @ wv
    du = np.array([dprime(uu, delta, S) for uu in U])
    pred_wh = norm.cdf(du / np.sqrt(2)) - norm.cdf((du - dw * cw) / np.sqrt(2))
    beta = (wv @ delta) / (wv @ wv)
    pred_eu = norm.cdf(du / np.sqrt(2)) - norm.cdf((du - beta * (U @ wv) / np.sqrt(np.einsum("ij,jk,ik->i", U, S, U))) / np.sqrt(2))
    def r2corr(p): return float(np.corrcoef(p, drop)[0, 1] ** 2)
    def r2raw(p): return float(1 - ((drop - p) ** 2).sum() / ((drop - drop.mean()) ** 2).sum())
    # refit: logistic regression on half, evaluated on other half
    from sklearn.linear_model import LogisticRegression
    idx = g.permutation(2 * n); tr, te = idx[: n], idx[n:]
    lr = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
    refit = roc_auc_score(y[te], lr.decision_function(Z[te]))
    # covariance change (is it a pure mean shift?)
    Sz = 0.5 * (np.cov(Z[y == 0].T) + np.cov(Z[y == 1].T))
    return dict(cost=cost, seed=seed, auroc_w_after=float(roc_auc_score(y, Z @ wv)),
                r2corr_whitened_formula=r2corr(pred_wh), r2_1mSSE_whitened_formula=r2raw(pred_wh),
                r2corr_euclid_formula=r2corr(pred_eu), r2_1mSSE_euclid_formula=r2raw(pred_eu),
                r2corr_raw_cos=r2corr(craw), r2corr_whitened_cos=r2corr(cw),
                refit_auroc=float(refit), refit_theory=float(norm.cdf(np.sqrt(max(0, 9 - (wv @ delta) ** 2 / (wv @ S @ wv))) / np.sqrt(2))),
                cov_rel_change=float(np.linalg.norm(Sz - S) / np.linalg.norm(S)),
                drop_range=[float(drop.min()), float(drop.max())])

import sys
if "--c5" in sys.argv:
    res["C5"] = [run_encoder(cost, s) for cost in ("mahal", "euclid") for s in (0, 1, 2)]
(OUT / ("derive_results_c5.json" if "--c5" in sys.argv else "derive_results.json")).write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
